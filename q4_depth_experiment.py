"""探索：H=2 的首步何时偏离完整四次行动目标。

固定官方地图，分别做单变量扫描、T2/T3 二维扫描和随机价值采样。
首步损失 R=max_a Q4(a)-Q4(a_H2)；固定该首步后，双方按完整最优策略应对。
这衡量单次首步的损失，不是双方整局反复使用 H=2 的对局成绩。
另比较四种根节点平局规则；只在 H=2 的最高分动作中选择，Q4 仅用于事后评价。
运行后写入 output/q4_depth；除绘图需 matplotlib 外，仅依赖标准库。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import csv
from dataclasses import asdict, dataclass, field
from itertools import permutations
import json
from pathlib import Path
import platform
import random
import statistics

from q4_treasure import (
    EXAMPLE_GRID, POINTS, TREASURE_VALUES, Distances, GameState, SearchStats,
    alpha_beta, apply_action, compute_distances, first_move_values, minimax,
)

NAMES = tuple(TREASURE_VALUES)
TIE_LABELS = {
    "all_correct": "所有浅层最优动作都正确",
    "mixed": "并列选择敏感",
    "all_wrong": "所有浅层最优动作都次优",
}
TIE_POLICIES = {
    "default": "编号小优先",
    "reverse": "编号大优先",
    "nearest": "移动代价最小优先",
    "gain": "即时净收益最大优先",
}


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    family: str
    values: dict[str, int]
    parameters: dict[str, int] = field(default_factory=dict)
    seed: int | None = None


def make_cases(
    value_min: int = 1, value_max: int = 20,
    random_cases: int = 1000, seed: int = 20261103,
) -> list[ExperimentCase]:
    if value_min < 0 or value_max < value_min or random_cases < 0:
        raise ValueError("require 0 <= value_min <= value_max and random_cases >= 0")
    cases = [ExperimentCase("official", "official", dict(TREASURE_VALUES))]
    for target in NAMES:
        for value in range(value_min, value_max + 1):
            values = dict(TREASURE_VALUES)
            values[target] = value
            cases.append(ExperimentCase(f"{target}_{value}", f"scan_{target}", values, {target: value}))
    for t2 in range(value_min, value_max + 1):
        for t3 in range(value_min, value_max + 1):
            values = {**TREASURE_VALUES, "T2": t2, "T3": t3}
            cases.append(ExperimentCase(f"T2_{t2}_T3_{t3}", "grid_T2_T3", values, {"T2": t2, "T3": t3}))
    rng = random.Random(seed)
    for index in range(random_cases):
        values = {target: rng.randint(value_min, value_max) for target in NAMES}
        cases.append(ExperimentCase(f"random_{index:04d}", "random", values, {"index": index}, seed))
    return cases


def oracle_action_values(
    distances: Distances, values: Mapping[str, int], depth: int,
) -> dict[str, int]:
    """独立 oracle：枚举行动排列、直接算叶子 U，再自底向上折叠。

    不调用 q4 的状态转移或搜索函数。用于本实验的全可达地图；根为 A。
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    horizon = min(depth, len(values))
    if horizon == 0:
        return {}
    scores: dict[tuple[str, ...], int] = {}
    for line in permutations(values, horizon):
        pos_a, pos_b, utility = "A", "B", 0
        for ply, target in enumerate(line):
            source = pos_a if ply % 2 == 0 else pos_b
            distance = distances[source][target]
            if distance < 0:
                raise ValueError("the depth experiment requires reachable targets")
            gain = values[target] - distance
            if ply % 2 == 0:
                utility += gain
                pos_a = target
            else:
                utility -= gain
                pos_b = target
        scores[line] = utility

    # 长度 k 的前缀轮到第 k 次行动：偶数 k 为 A，奇数 k 为 B。
    for length in range(horizon - 1, 0, -1):
        children: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for line, score in scores.items():
            children[line[:length]].append(score)
        choose = max if length % 2 == 0 else min
        scores = {prefix: choose(results) for prefix, results in children.items()}
    return {target: scores[(target,)] for target in values}


def select_tied_action(
    action_values: Mapping[str, int], distances: Distances,
    values: Mapping[str, int], policy: str = "default",
) -> str:
    """只对 A 的 H=2 根节点最优动作破平局，不改变评分或递归顺序。

    本函数不接收 Q4，避免把完整搜索的信息用于浅层选择。
    nearest/gain 再次同分时，均按 T1、T2、T3、T4 选择第一个。
    """
    if policy not in TIE_POLICIES:
        raise ValueError(f"unknown tie policy: {policy}")
    if set(action_values) != set(NAMES):
        raise ValueError("tie policies require all four root action values")
    best = max(action_values.values())
    candidates = [target for target in NAMES if action_values[target] == best]
    if policy == "reverse":
        return candidates[-1]
    if policy == "nearest":
        return min(candidates, key=lambda target: distances["A"][target])
    if policy == "gain":
        return max(candidates, key=lambda target: values[target] - distances["A"][target])
    return candidates[0]


def evaluate_tie_policies(
    shallow_values: Mapping[str, int], full_values: Mapping[str, int],
    distances: Distances, values: Mapping[str, int],
) -> dict[str, dict]:
    """用统一终局目标评价各规则；完整分值只在选定首步之后读取。"""
    selected = {
        policy: select_tied_action(shallow_values, distances, values, policy)
        for policy in TIE_POLICIES
    }
    full_best = max(full_values.values())
    results = {}
    for policy, target in selected.items():
        regret = full_best - full_values[target]
        results[policy] = {
            "chosen_target": target,
            "predicted_value": shallow_values[target],
            "full_value_after_chosen": full_values[target],
            "regret": regret,
            "mistake": regret > 0,
            "changed_first_move": target != selected["default"],
        }
    return results


def analyze_case(
    case: ExperimentCase, distances: Distances, *, verify: bool = True,
) -> dict:
    """计算首步损失，并区分稳健正确、并列敏感、内在截断失误。"""
    if set(case.values) != set(NAMES):
        raise ValueError("this experiment requires exactly T1, T2, T3 and T4")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in case.values.values()):
        raise ValueError("treasure values must be non-negative integers")
    state = GameState()
    analyses = {}
    for label, depth in (("shallow", 2), ("full", 4)):
        stats = SearchStats()
        result = minimax(state, distances, depth, values=case.values, stats=stats)
        action_values = first_move_values(state, distances, depth, values=case.values)
        if len(action_values) != len(NAMES):
            raise ValueError("the depth experiment requires reachable targets")
        optimal = [target for target in NAMES if action_values[target] == result.best_value]
        if verify:
            assert action_values == oracle_action_values(distances, case.values, depth), case.name
            pruned = alpha_beta(state, distances, depth, values=case.values)
            assert pruned == result, (case.name, depth, "Minimax/Alpha-beta mismatch")
        analyses[label] = {
            "depth": depth,
            "chosen_target": result.best_target,
            "predicted_value": result.best_value,
            "optimal_targets": optimal,
            "action_values": action_values,
            "line": list(result.line),
            "visited": stats.visited,
        }

    shallow, full = analyses["shallow"], analyses["full"]
    chosen = shallow["chosen_target"]
    full_value_after_chosen = full["action_values"][chosen]
    # 固定浅搜索首步，之后从该子状态完整搜索剩余三次行动。
    child = apply_action(state, chosen, distances, case.values)
    continuation = minimax(child, distances, 3, values=case.values)
    assert continuation.best_value == full_value_after_chosen
    shallow["full_value_after_chosen"] = full_value_after_chosen
    shallow["full_line_after_chosen"] = [chosen, *continuation.line]
    regret = full["predicted_value"] - full_value_after_chosen
    tie_regrets = [full["predicted_value"] - full["action_values"][t] for t in shallow["optimal_targets"]]
    best_tie, worst_tie = min(tie_regrets), max(tie_regrets)
    tie_class = "all_correct" if worst_tie == 0 else "all_wrong" if best_tie > 0 else "mixed"
    assert regret >= 0
    return {
        **asdict(case), **analyses,
        "regret": regret,
        "mistake": regret > 0,
        "different_first_move": chosen != full["chosen_target"],
        "best_tie_regret": best_tie,
        "worst_tie_regret": worst_tie,
        "tie_class": tie_class,
        "tie_policies": evaluate_tie_policies(
            shallow["action_values"], full["action_values"], distances, case.values,
        ),
    }


def summarize(records: list[dict]) -> dict[str, dict]:
    """按输入设计分别统计；不混合扫描、网格和随机组计算总体发生率。"""
    families: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        families[row["family"]].append(row)
    summary = {}
    for family, rows in families.items():
        summary[family] = {
            "cases": len(rows),
            "distinct_value_vectors": len({tuple(row["values"][t] for t in NAMES) for row in rows}),
            "mistakes": sum(row["mistake"] for row in rows),
            "mistake_fraction": statistics.mean(row["mistake"] for row in rows),
            "mean_regret": statistics.mean(row["regret"] for row in rows),
            "max_regret": max(row["regret"] for row in rows),
            "different_but_optimal": sum(row["different_first_move"] and not row["mistake"] for row in rows),
            "tie_classes": {name: sum(row["tie_class"] == name for row in rows) for name in TIE_LABELS},
            "tie_policies": {
                policy: {
                    "mistakes": sum(row["tie_policies"][policy]["mistake"] for row in rows),
                    "mistake_fraction": statistics.mean(row["tie_policies"][policy]["mistake"] for row in rows),
                    "mean_regret": statistics.mean(row["tie_policies"][policy]["regret"] for row in rows),
                    "max_regret": max(row["tie_policies"][policy]["regret"] for row in rows),
                    "changed_first_moves": sum(row["tie_policies"][policy]["changed_first_move"] for row in rows),
                    "repaired_mistakes": sum(row["mistake"] and not row["tie_policies"][policy]["mistake"] for row in rows),
                    "introduced_mistakes": sum(not row["mistake"] and row["tie_policies"][policy]["mistake"] for row in rows),
                }
                for policy in TIE_POLICIES
            },
        }
    return summary


def _ranges(values: list[int]) -> str:
    if not values:
        return "无"
    groups = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            groups.append(str(start) if start == end else f"{start}–{end}")
            start = end = value
    groups.append(str(start) if start == end else f"{start}–{end}")
    return "、".join(groups)


def _write_csv(path: Path, records: list[dict]) -> None:
    fields = ["name", "family", *NAMES, "shallow_target", "full_target", "shallow_optima", "full_optima",
              "predicted_U2", "full_optimal_U4", "chosen_first_U4", "regret", "mistake", "tie_class",
              "best_tie_regret", "worst_tie_regret", "different_first_move"]
    fields += [f"Q{depth}_{t}" for depth in (2, 4) for t in NAMES]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in records:
            shallow, full = row["shallow"], row["full"]
            data = {
                "name": row["name"], "family": row["family"], **row["values"],
                "shallow_target": shallow["chosen_target"], "full_target": full["chosen_target"],
                "shallow_optima": "|".join(shallow["optimal_targets"]),
                "full_optima": "|".join(full["optimal_targets"]),
                "predicted_U2": shallow["predicted_value"], "full_optimal_U4": full["predicted_value"],
                "chosen_first_U4": shallow["full_value_after_chosen"],
                **{key: row[key] for key in ("regret", "mistake", "tie_class", "best_tie_regret",
                                             "worst_tie_regret", "different_first_move")},
            }
            for label, depth in (("shallow", 2), ("full", 4)):
                data.update({f"Q{depth}_{t}": row[label]["action_values"][t] for t in NAMES})
            writer.writerow(data)


def _write_tie_policy_csv(path: Path, records: list[dict]) -> None:
    """一行对应一个案例的一种平局规则，方便复核报告 4.5 节。"""
    fields = ["name", "family", *NAMES, "policy", "chosen_target", "predicted_U2",
              "chosen_first_U4", "full_optimal_U4", "regret", "mistake", "changed_first_move"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in records:
            for policy, result in row["tie_policies"].items():
                writer.writerow({
                    "name": row["name"], "family": row["family"], **row["values"],
                    "policy": policy, "chosen_target": result["chosen_target"],
                    "predicted_U2": result["predicted_value"],
                    "chosen_first_U4": result["full_value_after_chosen"],
                    "full_optimal_U4": row["full"]["predicted_value"],
                    **{key: result[key] for key in ("regret", "mistake", "changed_first_move")},
                })


def _make_figures(output: Path, records: list[dict], value_min: int, value_max: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for target, ax in zip(NAMES, axes.flat):
        rows = [row for row in records if row["family"] == f"scan_{target}"]
        x = [row["parameters"][target] for row in rows]
        ax.fill_between(x, [r["best_tie_regret"] for r in rows], [r["worst_tie_regret"] for r in rows],
                        color="#7b9bb8", alpha=0.25, label="Range across H=2 optimal actions")
        ax.plot(x, [r["regret"] for r in rows], "o-", color="#235789", markersize=3,
                label="Fixed tie rule: first ID")
        ax.axhline(0, color="#999999", linewidth=0.8)
        if value_min <= TREASURE_VALUES[target] <= value_max:
            ax.axvline(TREASURE_VALUES[target], color="#999999", linestyle=":", linewidth=1,
                       label="Official value")
        ax.set(title=f"Vary {target}; other values fixed", xlabel=f"{target} value", ylabel="First-move regret")
        ax.set_ylim(bottom=-0.3)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("H=2 decisions evaluated by complete H=4 play")
    fig.savefig(str(output / "value_scans.png"), dpi=180)
    plt.close(fig)

    grid_rows = {(r["parameters"]["T2"], r["parameters"]["T3"]): r
                 for r in records if r["family"] == "grid_T2_T3"}
    levels = list(range(value_min, value_max + 1))
    regrets = [[grid_rows[t2, t3]["regret"] for t2 in levels] for t3 in levels]
    class_codes = {"all_correct": 0, "mixed": 1, "all_wrong": 2}
    classes = [[class_codes[grid_rows[t2, t3]["tie_class"]] for t2 in levels] for t3 in levels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    extent = (value_min - 0.5, value_max + 0.5, value_min - 0.5, value_max + 0.5)
    image = axes[0].imshow(regrets, origin="lower", extent=extent, interpolation="nearest", cmap="viridis", vmin=0)
    fig.colorbar(image, ax=axes[0], label="First-move regret")
    colors = ["#dce9df", "#edc97f", "#c66959"]
    axes[1].imshow(classes, origin="lower", extent=extent, interpolation="nearest",
                   cmap=ListedColormap(colors), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    for ax, title in zip(axes, ("Selected H=2 action", "All H=2 optimal actions")):
        ax.set(title=title, xlabel="T2 value", ylabel="T3 value")
        if value_min <= 6 <= value_max and value_min <= 7 <= value_max:
            ax.plot(6, 7, marker="*", markersize=12, color="white", markeredgecolor="black")
    fig.legend([Patch(facecolor=c) for c in colors], ["All correct", "Tie-sensitive", "All suboptimal"],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Vary T2 and T3; T1=9, T4=7; star = official setting")
    fig.savefig(str(output / "decision_map.png"), dpi=180)
    plt.close(fig)


def _write_report(output: Path, records: list[dict], summary: dict, config: dict, plots: bool) -> None:
    official = next(row for row in records if row["family"] == "official")
    lines = [
        "# 探索实验：浅搜索何时选错首步", "",
        "固定第四题地图与移动规则，研究 H=2 的首步是否符合完整 H=4 的最终收益差目标。",
        "H=2 对自己的截断评价仍然正确；这里的“选错”特指相对完整四次行动目标的次优首步。", "",
        "## 评价方法", "",
        "Q2(a)：固定首步为 a，在总计两次行动后评价当前 U；Q4(a)：固定首步为 a，之后双方完整最优应对直到四次行动结束。",
        "V4*=max_a Q4(a)，实际首步 a2 由 H=2、固定 T1→T2→T3→T4 平局规则选出。", "",
        "```text", "首步损失 regret = V4* - Q4(a2)", "```", "",
        "regret>0 才判为次优；首步不同但 regret=0 仍然正确。不能用 U4-U2 当损失，因为两者评价时点不同。",
        "后续采用完整最优应对来隔离首步的影响；本实验不等同于双方整局持续使用 H=2 的表现。", "",
        "设 A2、A4 为两种深度的全部最优动作集合，进一步区分：", "",
        "- 全部正确：A2 是 A4 的子集，浅层任意平局选择均正确。",
        "- 并列敏感：两集合有交集，但 A2 也包含完整目标下的次优动作。",
        "- 全部次优：两集合没有交集，任何浅层最优动作都会损失最终收益差。", "",
        "best_tie_regret / worst_tie_regret 给出 A2 内最小 / 最大损失；不假设随机打破平局。", "",
        "## 输入设计与验证", "",
        f"扫描价值范围 {config['value_min']}–{config['value_max']}，其他宝藏保持官方价值；地图、位置和 A* 距离矩阵全部固定。",
        "四条单变量扫描用于定位翻转边界；T2/T3 二维全组合用于观察交互；随机组的四个价值分别独立均匀有放回采样。",
        f"随机样本 {config['random_cases']} 个，种子 {config['seed']}。各组分别统计；它们会重复包含相同价值向量，不汇总一个总体错误率。",
        "比例只描述本地图、当前价值范围和输入设计；不能推广为任意地图或实际游戏中的失误概率。", "",
        "先用现有 A* 计算距离；每个案例用独立完整窗口 Minimax 获得全部 Q2/Q4，并核对 Alpha-beta 的根结果。",
        "独立 oracle 枚举 12 条两步排列或 24 条四步排列，直接累加双方费用与价值，再按各层 MAX/MIN 自底向上求值。",
        "正式搜索与 oracle 对所有案例的全部首步分值均一致，首步后悔值非负。本实验不作运行时间比较。", "",
        "## 官方案例", "",
        "| 首步 | Q2 | Q4 |", "| --- | ---: | ---: |",
    ]
    for target in NAMES:
        lines.append(f"| {target} | {official['shallow']['action_values'][target]} | {official['full']['action_values'][target]} |")
    lines += ["", f"H=2 选择 {official['shallow']['chosen_target']}，其完整对弈价值为 {official['shallow']['full_value_after_chosen']}；"
              f"完整最优值为 {official['full']['predicted_value']}，首步损失 **{official['regret']}**。", "",
              "## 各组结果", "",
              "| 输入组 | 样本数 | 次优首步数 | 平均损失 | 最大损失 | 全部正确 | 并列敏感 | 全部次优 | 首步不同但正确 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family, stats in summary.items():
        classes = stats["tie_classes"]
        lines.append(f"| {family} | {stats['cases']} | {stats['mistakes']} | {stats['mean_regret']:.3f} | "
                     f"{stats['max_regret']} | {classes['all_correct']} | {classes['mixed']} | {classes['all_wrong']} | {stats['different_but_optimal']} |")
    lines += ["", "单变量扫描中，固定平局规则产生正损失的价值范围：", "",
              "| 改变的价值 | 实际首步次优 | 所有浅层最优动作都次优 | 并列选择敏感 |",
              "| --- | --- | --- | --- |"]
    for target in NAMES:
        rows = [r for r in records if r["family"] == f"scan_{target}"]
        get_range = lambda pred: _ranges([r["parameters"][target] for r in rows if pred(r)])
        lines.append(f"| {target} | {get_range(lambda r: r['mistake'])} | "
                     f"{get_range(lambda r: r['tie_class'] == 'all_wrong')} | {get_range(lambda r: r['tie_class'] == 'mixed')} |")

    lines += ["", "## 决策边界与并列反例", "",
              "T2 单变量扫描中，相同的 Q4 评价标准揭示浅层与完整搜索的决策边界：", "",
              "| T2 价值 | H2 最优集合 | H4 最优集合 | 实际首步损失 | 类型 |",
              "| ---: | --- | --- | ---: | --- |"]
    for row in records:
        if row["family"] == "scan_T2" and 4 <= row["parameters"]["T2"] <= 8:
            lines.append(f"| {row['parameters']['T2']} | {','.join(row['shallow']['optimal_targets'])} | "
                         f"{','.join(row['full']['optimal_targets'])} | {row['regret']} | {TIE_LABELS[row['tie_class']]} |")
    if config["value_min"] <= 4 and config["value_max"] >= 8:
        lines += ["", "### 为什么浅层在 5 翻转，完整搜索在 7 翻转", "",
                  "令 T2 价值为 x，其余价值为 (T1,T3,T4)=(9,7,7)。仅在 4≤x≤8 这个局部区间，有：", "",
                  "```text", "Q2(T2) = (x-4) - (7-4) = x-7",
                  "Q2(T3) = (7-8) - (9-8) = -2", "",
                  "Q4(T2) = x-5", "Q4(T3) = min(x-3, 9-x)", "```", "",
                  "浅层中，A 先取 T2 后，B 的最佳即时选择为 T3；A 先取 T3 后，B 的最佳即时选择为 T1。"
                  "因此 x=5 时两个首步打平，x>5 时浅层更偏好 T2。", "",
                  "完整应对下，固定 T2 的一条最优序列为 T2→T3→T1→T4，收益差为 x-5。"
                  "固定 T3 时，4≤x≤6 的一条最优序列为 T3→T1→T2→T4，收益差为 x-3；"
                  "6≤x≤8 时可选 T3→T2→T1→T4，收益差为 9-x。"
                  "x=6 时这两个 B 应对同分；之后 B 更愿意先抢 T2，所以 T3 的完整分值开始下降。", "",
                  "这两个首步也是该局部区间内的完整最优候选。完整搜索要到 x=7 才认为取 T2 与取 T3 一样好，"
                  "这解释了浅层在 x=5、6 提前改选所产生的损失。这些表达式仅用于上述局部区间。"]
    counter = next((r for r in records if r["family"] == "scan_T4" and r["parameters"]["T4"] == 15), None)
    if counter:
        lines += ["", f"仅把 T4 价值改为 15：H2 选 {counter['shallow']['chosen_target']}，H4 固定平局规则选 "
                  f"{counter['full']['chosen_target']}，但完整最优集合为 {counter['full']['optimal_targets']}，损失为 {counter['regret']}。"
                  "因此不能把“首步不同”直接计为错误。"]
    lines += ["", "## 根节点平局策略比较", "",
              "T2=5、T3=7 的阈值案例中，H2 认为 T2、T3 同分，但完整目标下只有 T3 最优。"
              "因此进一步比较：保持 H2 搜索与评分不变，只改变 A 首步同分时的选择。", "",
              "default 按编号从小到大，reverse 按编号从大到小；nearest 优先选从 A 出发移动代价最小的候选，"
              "gain 优先选宝藏价值减该移动代价最大的候选。nearest/gain 再次同分时按编号从小到大。"
              "这些规则只读取 H2 分值、宝藏价值与距离，Q4 只用于选定后的评价，不参与选动作。", "",
              "以下为固定 T1=9、T4=7 的 T2/T3 二维组合结果（其他输入组见 summary.json）：", "",
              "| 平局规则 | 次优次数 | 平均损失 | 修复默认错误 | 新增错误 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for policy, stats in summary["grid_T2_T3"]["tie_policies"].items():
        lines.append(f"| {TIE_POLICIES[policy]} | {stats['mistakes']} | {stats['mean_regret']:.3f} | "
                     f"{stats['repaired_mistakes']} | {stats['introduced_mistakes']} |")
    lines += ["", "改变平局规则可以缓解部分浅搜索失误，但不能选回已被 H2 排除的动作。"
              "局部案例的较好选择也未必构成总体更好的规则，例如 nearest 仍会在 T2=5、T3=7 时选 T2。"
              "上述结果只适用于当前地图与取值范围，不保证某种规则在其他场景中最优。"]
    lines += ["", "## 为什么会失误", "",
              "H2 只计算双方第一笔收益。第一步同时改变了玩家后续出发点、剩余宝藏和争夺机会，而这些影响要到第三、第四次行动才计入当前评价。",
              "所以某个宝藏的即时价值提高后，浅层可能提前把它排为最优；完整搜索仍可能更看重后续位置与对手剩余选择。",
              "并列敏感案例允许通过选择另一个 H2 最优动作避开损失；全部次优案例则无法仅靠改变浅层平局规则补救。",
              "该分析针对当前 U 截断评价。叶点评价函数、滚动搜索策略、地图布局改变后，需要重新验证。", ""]
    if plots:
        lines += ["## 对照图", "", "浅色带表示 H2 全部最优动作的损失范围，折线是固定平局规则实际选出的动作损失。", "",
                  "![四条单变量扫描](value_scans.png)", "",
                  "左图显示实际动作损失，右图区分全部正确、并列敏感、全部次优；星号为官方设置。", "",
                  "![T2/T3 价值二维实验](decision_map.png)", ""]
    lines += ["## 复现与原始数据", "", "```powershell", "python q4_depth_experiment.py",
              "python -m unittest -v test_q4_depth_experiment.py", "```", "",
              "`cases.json` 保存每个输入及分组；`results.json` 保存全部分值、最优集合、行动序列和统计；",
              "`results.csv` 便于逐行筛选；`tie_policies.csv` 每行对应一个案例的一种平局规则；"
              "`summary.json` 保存各组汇总及 tie_policies 策略统计。绘图需 matplotlib；`--no-plots` 仅生成文本与数据。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="H=2 decision loss and four root tie-breaking policies")
    parser.add_argument("--value-min", type=int, default=1)
    parser.add_argument("--value-max", type=int, default=20)
    parser.add_argument("--random-cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path, default=Path("output/q4_depth"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    cases = make_cases(args.value_min, args.value_max, args.random_cases, args.seed)
    distances = compute_distances()
    records = [analyze_case(case, distances) for case in cases]
    summary = summarize(records)
    config = {"value_min": args.value_min, "value_max": args.value_max,
              "random_cases": args.random_cases, "seed": args.seed,
              "shallow_depth": 2, "full_depth": 4, "tie_order": list(NAMES),
              "tie_policies": TIE_POLICIES, "tie_policy_scope": "A's H=2 root action only",
              "secondary_tie_order": list(NAMES),
              "python": platform.python_version(), "platform": platform.platform()}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "cases.json": {"config": config, "grid": EXAMPLE_GRID, "points": POINTS,
                       "distances": distances, "cases": [asdict(case) for case in cases]},
        "results.json": {"config": config, "records": records},
        "summary.json": summary,
    }
    for filename, payload in payloads.items():
        (output / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output / "results.csv", records)
    _write_tie_policy_csv(output / "tie_policies.csv", records)
    if not args.no_plots:
        _make_figures(output, records, args.value_min, args.value_max)
    _write_report(output, records, summary, config, not args.no_plots)
    print(f"Validated {len(records)} cases; output: {output}")
    for family, stats in summary.items():
        print(f"{family}: suboptimal={stats['mistakes']}/{stats['cases']}, "
              f"mean_regret={stats['mean_regret']:.3f}, max_regret={stats['max_regret']}")
    print("Root tie-policy comparison on grid_T2_T3:")
    for policy, stats in summary["grid_T2_T3"]["tie_policies"].items():
        print(f"  {policy}: suboptimal={stats['mistakes']}/{summary['grid_T2_T3']['cases']}, "
              f"mean_regret={stats['mean_regret']:.3f}, repaired={stats['repaired_mistakes']}, "
              f"introduced={stats['introduced_mistakes']}")


if __name__ == "__main__":
    main()
