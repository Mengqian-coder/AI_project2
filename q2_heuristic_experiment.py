"""题二启发函数对照：同一 A*，比较 h0 与有限步前瞻。

运行 python q2_heuristic_experiment.py；计数、质量诊断、计时相互独立。
结果写入 output/q2_heuristics。仅绘图需要 matplotlib。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush
import json
from math import exp, inf, isfinite, log
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter_ns

from q2_weighted import (
    CostGrid, DIRECTIONS, EXAMPLE_GOAL, EXAMPLE_GRID, EXAMPLE_START,
    Grid, Position, WeightedSearchStats, _prepare_costs, adjusted_manhattan,
    astar_weighted,
)


@dataclass
class HeuristicMetrics:
    queries: int = 0
    evaluations: int = 0
    edge_evaluations: int = 0
    cache_hits: int = 0


class LookaheadHeuristic:
    """H[k+1](u)=min(cost(v)+H[k](v)), H[k](goal)=0。

    depth=0 使用原公式，不缓存；前瞻版本按 (剩余深度, 坐标) 缓存。
    所有版本都按地图合法邻居估值，不依赖 A* 的 closed。
    instrument=False 关闭操作计数，用于计时；缓存策略保持相同。
    """

    def __init__(self, costs: CostGrid, goal: Position, min_cost: int,
                 depth: int, instrument: bool = True):
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.costs = costs
        self.goal = goal
        self.min_cost = min_cost
        self.depth = depth
        self.instrument = instrument
        self.metrics = HeuristicMetrics()
        self._cache: dict[tuple[int, Position], int | float] = {}
        self.rows, self.cols = len(costs), len(costs[0])

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def __call__(self, position: Position, target: Position) -> int | float:
        if target != self.goal:
            raise ValueError("heuristic was constructed for a different goal")
        if self.instrument:
            self.metrics.queries += 1
        return self._value(position, self.depth)

    def _value(self, position: Position, depth: int) -> int | float:
        key = (depth, position)
        if self.depth > 0 and key in self._cache:
            if self.instrument:
                self.metrics.cache_hits += 1
            return self._cache[key]
        if self.instrument:
            self.metrics.evaluations += 1
        if position == self.goal:
            result = 0
        elif depth == 0:
            result = adjusted_manhattan(position, self.goal, self.min_cost)
        else:
            row, col = position
            result = inf
            for dr, dc in DIRECTIONS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                enter_cost = self.costs[nr][nc]
                if enter_cost is None:
                    continue
                if self.instrument:
                    self.metrics.edge_evaluations += 1
                result = min(result, enter_cost + self._value((nr, nc), depth - 1))
        if self.depth > 0:
            self._cache[key] = result
        return result


def reverse_distances(costs: CostGrid, goal: Position) -> dict[Position, int]:
    """独立反向 Dijkstra：返回各状态到 goal 的真实最小代价。

    正向 u->v 收取 costs[v]，所以反向从 v 松弛 u 时仍收 costs[v]。
    不可达状态不放入返回字典；本 oracle 只用于验证，不参与 A* 估值。
    """
    rows, cols = len(costs), len(costs[0])
    distance = {goal: 0}
    queue = [(0, goal)]
    while queue:
        value, node = heappop(queue)
        if value != distance[node]:
            continue
        row, col = node
        edge_cost = costs[row][col]
        assert edge_cost is not None
        for dr, dc in DIRECTIONS:
            predecessor = (row + dr, col + dc)
            pr, pc = predecessor
            if not (0 <= pr < rows and 0 <= pc < cols) or costs[pr][pc] is None:
                continue
            candidate = value + edge_cost
            if candidate < distance.get(predecessor, inf):
                distance[predecessor] = candidate
                heappush(queue, (candidate, predecessor))
    return distance


@dataclass
class Case:
    name: str
    family: str
    grid: Grid
    start: Position
    goal: Position
    seed: int | None = None


def _connected(grid: Grid, start: Position, goal: Position) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for dr, dc in DIRECTIONS:
            neighbor = (node[0] + dr, node[1] + dc)
            row, col = neighbor
            if (0 <= row < len(grid) and 0 <= col < len(grid[0])
                    and grid[row][col] != "X" and neighbor not in seen):
                seen.add(neighbor)
                queue.append(neighbor)
    return False


def make_cases(seed: int = 20261028, sizes: tuple[int, ...] = (16, 32, 64),
               seeds_per_family: int = 5) -> list[Case]:
    cases = [
        Case("official", "official", EXAMPLE_GRID, EXAMPLE_START, EXAMPLE_GOAL),
        Case("forced_cost_corridor", "corridor", [["S", 8, 1, 8, 1, 8, 1, "E"]], (0, 0), (0, 7)),
        Case("unreachable", "unreachable", [["S", 1, "X", 1, "E"]], (0, 0), (0, 4)),
    ]
    for size in sizes:
        for direction in ("diagonal", "straight"):
            goal = (size - 1, size - 1) if direction == "diagonal" else (0, size - 1)
            grid = [[1] * size for _ in range(size)]
            grid[0][0], grid[goal[0]][goal[1]] = "S", "E"
            cases.append(Case(f"uniform_{direction}_{size}", "uniform", grid, (0, 0), goal))
        for family_index, family in enumerate(("weighted", "obstacles", "zero_cost")):
            for index in range(seeds_per_family):
                case_seed = seed + size * 10000 + family_index * 1000 + index
                rng = random.Random(case_seed)
                # 只保留起终点连通的随机图；不人为铺设廉价通道。
                for _ in range(100):
                    grid = []
                    for row in range(size):
                        values = []
                        for col in range(size):
                            cost = rng.choice((0, 0, 1, 1, 3, 8)) if family == "zero_cost" else rng.randint(1, 9)
                            obstacle_rate = 0.20 if family == "obstacles" else 0.10 if family == "zero_cost" else 0
                            values.append("X" if rng.random() < obstacle_rate else cost)
                        grid.append(values)
                    grid[0][0], grid[-1][-1] = "S", "E"
                    if _connected(grid, (0, 0), (size - 1, size - 1)):
                        break
                else:
                    raise RuntimeError("could not generate a connected case")
                cases.append(Case(f"{family}_{size}_{index}", family, grid,
                                  (0, 0), (size - 1, size - 1), case_seed))
    return cases


def run_search(case: Case, depth: int, instrument: bool = False) -> dict:
    # 为公平比较，h0 与全部前瞻方案都走相同准备流程。
    # 这里的转换以及 astar_weighted 内部的转换，均计入端到端耗时。
    costs, min_cost = _prepare_costs(case.start, case.goal, case.grid)
    heuristic = LookaheadHeuristic(costs, case.goal, min_cost, depth, instrument)
    stats = WeightedSearchStats()
    cost = astar_weighted(case.start, case.goal, case.grid, heuristic=heuristic, stats=stats)
    return {
        "cost": cost, "expanded": stats.expanded, "pushed": stats.pushed,
        "path_steps": len(stats.path) - 1 if stats.path else None,
        "heuristic": asdict(heuristic.metrics) if instrument else None,
        "cache_entries": heuristic.cache_size,
    }


def diagnose(case: Case, depths: tuple[int, ...]) -> dict:
    """独立对象遍历全图，检查性质并衡量估值强度；不复用到计时。"""
    costs, min_cost = _prepare_costs(case.start, case.goal, case.grid)
    exact = reverse_distances(costs, case.goal)
    nodes = [(r, c) for r, row in enumerate(costs) for c, cost in enumerate(row) if cost is not None]
    estimates = {}
    details = {}
    checked_edges = 0
    for depth in depths:
        heuristic = LookaheadHeuristic(costs, case.goal, min_cost, depth, instrument=False)
        values = {node: heuristic(node, case.goal) for node in nodes}
        estimates[depth] = values
        if values[case.goal] != 0:
            raise AssertionError("goal heuristic must be zero")
        for node in nodes:
            if values[node] > exact.get(node, inf):
                raise AssertionError(f"inadmissible: {case.name}, H{depth}, {node}")
            for dr, dc in DIRECTIONS:
                neighbor = (node[0] + dr, node[1] + dc)
                if neighbor not in values:
                    continue
                edge_cost = costs[neighbor[0]][neighbor[1]]
                if values[node] > edge_cost + values[neighbor]:
                    raise AssertionError(f"inconsistent: {case.name}, H{depth}, {node}")
                checked_edges += 1
        positive_nodes = [node for node, value in exact.items() if value > 0]
        details[str(depth)] = {
            "start_h": values[case.start] if isfinite(values[case.start]) else None,
            "mean_h_over_exact": statistics.fmean(values[node] / exact[node] for node in positive_nodes) if positive_nodes else None,
            "mean_underestimate": statistics.fmean(exact[node] - values[node] for node in positive_nodes) if positive_nodes else None,
        }
    for lower, higher in zip(depths, depths[1:]):
        if any(estimates[lower][node] > estimates[higher][node] for node in nodes):
            raise AssertionError("lookahead did not strengthen heuristic")
    return {
        "oracle_cost": exact.get(case.start, -1), "min_cost": min_cost,
        "passable_nodes": len(nodes), "goal_reachable_nodes": len(exact),
        "zero_exact_nodes": sum(value == 0 for value in exact.values()),
        "verified_states": len(nodes) * len(depths), "verified_edges": checked_edges,
        "heuristics": details,
    }


def _quantiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _geomean(values) -> float:
    return exp(statistics.fmean(log(value) for value in values))


def summarize(results: list[dict], depths: tuple[int, ...]) -> dict:
    groups = {"all": results}
    for case in results:
        groups.setdefault(case["family"], []).append(case)
        groups.setdefault(f"size_{case['rows']}x{case['cols']}", []).append(case)
    summary = {}
    for group, cases in groups.items():
        entries = {}
        for depth in depths:
            key = str(depth)
            ratios = [case["methods"][key]["median_ms"] / case["methods"]["0"]["median_ms"] for case in cases]
            expansion_ratios = [case["methods"][key]["expanded"] / case["methods"]["0"]["expanded"]
                                for case in cases if case["methods"]["0"]["expanded"]]
            quality = [case["diagnostics"]["heuristics"][key]["mean_h_over_exact"] for case in cases]
            quality = [value for value in quality if value is not None]
            entries[key] = {
                "cases": len(cases), "geomean_time_ratio": _geomean(ratios),
                "geomean_expansion_ratio": _geomean(expansion_ratios),
                "mean_quality_ratio": statistics.fmean(quality) if quality else None,
                "sum_median_ms": sum(case["methods"][key]["median_ms"] for case in cases),
                "total_expanded": sum(case["methods"][key]["expanded"] for case in cases),
                "wins": sum(case["fastest_depth"] == depth for case in cases),
                "within_5_percent_of_fastest": sum(case["methods"][key]["median_ms"] <= 1.05 * min(method["median_ms"] for method in case["methods"].values()) for case in cases),
            }
        summary[group] = entries
    return summary


def run_benchmark(cases: list[Case], depths: tuple[int, ...], repeats: int,
                  target_batch_ms: float, seed: int) -> list[dict]:
    results = []
    for case_index, case in enumerate(cases):
        diagnostic = diagnose(case, depths)
        methods = {}
        batches = {}
        for depth in depths:
            measured = run_search(case, depth, instrument=True)
            if measured["cost"] != diagnostic["oracle_cost"]:
                raise AssertionError(f"wrong path cost: {case.name}, H{depth}")
            # 先校准批量大小；小地图用多次全新查询均摊时钟粒度。
            begin = perf_counter_ns()
            result = run_search(case, depth, instrument=False)
            elapsed_ms = (perf_counter_ns() - begin) / 1e6
            if result["cost"] != measured["cost"]:
                raise AssertionError("timed and instrumented versions disagree")
            batches[depth] = max(1, min(200, int(target_batch_ms / max(elapsed_ms, 0.000001))))
            methods[str(depth)] = {**measured, "samples_ms": [], "batch_size": batches[depth]}
        rng = random.Random(seed + case_index)
        for _ in range(repeats):
            order = list(depths)
            rng.shuffle(order)
            for depth in order:
                begin = perf_counter_ns()
                for _ in range(batches[depth]):
                    run_search(case, depth, instrument=False)
                average_ms = (perf_counter_ns() - begin) / 1e6 / batches[depth]
                methods[str(depth)]["samples_ms"].append(average_ms)
        for method in methods.values():
            samples = method["samples_ms"]
            q1, q3 = _quantiles(samples)
            method.update(median_ms=statistics.median(samples), q1_ms=q1, q3_ms=q3, iqr_ms=q3 - q1)
        fastest = min(depths, key=lambda depth: methods[str(depth)]["median_ms"])
        result = {
            "name": case.name, "family": case.family, "rows": len(case.grid),
            "cols": len(case.grid[0]), "seed": case.seed,
            "diagnostics": diagnostic, "methods": methods, "fastest_depth": fastest,
        }
        results.append(result)
        print(f"[{case_index + 1:02d}/{len(cases)}] {case.name}: cost={diagnostic['oracle_cost']}, fastest=H{fastest}", flush=True)
    return results


def write_report(payload: dict, destination: Path) -> None:
    results, summary = payload["results"], payload["summary"]
    depths = tuple(payload["config"]["depths"])
    count = len(results)
    official = next(case for case in results if case["name"] == "official")
    total_states = sum(case["diagnostics"]["verified_states"] for case in results)
    total_edges = sum(case["diagnostics"]["verified_edges"] for case in results)
    best = min(depths, key=lambda depth: summary["all"][str(depth)]["geomean_time_ratio"])
    tightest = max(depths, key=lambda depth: summary["all"][str(depth)]["mean_quality_ratio"])
    strongest = summary["all"][str(tightest)]
    lines = [
        "# 题二：启发函数比较实验", "",
        f"本次共 {count} 张地图，比较 H0 至 H{max(depths)}；每张地图每种方法重复 {payload['config']['repeats']} 轮。",
        f"按每张地图相对 H0 的耗时比取几何平均，本机本次实验最快为 **H{best}**。结论仅适用于这里的地图与实现。", "",
        f"估值最紧的是 **H{tightest}**，但相对 H0 的扩展数比为 {strongest['geomean_expansion_ratio']:.3f}，总耗时比为 {strongest['geomean_time_ratio']:.3f}。",
        "因此评价应分开：估值准确度、搜索量、实际运行时间。本次选默认实现时优先考虑总时间，前瞻则作为有数据支持的拓展。", "",
        "## 怎样衡量启发函数", "",
        "| 指标 | 如何解释 | 优先级 |", "| --- | --- | --- |",
        "| 最优性与性质 | 与独立 Dijkstra 的代价一致；可采纳、一致且终点 h=0 | 必须满足 |",
        "| 单次完整查询时间 | 包含地图准备、新缓存、启发计算与 A*；报告中位数与四分位距 IQR | 实际速度的主要指标 |",
        "| 扩展与入堆数 | 说明 A* 少做了多少搜索工作 | 解释性能 |",
        "| 估值比 h/h* | 在可达且 h*>0 的节点上，越接近 1 表示估计越强 | 解释估计质量 |",
        "| 启发计算量与缓存条目 | 前瞻额外检查多少条边、保存多少个估值 | 解释时间与空间成本 |", "",
        "更强的启发函数并不保证总时间更少；FIFO 下最优代价平台上的平局也可能影响扩展数。", "",
        "h* 是独立 Dijkstra 求出的真实剩余最小代价。h/h* 越接近 1，低估越少，但不能据此单独选默认算法。",
        "耗时比 = 当前方法的逐图耗时中位数 / 同图 H0 的耗时中位数；例如 1.5 表示耗时增加 50%，0.8 表示减少 20%。", "",
        "## 方法与控制变量", "",
        "H0(n)=m*max(Manhattan(n,E)-1,0)；H(k+1)(n)=min_v(cost(v)+Hk(v))；所有层的 Hk(E)=0。",
        "所有方法调用现有 q2_weighted.astar_weighted，方向固定右/下/左/上，f 平局按入队先后。",
        "H0 保留原公式直接计算；H1 及以上用 (层数,坐标) 缓存，包括底层 H0。缓存优化属于前瞻实现的一部分。",
        "每次计时重新创建启发对象、缓存和搜索统计。计数、全图性质检查和计时分别使用独立对象；计时关闭启发操作计数。",
        "为控制接口开销，所有方法都先构造代价表，再调用现有 A*（其内部还会做同样解析）；两次解析都计入时间。",
        "因此本表比较的是统一实验适配器下的 H0 与前瞻方法，绝对时间与直接运行题二默认接口略有不同。",
        "每轮随机交错方法顺序；小图在一轮内批量运行多次全新查询，表中时间是每查询均值的跨轮中位数。",
        "批量大小先校准，预热/校准不计入结果；不禁用 Python 默认 GC，缓存清理位于被测函数返回前。",
        "反向 Dijkstra 仅作验证，不参与启发估值或计时；反向松弛 u<-v 加 costs[v]，保留入格收费的方向性。", "",
        "## 地图设置", "",
        f"规模：{payload['config']['sizes']}；每个随机类型、每个规模 {payload['config']['seeds_per_family']} 个固定种子。",
        "包括官方、强制经过昂贵格的走廊、无解、同权对角与直线终点、随机权、20%障碍随机权、10%障碍含零权。",
        "普通随机权为 1..9；零权地形从 [0,0,1,1,3,8] 抽取。随机图仅保留起终点连通的样本，不人为铺廉价路径。",
        "所有地图与种子在 maps.json 中保存；含零权地图是对官方正权样例的拓展。", "",
        "## 正确性", "",
        f"全部 {count * len(depths)} 次方法/地图计数运行与独立 Dijkstra 一致。",
        f"全图诊断检查了 {total_states:,} 个状态/方法与 {total_edges:,} 条有向边/方法，均满足可采纳性、一致性、终点值及逐层增强。",
        "估值比排除 h*=0 与不可达状态，避免除零或无穷大误导；详细数量见原始 JSON。", "",
        "## 官方样例", "",
        "| 方法 | 最优代价 | 扩展 | 入堆 | 中位数 ms | IQR ms | 启发额外邻边计算 | 缓存条目 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for depth in depths:
        method = official["methods"][str(depth)]
        lines.append(f"| H{depth} | {method['cost']} | {method['expanded']} | {method['pushed']} | {method['median_ms']:.4f} | {method['iqr_ms']:.4f} | {method['heuristic']['edge_evaluations']} | {method['cache_entries']} |")
    lines += ["", "## 所有地图的配对比较", "",
              "每张地图先除以该图 H0 的指标，再取几何平均，使不同大小的地图都各占一票。时间比小于 1 才表示快于 H0。",
              "估值比是各地图全图平均值的再平均；缓存条目衡量额外空间，未将其换算为字节。", "",
              "| 方法 | 耗时比/H0 | 扩展比/H0 | 平均 h/h* | 最快地图数 | 距最快不超过5%的地图数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for depth in depths:
        entry = summary["all"][str(depth)]
        lines.append(f"| H{depth} | {entry['geomean_time_ratio']:.3f} | {entry['geomean_expansion_ratio']:.3f} | {entry['mean_quality_ratio']:.3f} | {entry['wins']} | {entry['within_5_percent_of_fastest']} |")
    lines += ["", "## 按地形分类", "",
              "| 地图类型 | 张数 | H0 时间比 | H1 时间比 | H2 时间比 | H3 时间比 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for family in ("official", "corridor", "unreachable", "uniform", "weighted", "obstacles", "zero_cost"):
        entries = summary.get(family)
        if not entries:
            continue
        ratios = " | ".join(f"{entries[str(depth)]['geomean_time_ratio']:.3f}" for depth in depths)
        lines.append(f"| {family} | {entries['0']['cases']} | {ratios} |")
    lines += ["", "## 按方形地图规模分类", "",
              "| 规模 | 张数 | H0 时间比 | H1 时间比 | H2 时间比 | H3 时间比 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for size in payload["config"]["sizes"]:
        entries = summary[f"size_{size}x{size}"]
        ratios = " | ".join(f"{entries[str(depth)]['geomean_time_ratio']:.3f}" for depth in depths)
        lines.append(f"| {size}x{size} | {entries['0']['cases']} | {ratios} |")
    lines += ["", "## 如何解释结果", "",
              "1. 首先看代价是否正确，再看总时间；不要用更短的路径步数评价加权最优性。",
              "2. Hk 越深，安全下界越强，但递归计算和缓存查询也更多。若扩展减少而时间增加，说明额外估值成本超过节约的搜索成本。",
              "3. 同权地图的 H0 已是精确几何代价，前瞻通常无法改善；含零费用的局部循环可使浅前瞻也保持很低的估值。",
              "4. 地图越大不代表浅前瞻一定有用；只有有限步地形信息能排除足够多的候选节点，额外成本才可能回本。",
              "5. 昂贵走廊能体现估值提升，但没有分支可排除；即使估值准确，也未必减少扩展。",
              "6. 含零费用的往返循环可让浅前瞻一直选择局部免费移动，难以感知远处的必付代价；有限前瞻并非完整求解。",
              "7. 几个百分点的微小时间差应结合 IQR 看待；这里是单机、单进程实验，不是普遍最优证明。", "",
              "![启发强度、扩展数与时间的对照](comparison.png)", "",
              "## 复现", "", "```powershell", "python q2_heuristic_experiment.py",
              "python -m unittest -v test_q2_heuristic_experiment.py", "```", "",
              "快速试跑：`python q2_heuristic_experiment.py --sizes 8 16 --seeds-per-family 1 --repeats 3 --output tmp/q2_pilot`。",
              "results.json 包含全部逐轮计时、操作计数、诊断和环境；maps.json 保存完整输入。", "",
              f"环境：Python {payload['environment']['python']}；{payload['environment']['platform']}；处理器 {payload['environment']['processor']}。",
              "计时函数 perf_counter_ns；随机种子和参数见 results.json。",
              ""]
    destination.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(payload: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    depths = payload["config"]["depths"]
    summary = payload["summary"]
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B07AA1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    entries = summary["all"]
    labels = [f"H{depth}" for depth in depths]
    quality = [entries[str(depth)]["mean_quality_ratio"] for depth in depths]
    axes[0].bar(labels, quality, color=colors)
    axes[0].set(title="Estimate strength (higher is tighter)", ylabel="Mean h / exact remaining cost", ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    xs = list(range(len(depths)))
    axes[1].bar([x - .18 for x in xs], [entries[str(depth)]["geomean_expansion_ratio"] for depth in depths], .36, label="A* expansions", color="#4C78A8")
    axes[1].bar([x + .18 for x in xs], [entries[str(depth)]["geomean_time_ratio"] for depth in depths], .36, label="Total query time", color="#F28E2B")
    axes[1].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xticks=xs, xticklabels=labels, title="Less search can still take more time", ylabel="Geometric mean ratio to H0")
    axes[1].legend(frameon=False, fontsize=9)
    families = [family for family in ("uniform", "weighted", "obstacles", "zero_cost") if family in summary]
    for depth, color in zip(depths[1:], colors[1:]):
        axes[2].plot(families, [summary[family][str(depth)]["geomean_time_ratio"] for family in families], marker="o", label=f"H{depth}", color=color)
    axes[2].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[2].set(title="Timing depends on terrain", ylabel="Time ratio to H0 (lower is faster)")
    axes[2].tick_params(axis="x", labelrotation=15)
    axes[2].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle(f"Weighted A*: H0 versus finite lookahead | {len(payload['results'])} maps, fresh caches per query", fontsize=14)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seeds-per-family", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--target-batch-ms", type=float, default=8)
    parser.add_argument("--output", type=Path, default=Path("output/q2_heuristics"))
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.seeds_per_family < 1 or args.repeats < 3 or args.target_batch_ms <= 0:
        parser.error("sizes >= 2, seeds-per-family >= 1, repeats >= 3, target-batch-ms > 0 required")
    depths = (0, 1, 2, 3)
    cases = make_cases(args.seed, tuple(args.sizes), args.seeds_per_family)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "maps.json").write_text(json.dumps([asdict(case) for case in cases], indent=2, ensure_ascii=False), encoding="utf-8")
    results = run_benchmark(cases, depths, args.repeats, args.target_batch_ms, args.seed)
    payload = {
        "config": {"sizes": args.sizes, "seeds_per_family": args.seeds_per_family,
                   "repeats": args.repeats, "seed": args.seed, "depths": depths,
                   "target_batch_ms": args.target_batch_ms},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "processor": platform.processor(), "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat()},
        "results": results, "summary": summarize(results, depths),
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_report(payload, args.output / "report.md")
    plot_summary(payload, args.output / "comparison.png")
    print(f"Saved results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
