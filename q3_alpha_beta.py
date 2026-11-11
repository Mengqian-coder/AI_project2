"""题目三：使用 Minimax 和 Alpha-beta 剪枝求博弈树的最佳得分。

根节点为 MAX，之后逐层交替；默认严格按 children 的从左到右顺序搜索。
本文件独立运行，仅使用 Python 标准库。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import inf


class TreeNode:
    """叶子用 value 保存整数得分，非叶子用 children 保存有序子节点。"""

    def __init__(
        self,
        value: int | None = None,
        children: list[TreeNode] | None = None,
    ) -> None:
        self.value = value
        self.children = children if children is not None else []


@dataclass
class SearchStats:
    """按递归访问次数计数；同一对象从不同路径访问时分别统计。"""

    visited: int = 0             # 进入递归函数的次数，含根节点和叶子。
    expanded: int = 0            # 实际访问的非叶子节点次数。
    leaves: int = 0              # 实际读取叶子得分的次数。
    cutoffs: int = 0             # 确实跳过至少一个剩余孩子的剪枝次数。
    skipped_children: int = 0    # 跳过的直接孩子数量，不枚举其后代。
    trace: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.visited = self.expanded = self.leaves = 0
        self.cutoffs = self.skipped_children = 0
        self.trace.clear()


def _record(stats: SearchStats | None, path: str, message: str) -> None:
    if stats is not None:
        indent = "  " * path.count("/")
        stats.trace.append(f"{indent}{path}: {message}")


def _leaf_value(node: TreeNode) -> int:
    if isinstance(node.value, bool) or not isinstance(node.value, int):
        raise ValueError("a leaf must have an integer value")
    return node.value


def minimax(root: TreeNode, *, stats: SearchStats | None = None) -> int:
    """普通 Minimax 对照：搜索全部叶子，返回根 MAX 的最佳得分。"""
    if stats is not None:
        stats.reset()

    def search(node: TreeNode, is_max: bool, path: str) -> int:
        if stats is not None:
            stats.visited += 1

        # 递归出口：叶子不再选择，直接把得分返回给父节点。
        if not node.children:
            score = _leaf_value(node)
            if stats is not None:
                stats.leaves += 1
            _record(stats, path, f"leaf -> {score}")
            return score

        if stats is not None:
            stats.expanded += 1
        role = "MAX" if is_max else "MIN"
        _record(stats, path, role)
        best = -inf if is_max else inf

        for index, child in enumerate(node.children):
            # 父调用在这里等待子调用返回；下一层切换玩家。
            score = search(child, not is_max, f"{path}/{index}")
            best = max(best, score) if is_max else min(best, score)
            _record(stats, path, f"child[{index}] -> {score}; best={best}")

        # 非叶子至少有一个孩子，best 已被更新为整数得分。
        return int(best)

    return search(root, True, "root")


def alpha_beta(root: TreeNode, *, stats: SearchStats | None = None) -> int:
    """返回根 MAX 的最佳得分；与 Minimax 相同，但跳过无关分支。

    alpha 是 MAX 已掌握的下界，beta 是 MIN 已掌握的上界，包含祖先传来的信息。
    输入应为有限无环结构；允许示例中的共享子节点，不修改节点或孩子顺序。
    被剪枝的内部调用可能只返回一个界；完整窗口下的根结果是精确值。
    """
    if stats is not None:
        stats.reset()

    def search(
        node: TreeNode,
        is_max: bool,
        alpha: int | float,
        beta: int | float,
        path: str,
    ) -> int:
        if stats is not None:
            stats.visited += 1

        if not node.children:
            score = _leaf_value(node)
            if stats is not None:
                stats.leaves += 1
            _record(stats, path, f"leaf -> {score}")
            return score

        if stats is not None:
            stats.expanded += 1
        role = "MAX" if is_max else "MIN"
        _record(stats, path, f"{role}, alpha={alpha}, beta={beta}")
        best = -inf if is_max else inf

        for index, child in enumerate(node.children):
            # 左到右搜索；把当前边界传给孩子，帮助孩子判断能否剪枝。
            score = search(child, not is_max, alpha, beta, f"{path}/{index}")

            if is_max:
                best = max(best, score)
                alpha = max(alpha, best)  # MAX 找到更高的保底分数。
            else:
                best = min(best, score)
                beta = min(beta, best)    # MIN 找到更低的分数上限。

            _record(
                stats, path,
                f"child[{index}] -> {score}; best={best}, alpha={alpha}, beta={beta}",
            )

            if alpha >= beta:
                # 只跳过本次调用中尚未搜索的孩子，随后返回父节点。
                remaining = len(node.children) - index - 1
                if remaining > 0:
                    if stats is not None:
                        stats.cutoffs += 1
                        stats.skipped_children += remaining
                    _record(
                        stats, path,
                        f"PRUNE: alpha={alpha} >= beta={beta}; "
                        f"skip children[{index + 1}:] ({remaining} direct children)",
                    )
                break

        # 必须返回本节点的 best，不能直接返回继承来的 alpha 或 beta。
        return int(best)

    return search(root, True, -inf, inf, "root")


def build_example_tree() -> TreeNode:
    """按题目构建样例；max2 同时作为左右两个 MIN 的孩子。"""
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def _copy_with_min_order_reversed(root: TreeNode) -> TreeNode:
    """顺序实验：复制为新树，仅反转 MIN 的孩子，不预先计算子树得分。"""
    def copy(node: TreeNode, is_max: bool) -> TreeNode:
        children = [copy(child, not is_max) for child in node.children]
        if not is_max:
            children.reverse()
        return TreeNode(value=node.value, children=children)

    return copy(root, True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 3: minimax and alpha-beta pruning")
    parser.add_argument("--trace", action="store_true", help="show recursive calls and pruning")
    parser.add_argument("--compare", action="store_true", help="compare minimax and alpha-beta")
    parser.add_argument(
        "--order-experiment", action="store_true",
        help="also search a copied tree with MIN children reversed",
    )
    args = parser.parse_args()

    root = build_example_tree()
    cases = [("Left-to-right", root)]
    if args.order_experiment:
        cases.append(("MIN children reversed", _copy_with_min_order_reversed(root)))
    algorithms = [("Alpha-beta", alpha_beta)]
    if args.compare:
        algorithms.insert(0, ("Minimax", minimax))

    for order_name, tree in cases:
        if args.order_experiment:
            print(f"Order: {order_name}")
        for name, algorithm in algorithms:
            stats = SearchStats() if args.trace or args.compare or args.order_experiment else None
            score = algorithm(tree, stats=stats)
            if args.compare or args.order_experiment:
                assert stats is not None
                print(
                    f"{name}: score={score}, visited={stats.visited}, "
                    f"expanded={stats.expanded}, leaves={stats.leaves}, "
                    f"cutoffs={stats.cutoffs}, skipped_children={stats.skipped_children}"
                )
            else:
                print(score)
            if args.trace:
                assert stats is not None
                for line in stats.trace:
                    print(line)


if __name__ == "__main__":
    main()
