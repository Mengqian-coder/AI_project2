"""题目四：A* 最短路 + Minimax / Alpha-beta 宝藏争夺。

本文件独立于题一至题三，仅依赖标准库。进入 A、B、宝藏格的代价均为 1。
H 以总行动次数计数：H=2 搜索 A -> B，H=4 搜索 A -> B -> A -> B。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from itertools import count
from math import inf

Position = tuple[int, int]
Grid = Sequence[Sequence[int | str]]
Distances = Mapping[str, Mapping[str, int]]
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

EXAMPLE_GRID = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
POINTS = {
    "A": (0, 0), "B": (6, 6),
    "T1": (0, 6), "T2": (2, 2), "T3": (4, 4), "T4": (6, 0),
}
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


# ---------- 模块一：用 A* 计算各兴趣点之间的最小移动代价 ----------

@dataclass
class PathStats:
    expanded: int = 0
    pushed: int = 0
    path: list[Position] = field(default_factory=list)


def _prepare_costs(grid: Grid) -> tuple[list[list[int | None]], int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be rectangular")
    costs: list[list[int | None]] = []
    for row in grid:
        cost_row: list[int | None] = []
        for cell in row:
            if cell == "X":
                cost_row.append(None)
            elif isinstance(cell, str) and (cell in ("A", "B") or cell in TREASURE_VALUES):
                cost_row.append(1)
            elif isinstance(cell, int) and not isinstance(cell, bool) and cell >= 0:
                cost_row.append(cell)
            else:
                raise ValueError("cells must be non-negative integers, X, A, B or T1..T4")
        costs.append(cost_row)
    min_cost = min((c for row in costs for c in row if c is not None), default=0)
    return costs, min_cost


def astar_cost(
    start: Position,
    goal: Position,
    grid: Grid,
    *,
    stats: PathStats | None = None,
) -> int:
    """返回最小进入代价，无路返回 -1；起点不收费，目标格正常收费。

    h = 全图最小通行代价 * 曼哈顿距离，满足一致性；全图有零权格时 h=0。
    不修改原地图，可通过 stats.path 查看一条最优路径。
    """
    if stats is not None:
        stats.expanded = stats.pushed = 0
        stats.path.clear()
    costs, min_cost = _prepare_costs(grid)
    rows, cols = len(costs), len(costs[0])
    for row, col in (start, goal):
        if not (0 <= row < rows and 0 <= col < cols) or costs[row][col] is None:
            raise ValueError("start and goal must be passable cells inside grid")

    def heuristic(position: Position) -> int:
        return min_cost * (abs(position[0] - goal[0]) + abs(position[1] - goal[1]))

    order = count()
    frontier = [(heuristic(start), next(order), 0, start)]
    g_score = {start: 0}
    came_from: dict[Position, Position] = {}
    closed: set[Position] = set()
    if stats is not None:
        stats.pushed = 1

    while frontier:
        _, _, current_g, current = heappop(frontier)
        if current in closed or current_g != g_score[current]:
            continue
        if current == goal:
            if stats is not None:
                stats.path.append(current)
                while current in came_from:
                    current = came_from[current]
                    stats.path.append(current)
                stats.path.reverse()
            return current_g

        closed.add(current)
        if stats is not None:
            stats.expanded += 1
        row, col = current
        for dr, dc in DIRECTIONS:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            enter_cost = costs[nr][nc]
            if enter_cost is None or neighbor in closed:
                continue
            new_g = current_g + enter_cost
            if new_g < g_score.get(neighbor, inf):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heappush(frontier, (new_g + heuristic(neighbor), next(order), new_g, neighbor))
                if stats is not None:
                    stats.pushed += 1
    return -1


def compute_distances(
    grid: Grid = EXAMPLE_GRID,
    points: Mapping[str, Position] = POINTS,
) -> dict[str, dict[str, int]]:
    """分别用 A* 求各有向点对的距离；不假定矩阵对称，无路记 -1。"""
    return {
        source: {target: astar_cost(start, goal, grid) for target, goal in points.items()}
        for source, start in points.items()
    }


# ---------- 模块二：博弈状态、合法动作、状态转移 ----------

@dataclass(frozen=True)
class GameState:
    # 玩家只会停在初始点或已选宝藏处，用兴趣点名称表示坐标。
    pos_a: str = "A"
    pos_b: str = "B"
    remaining: tuple[str, ...] = tuple(TREASURE_VALUES)
    score_a: int = 0
    score_b: int = 0
    current_player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


def _actions(
    state: GameState, distances: Distances, values: Mapping[str, int], order: str,
) -> list[str]:
    position = state.pos_a if state.current_player == "A" else state.pos_b
    actions = [target for target in state.remaining if distances[position][target] >= 0]
    if order == "reverse":
        actions.reverse()
    elif order == "gain":
        # A 增加自身收益使 U 变大；B 增加自身收益使 U 变小。
        # 双方都优先尝试自己的即时收益较大的动作，不预先求子树真实分值。
        actions.sort(key=lambda target: values[target] - distances[position][target], reverse=True)
    return actions


def apply_action(
    state: GameState,
    target: str,
    distances: Distances,
    values: Mapping[str, int] = TREASURE_VALUES,
) -> GameState:
    """选择一个宝藏并返回新状态；只收集所选宝藏，收益允许为负。"""
    if state.current_player not in ("A", "B"):
        raise ValueError("current_player must be A or B")
    if target not in state.remaining:
        raise ValueError("target must be a remaining treasure")
    position = state.pos_a if state.current_player == "A" else state.pos_b
    distance = distances[position][target]
    if distance < 0:
        raise ValueError("target must be reachable")
    gain = values[target] - distance
    remaining = tuple(t for t in state.remaining if t != target)

    if state.current_player == "A":
        return replace(
            state, pos_a=target, remaining=remaining,
            score_a=state.score_a + gain, current_player="B",
        )
    return replace(
        state, pos_b=target, remaining=remaining,
        score_b=state.score_b + gain, current_player="A",
    )


# ---------- 模块三：Minimax 与 Alpha-beta，按行动数逐层递归 ----------

@dataclass
class SearchStats:
    visited: int = 0
    expanded: int = 0
    leaves: int = 0
    cutoffs: int = 0
    skipped_children: int = 0

    def reset(self) -> None:
        self.visited = self.expanded = self.leaves = 0
        self.cutoffs = self.skipped_children = 0


@dataclass(frozen=True)
class SearchResult:
    best_target: str | None
    best_value: int
    line: tuple[str, ...] = ()  # 双方最优应对下的一条行动序列。


def _check_search(state: GameState, depth: int, order: str) -> None:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    if order not in ("fixed", "reverse", "gain"):
        raise ValueError("order must be fixed, reverse or gain")
    if state.current_player not in ("A", "B"):
        raise ValueError("current_player must be A or B")


def minimax(
    state: GameState,
    distances: Distances,
    depth: int,
    *,
    values: Mapping[str, int] = TREASURE_VALUES,
    order: str = "fixed",
    stats: SearchStats | None = None,
) -> SearchResult:
    """返回指定深度下的最佳首步、收益差和一条最优行动序列。

    达到深度或宝藏取完时返回当前 U；扩展约定为无可达动作时也直接终止，
    不执行 pass。收益为负的可达动作仍须参与选择。平局保留访问顺序中的首个动作。
    """
    _check_search(state, depth, order)
    if stats is not None:
        stats.reset()

    def search(current: GameState, remaining_depth: int) -> tuple[int, tuple[str, ...]]:
        if stats is not None:
            stats.visited += 1
        actions = [] if remaining_depth == 0 else _actions(current, distances, values, order)
        if not actions:
            if stats is not None:
                stats.leaves += 1
            return current.utility, ()

        if stats is not None:
            stats.expanded += 1
        is_max = current.current_player == "A"
        best = -inf if is_max else inf
        best_line: tuple[str, ...] = ()
        for target in actions:
            child = apply_action(current, target, distances, values)
            score, child_line = search(child, remaining_depth - 1)
            if (is_max and score > best) or (not is_max and score < best):
                best, best_line = score, (target,) + child_line
        return int(best), best_line

    value, line = search(state, depth)
    return SearchResult(line[0] if line else None, value, line)


def alpha_beta(
    state: GameState,
    distances: Distances,
    depth: int,
    *,
    values: Mapping[str, int] = TREASURE_VALUES,
    order: str = "fixed",
    stats: SearchStats | None = None,
) -> SearchResult:
    """与 Minimax 相同的规则和决策，增加 alpha/beta 边界与剪枝。"""
    _check_search(state, depth, order)
    if stats is not None:
        stats.reset()

    def search(
        current: GameState, remaining_depth: int, alpha: int | float, beta: int | float,
    ) -> tuple[int, tuple[str, ...]]:
        if stats is not None:
            stats.visited += 1
        actions = [] if remaining_depth == 0 else _actions(current, distances, values, order)
        if not actions:
            if stats is not None:
                stats.leaves += 1
            return current.utility, ()

        if stats is not None:
            stats.expanded += 1
        is_max = current.current_player == "A"
        best = -inf if is_max else inf
        best_line: tuple[str, ...] = ()
        for index, target in enumerate(actions):
            child = apply_action(current, target, distances, values)
            score, child_line = search(child, remaining_depth - 1, alpha, beta)
            if (is_max and score > best) or (not is_max and score < best):
                best, best_line = score, (target,) + child_line

            if is_max:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta:
                remaining = len(actions) - index - 1
                if stats is not None and remaining > 0:
                    stats.cutoffs += 1
                    stats.skipped_children += remaining
                break
        # 独立返回 best；继承的 alpha/beta 可能来自祖先，不能当作本节点值。
        return int(best), best_line

    value, line = search(state, depth, -inf, inf)
    return SearchResult(line[0] if line else None, value, line)


def first_move_values(
    state: GameState, distances: Distances, depth: int,
    *, values: Mapping[str, int] = TREASURE_VALUES,
) -> dict[str, int]:
    """单独精确评估每个首步，供分析深度差异；不复用剪枝时返回的界。"""
    _check_search(state, depth, "fixed")
    if depth == 0:
        return {}
    return {
        target: minimax(
            apply_action(state, target, distances, values), distances, depth - 1, values=values,
        ).best_value
        for target in _actions(state, distances, values, "fixed")
    }


# ---------- 模块四：结果展示与可复现实验 ----------

def _print_matrix(distances: Distances) -> None:
    labels = list(distances)
    print("A* distance matrix (row=start, column=goal):")
    print("    " + " ".join(f"{label:>4}" for label in labels))
    for source in labels:
        print(f"{source:>4}" + " ".join(f"{distances[source][target]:>4}" for target in labels))


def _print_line(state: GameState, line: tuple[str, ...], distances: Distances) -> None:
    print("  player  from  target  cost  gain  score_A  score_B  U")
    for target in line:
        player = state.current_player
        source = state.pos_a if player == "A" else state.pos_b
        cost = distances[source][target]
        gain = TREASURE_VALUES[target] - cost
        state = apply_action(state, target, distances)
        print(
            f"  {player:>6}  {source:>4}  {target:>6}  {cost:>4}  {gain:>4}"
            f"  {state.score_a:>7}  {state.score_b:>7}  {state.utility:>2}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 4: A* and treasure game search")
    parser.add_argument("--trace", action="store_true", help="show an optimal line with move costs")
    parser.add_argument("--compare", action="store_true", help="show visited nodes and pruning counts")
    parser.add_argument("--order-experiment", action="store_true", help="compare fixed, reverse and gain order")
    parser.add_argument("--first-moves", action="store_true", help="show the exact value of each first move")
    args = parser.parse_args()

    distances = compute_distances()
    _print_matrix(distances)
    state = GameState()
    orders = ("fixed", "reverse", "gain") if args.order_experiment else ("fixed",)
    for depth in (2, 4):
        print(f"\nH={depth}")
        if args.first_moves:
            print("Exact first-move values:", json.dumps(first_move_values(state, distances, depth)))
        for order in orders:
            for name, algorithm in (("Minimax", minimax), ("Alpha-beta", alpha_beta)):
                stats = SearchStats()
                result = algorithm(state, distances, depth, order=order, stats=stats)
                payload = {"best_target": result.best_target, "best_value": result.best_value}
                label = f"{name} [{order}]" if args.order_experiment else name
                print(f"{label}: {json.dumps(payload)}")
                if args.compare or args.order_experiment:
                    print(
                        f"  visited={stats.visited}, expanded={stats.expanded}, leaves={stats.leaves}, "
                        f"cutoffs={stats.cutoffs}, skipped_children={stats.skipped_children}"
                    )
                if args.trace:
                    _print_line(state, result.line, distances)


if __name__ == "__main__":
    main()
