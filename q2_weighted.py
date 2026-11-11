"""题目二：使用 A* 计算加权地形上的最小移动代价。"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import inf

Position = tuple[int, int]
Heuristic = Callable[[Position, Position], int]
Grid = Sequence[Sequence[int | str]]
CostGrid = list[list[int | None]]
# 右、下、左、上；f 相同时按入队先后处理。
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def manhattan(position: Position, goal: Position) -> int:
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def zero_heuristic(position: Position, goal: Position) -> int:
    return 0


def adjusted_manhattan(position: Position, goal: Position, min_cost: int = 1) -> int:
    """最后一步进入终点免费，其余每步至少花费 min_cost。"""
    return min_cost * max(manhattan(position, goal) - 1, 0)


@dataclass
class WeightedSearchStats:
    """统计口径同题一；额外保留最优路径，供分析累计代价。"""

    expanded: int = 0
    pushed: int = 0
    trace: list[tuple[Position, int, int, int]] = field(default_factory=list)
    path: list[Position] = field(default_factory=list)


def _prepare_costs(start: Position, goal: Position, grid: Grid) -> tuple[CostGrid, int]:
    """把地图转换为进入代价；None 表示障碍，终点进入代价固定为 0。"""
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("grid must be rectangular")
    for row, col in (start, goal):
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("start and goal must be inside grid")

    costs: CostGrid = []
    for row, cells in enumerate(grid):
        cost_row: list[int | None] = []
        for col, cell in enumerate(cells):
            position = (row, col)
            if cell == "X":
                cost_row.append(None)
                continue
            if cell == "S":
                if position != start:
                    raise ValueError("S must match the start coordinate")
                cost = 1  # 初始不收费；若重新进入 S，按普通格代价 1 处理。
            elif cell == "E":
                if position != goal:
                    raise ValueError("E must match the goal coordinate")
                cost = 0
            else:
                if isinstance(cell, bool) or not isinstance(cell, (int, str)):
                    raise ValueError("terrain costs must be non-negative integers")
                try:
                    cost = int(cell)
                except ValueError as error:
                    raise ValueError("terrain costs must be non-negative integers") from error
                if cost < 0:
                    raise ValueError("terrain costs must be non-negative integers")
            # 坐标参数是依据；即使终点位置写的是数字，也按题目规则免费进入。
            cost_row.append(0 if position == goal else cost)
        costs.append(cost_row)

    if any(costs[row][col] is None for row, col in (start, goal)):
        raise ValueError("start and goal must be passable")

    # 包含非终点的 0 代价格；若存在，启发函数安全地退化为 h=0。
    min_cost = min(
        (cost for row, values in enumerate(costs) for col, cost in enumerate(values)
         if cost is not None and (row, col) != goal),
        default=0,
    )
    return costs, min_cost


def astar_weighted(
    start: Position,
    goal: Position,
    grid: Grid,
    *,
    heuristic: Heuristic | None = None,
    stats: WeightedSearchStats | None = None,
) -> int:
    """返回最小总代价；无解返回 -1，起终点相同返回 0。

    初始 g(start)=0，进入邻居按该格权值收费，进入 goal 固定为 0。
    数字可为非负整数或数字字符串；可用 X 表示障碍。
    默认启发函数为 min_cost * max(曼哈顿距离 - 1, 0)。
    本实现不重开 closed；自定义启发函数须一致，且在终点取 0。
    最优路径可通过 stats.path 查看，不修改输入地图。
    """
    costs, min_cost = _prepare_costs(start, goal, grid)
    rows, cols = len(costs), len(costs[0])
    if heuristic is None:
        def estimate(position: Position, target: Position) -> int:
            return adjusted_manhattan(position, target, min_cost)
        heuristic = estimate
    if stats is None:
        stats = WeightedSearchStats()
    stats.expanded = stats.pushed = 0
    stats.trace.clear()
    stats.path.clear()

    order = count()
    frontier = [(heuristic(start, goal), next(order), 0, start)]
    stats.pushed = 1
    g_score = {start: 0}
    came_from: dict[Position, Position] = {}
    closed: set[Position] = set()

    while frontier:
        current_f, _, current_g, current = heappop(frontier)
        if current in closed or current_g != g_score[current]:
            continue
        stats.trace.append((current, current_g, current_f - current_g, current_f))

        if current == goal:
            # 题二返回整数代价；前驱回溯得到的路径只放入辅助统计对象。
            stats.path.append(current)
            while current in came_from:
                current = came_from[current]
                stats.path.append(current)
            stats.path.reverse()
            return current_g

        closed.add(current)
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

            # 与题一的核心区别：由固定 +1 改为加上“进入邻居”的代价。
            tentative_g = current_g + enter_cost
            if tentative_g < g_score.get(neighbor, inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                # f 的定义没有变化，仍然是 g+h。
                neighbor_f = tentative_g + heuristic(neighbor, goal)
                heappush(frontier, (neighbor_f, next(order), tentative_g, neighbor))
                stats.pushed += 1

    return -1


EXAMPLE_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
EXAMPLE_START = (0, 0)
EXAMPLE_GOAL = (4, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 2: weighted A* search")
    parser.add_argument("--trace", action="store_true", help="show search and path costs")
    parser.add_argument("--compare", action="store_true", help="compare h=0 and adjusted Manhattan")
    args = parser.parse_args()
    heuristics: list[tuple[str, Heuristic | None]] = [
        ("h=0 (UCS)", zero_heuristic), ("Adjusted Manhattan", None),
    ]
    if not args.compare:
        heuristics = heuristics[1:]
    for name, heuristic in heuristics:
        stats = WeightedSearchStats()
        cost = astar_weighted(EXAMPLE_START, EXAMPLE_GOAL, EXAMPLE_GRID, heuristic=heuristic, stats=stats)
        if args.compare:
            print(f"{name}: cost={cost}, expanded={stats.expanded}, pushed={stats.pushed}")
        else:
            print(cost)
        if args.trace:
            print("order  position  g  h  f")
            for index, (position, g, h, f) in enumerate(stats.trace, start=1):
                print(f"{index:>5}  {str(position):>8}  {g}  {h}  {f}")
            print("path:", stats.path)
            costs, _ = _prepare_costs(EXAMPLE_START, EXAMPLE_GOAL, EXAMPLE_GRID)
            total = 0
            print("position  enter_cost  cumulative_cost")
            for index, position in enumerate(stats.path):
                row, col = position
                enter_cost = 0 if index == 0 else costs[row][col]
                assert enter_cost is not None
                total += enter_cost
                print(f"{str(position):>8}  {enter_cost:>10}  {total:>15}")


if __name__ == "__main__":
    main()
