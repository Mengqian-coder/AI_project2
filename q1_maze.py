"""题目一：使用 A* 图搜索求四方向迷宫的最短路径。"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import inf

Position = tuple[int, int]
Heuristic = Callable[[Position, Position], int]
# 固定访问顺序；f 相同时按入队先后处理，避免坐标隐式决定优先级。
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def manhattan(position: Position, goal: Position) -> int:
    """忽略障碍物时，四方向移动所需的最少步数。"""
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def zero_heuristic(position: Position, goal: Position) -> int:
    """h=0 时，A* 退化为一致代价搜索（UCS）。"""
    return 0


@dataclass
class SearchStats:
    """expanded 不含终点；pushed 含起点及重复入队的条目。"""

    expanded: int = 0
    pushed: int = 0
    # 每次有效出队时记录 (位置, g, h, f)，包含最终的终点。
    trace: list[tuple[Position, int, int, int]] = field(default_factory=list)


def astar(
    start: Position,
    goal: Position,
    grid: Sequence[Sequence[str]],
    *,
    heuristic: Heuristic = manhattan,
    stats: SearchStats | None = None,
) -> list[Position]:
    """返回包含起终点的最短路径，无路可走时返回 []。

    为保证最优性，传入的 heuristic 应一致，且在终点取 0。
    start/goal 是坐标依据；除 X 外的格子均可通行，每步代价为 1。
    不修改 grid。无效地图或端点抛出 ValueError。
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("grid must be rectangular")
    for row, col in (start, goal):
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("start and goal must be inside grid")
        if grid[row][col] == "X":
            raise ValueError("start and goal must be passable")

    if stats is None:
        stats = SearchStats()
    stats.expanded = stats.pushed = 0
    stats.trace.clear()

    # frontier 即尚待处理的候选节点优先队列。
    # 条目：(f, 入队序号, 入队时的 g, 坐标)。序号用于打破 f 相同的平局。
    order = count()
    frontier = [(heuristic(start, goal), next(order), 0, start)]
    stats.pushed = 1
    g_score = {start: 0}       # 当前已发现的、到各状态的最小代价
    came_from: dict[Position, Position] = {}  # 当前最好路径上的前驱
    closed: set[Position] = set()            # 已经扩展的状态

    while frontier:#只有还有候选节点，就继续搜索
        current_f, _, current_g, current = heappop(frontier)#取出最小条目，_接收入队序号，表示后续不适用它

        # heapq 没有直接的 decrease-key；改进路径时放入新条目，旧条目略过。
        if current in closed or current_g != g_score[current]:#堆里记录已经过期，跳过（两者不相等直接跳过，更新堆中已过期记录）
            continue
        stats.trace.append((current, current_g, current_f - current_g, current_f))

        # 终点有效出队时才能结束，不能在发现/入队时结束。
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path#只需要保存前驱关系，不必给每个堆条目都保存整条路径

        closed.add(current)#当前节点确定需要拓展后加入closed集合
        stats.expanded += 1
        row, col = current
        for dr, dc in DIRECTIONS:#遍历四个方向的邻居
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue#过滤超出地图范围位置
            if grid[nr][nc] == "X" or neighbor in closed:
                continue#过滤障碍物和已扩展位置

            tentative_g = current_g + 1#每次移动代价1，所以当前节点到这个邻居代价+1
            # 尚未扩展的同一状态可能被再次发现；只保留更好的到达路径。
            if tentative_g < g_score.get(neighbor, inf):#邻居两种情况：从未发现过（inf）；已经发现过(已记录最小代价)
                g_score[neighbor] = tentative_g#发现更短路径，更新到达代价g_score
                came_from[neighbor] = current#更新前驱
                neighbor_f = tentative_g + heuristic(neighbor, goal)#计算新的f，重新入堆
                heappush(frontier, (neighbor_f, next(order), tentative_g, neighbor))
                stats.pushed += 1

    return []#如果堆最终清空也没有找到路径，返回空列表表示无路可走


EXAMPLE_GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
EXAMPLE_START = (0, 0)
EXAMPLE_GOAL = (3, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 1: A* maze search")
    parser.add_argument("--compare", action="store_true", help="compare h=0 and Manhattan")
    parser.add_argument("--trace", action="store_true", help="show valid dequeue order")
    args = parser.parse_args()

    heuristics = [("h=0 (UCS)", zero_heuristic), ("Manhattan", manhattan)]
    if not args.compare:
        heuristics = heuristics[1:]
    for name, heuristic in heuristics:
        stats = SearchStats()
        path = astar(EXAMPLE_START, EXAMPLE_GOAL, EXAMPLE_GRID, heuristic=heuristic, stats=stats)
        if args.compare:
            steps = len(path) - 1 if path else None
            print(f"{name}: steps={steps}, expanded={stats.expanded}, pushed={stats.pushed}")
        else:
            print(path)
        if args.trace:
            print("order  position  g  h  f")
            for index, (position, g, h, f) in enumerate(stats.trace, start=1):
                print(f"{index:>5}  {str(position):>8}  {g}  {h}  {f}")


if __name__ == "__main__":
    main()
