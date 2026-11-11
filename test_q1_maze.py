"""使用独立 BFS 作为最短路径参照，验证题目一。"""

from collections import deque
from copy import deepcopy
import random
import unittest

from q1_maze import (
    EXAMPLE_GOAL,
    EXAMPLE_GRID,
    EXAMPLE_START,
    SearchStats,
    astar,
    manhattan,
    zero_heuristic,
)


def bfs_distance(grid, start, goal):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != "X" and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append(((nr, nc), distance + 1))
    return None


class AStarMazeTests(unittest.TestCase):
    def assert_shortest_valid_path(self, grid, start, goal, heuristic=manhattan):
        before = deepcopy(grid)
        path = astar(start, goal, grid, heuristic=heuristic)
        distance = bfs_distance(grid, start, goal)
        self.assertEqual(grid, before, "search must not modify the map")
        if distance is None:
            self.assertEqual(path, [])
            return
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(path) - 1, distance)
        for row, col in path:
            self.assertTrue(0 <= row < len(grid) and 0 <= col < len(grid[0]))
            self.assertNotEqual(grid[row][col], "X")
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1)

    def test_official_example(self):
        self.assert_shortest_valid_path(EXAMPLE_GRID, EXAMPLE_START, EXAMPLE_GOAL)
        self.assertEqual(bfs_distance(EXAMPLE_GRID, EXAMPLE_START, EXAMPLE_GOAL), 7)

    def test_unreachable_goal(self):
        self.assertEqual(astar((0, 0), (0, 2), [list("SXE")]), [])

    def test_start_equals_goal(self):
        self.assertEqual(astar((0, 0), (0, 0), [["."]]), [(0, 0)])

    def test_rectangular_map_and_detour(self):
        self.assert_shortest_valid_path([list("SX..E"), list("...X."), list(".....")], (0, 0), (0, 4))

    def test_better_route_to_an_already_discovered_state(self):
        # (3, 2) 先被发现 g=5，随后应更新为 g=3。
        # 若把“已入队”当成 closed，本图会错误返回 11 步。
        grid = [list(row) for row in ("..XE.", "...X.", "SX.X.", ".....", "X.X..")]
        self.assert_shortest_valid_path(grid, (2, 0), (0, 3))
        self.assertEqual(bfs_distance(grid, (2, 0), (0, 3)), 9)

    def test_invalid_inputs(self):
        for grid, start, goal in [
            ([], (0, 0), (0, 0)),
            ([[]], (0, 0), (0, 0)),
            ([list(".."), list(".")], (0, 0), (1, 0)),
            ([["."]], (-1, 0), (0, 0)),
            ([["."]], (0, 0), (0, 1)),
            ([list("X.")], (0, 0), (0, 1)),
            ([list(".X")], (0, 0), (0, 1)),
        ]:
            with self.subTest(grid=grid, start=start, goal=goal):
                with self.assertRaises(ValueError):
                    astar(start, goal, grid)

    def test_search_statistics(self):
        stats = SearchStats()
        astar(EXAMPLE_START, EXAMPLE_GOAL, EXAMPLE_GRID, stats=stats)
        self.assertEqual(stats.trace[0][0], EXAMPLE_START)
        self.assertEqual(stats.trace[-1][0], EXAMPLE_GOAL)
        self.assertEqual(stats.expanded, len(stats.trace) - 1)
        self.assertEqual(len({step[0] for step in stats.trace}), len(stats.trace))
        # 一致启发函数下，有效出队的 f 不下降。
        priorities = [step[3] for step in stats.trace]
        self.assertEqual(priorities, sorted(priorities))
        # 复用统计对象时，每次调用从零开始计数。
        self.assertEqual(astar((0, 0), (0, 0), [["."]], stats=stats), [(0, 0)])
        self.assertEqual((stats.expanded, stats.pushed), (0, 1))

    def test_random_maps_against_bfs(self):
        rng = random.Random(20261028)
        for case in range(200):
            rows, cols = rng.randint(1, 9), rng.randint(1, 9)
            grid = [["X" if rng.random() < 0.3 else "." for _ in range(cols)] for _ in range(rows)]
            start, goal = (0, 0), (rows - 1, cols - 1)
            grid[0][0] = grid[-1][-1] = "."
            for heuristic in (manhattan, zero_heuristic):
                with self.subTest(case=case, heuristic=heuristic.__name__):
                    self.assert_shortest_valid_path(grid, start, goal, heuristic)


if __name__ == "__main__":
    unittest.main()
