"""使用独立 Dijkstra 验证加权地图上的最小代价。"""

from copy import deepcopy
from heapq import heappop, heappush
import random
import unittest

from q2_weighted import (
    EXAMPLE_GOAL,
    EXAMPLE_GRID,
    EXAMPLE_START,
    WeightedSearchStats,
    adjusted_manhattan,
    astar_weighted,
    manhattan,
    zero_heuristic,
)


def entry_cost(grid, position, goal):
    if position == goal:
        return 0
    cell = grid[position[0]][position[1]]
    return 1 if cell == "S" else int(cell)


def dijkstra_cost(grid, start, goal):
    queue = [(0, start)]
    visited = set()
    while queue:
        cost, position = heappop(queue)
        if position in visited:
            continue
        visited.add(position)
        if position == goal:
            return cost
        row, col = position
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            nr, nc = neighbor
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != "X" and neighbor not in visited):
                heappush(queue, (cost + entry_cost(grid, neighbor, goal), neighbor))
    return -1


class WeightedAStarTests(unittest.TestCase):
    def assert_optimal_valid_path(self, grid, start, goal, heuristic=None):
        original = deepcopy(grid)
        stats = WeightedSearchStats()
        cost = astar_weighted(start, goal, grid, heuristic=heuristic, stats=stats)
        self.assertEqual(grid, original)
        self.assertEqual(cost, dijkstra_cost(grid, start, goal))
        if cost == -1:
            self.assertEqual(stats.path, [])
            return
        self.assertEqual(stats.path[0], start)
        self.assertEqual(stats.path[-1], goal)
        for row, col in stats.path:
            self.assertTrue(0 <= row < len(grid) and 0 <= col < len(grid[0]))
            self.assertNotEqual(grid[row][col], "X")
        for (r1, c1), (r2, c2) in zip(stats.path, stats.path[1:]):
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1)
        self.assertEqual(sum(entry_cost(grid, position, goal) for position in stats.path[1:]), cost)
        priorities = [step[3] for step in stats.trace]
        self.assertEqual(priorities, sorted(priorities))

    def test_official_example(self):
        self.assertEqual(astar_weighted(EXAMPLE_START, EXAMPLE_GOAL, EXAMPLE_GRID), 7)
        self.assert_optimal_valid_path(EXAMPLE_GRID, EXAMPLE_START, EXAMPLE_GOAL)

    def test_adjacent_goal_is_free(self):
        self.assertEqual(astar_weighted((0, 0), (0, 1), [["S", "E"]]), 0)
        self.assertEqual(adjusted_manhattan((0, 0), (0, 1)), 0)
        self.assertGreater(manhattan((0, 0), (0, 1)), 0)

    def test_cheapest_route_can_take_more_steps(self):
        grid = [["S", 9, "E"], [1, 1, 1]]
        stats = WeightedSearchStats()
        self.assertEqual(astar_weighted((0, 0), (0, 2), grid, stats=stats), 3)
        self.assertEqual(len(stats.path) - 1, 4)
        self.assert_optimal_valid_path(grid, (0, 0), (0, 2))

    def test_start_is_not_charged_and_numeric_goal_is_free(self):
        self.assertEqual(astar_weighted((0, 0), (0, 2), [[99, 2, 99]]), 2)
        self.assertEqual(astar_weighted((0, 0), (0, 2), [["S", "3", "E"]]), 3)

    def test_no_path_and_same_endpoint(self):
        self.assert_optimal_valid_path([["S", "X", "E"]], (0, 0), (0, 2))
        self.assertEqual(astar_weighted((0, 0), (0, 0), [[9]]), 0)

    def test_zero_cost_terrain(self):
        grid = [["S", 5, "E"], [0, 0, 0]]
        stats = WeightedSearchStats()
        self.assertEqual(astar_weighted((0, 0), (0, 2), grid, stats=stats), 0)
        self.assertTrue(all(step[2] == 0 for step in stats.trace))
        self.assert_optimal_valid_path(grid, (0, 0), (0, 2))

    def test_heuristic_scales_with_minimum_terrain_cost(self):
        self.assertEqual(adjusted_manhattan((0, 0), (2, 3), 3), 12)
        grid = [[3, 9, 9], [3, 3, 3]]
        self.assert_optimal_valid_path(grid, (0, 0), (0, 2))
        stats = WeightedSearchStats()
        astar_weighted((0, 0), (0, 2), grid, stats=stats)
        self.assertEqual(stats.trace[0][2], 3)
        # S 重入代价 1 也参与最小代价计算。
        astar_weighted((0, 0), (0, 3), [["S", 3, 3, "E"]], stats=stats)
        self.assertEqual(stats.trace[0][2], 2)

    def test_invalid_inputs(self):
        for grid, start, goal in [
            ([], (0, 0), (0, 0)),
            ([[]], (0, 0), (0, 0)),
            ([[1, 2], [1]], (0, 0), (1, 0)),
            ([[1]], (-1, 0), (0, 0)),
            ([[1]], (0, 0), (0, 1)),
            ([["X", "E"]], (0, 0), (0, 1)),
            ([["S", "X"]], (0, 0), (0, 1)),
            ([["S", -1, "E"]], (0, 0), (0, 2)),
            ([["S", "-2", "E"]], (0, 0), (0, 2)),
            ([["S", 1.5, "E"]], (0, 0), (0, 2)),
            ([["S", True, "E"]], (0, 0), (0, 2)),
            ([["S", ".", "E"]], (0, 0), (0, 2)),
            ([["S", "S", "E"]], (0, 0), (0, 2)),
            ([["S", "E", 1]], (0, 0), (0, 2)),
        ]:
            with self.subTest(grid=grid, start=start, goal=goal):
                with self.assertRaises(ValueError):
                    astar_weighted(start, goal, grid)

    def test_statistics_reset(self):
        stats = WeightedSearchStats()
        astar_weighted(EXAMPLE_START, EXAMPLE_GOAL, EXAMPLE_GRID, stats=stats)
        self.assertEqual(stats.expanded, len(stats.trace) - 1)
        self.assertEqual(stats.trace[-1][0], EXAMPLE_GOAL)
        astar_weighted((0, 0), (0, 0), [[1]], stats=stats)
        self.assertEqual((stats.expanded, stats.pushed, stats.path), (0, 1, [(0, 0)]))

    def test_random_maps_against_dijkstra(self):
        rng = random.Random(20261029)
        for case in range(200):
            rows, cols = rng.randint(1, 8), rng.randint(1, 8)
            grid = [["X" if rng.random() < 0.25 else rng.randint(0, 9)
                     for _ in range(cols)] for _ in range(rows)]
            start, goal = (0, 0), (rows - 1, cols - 1)
            grid[0][0] = grid[-1][-1] = 1
            for heuristic in (None, zero_heuristic):
                with self.subTest(case=case, heuristic=heuristic):
                    self.assert_optimal_valid_path(grid, start, goal, heuristic)


if __name__ == "__main__":
    unittest.main()
