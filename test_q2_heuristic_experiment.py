"""独立验证前瞻启发函数的性质，而不依赖实验统计结果。"""

from heapq import heappop, heappush
from math import inf
import random
import unittest

from q2_heuristic_experiment import LookaheadHeuristic, reverse_distances
from q2_weighted import WeightedSearchStats, astar_weighted


def neighbors(costs, position):
    row, col = position
    for nr, nc in ((row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)):
        if (0 <= nr < len(costs) and 0 <= nc < len(costs[0])
                and costs[nr][nc] is not None):
            yield nr, nc


def forward_dijkstra(costs, start, goal):
    """从指定起点正向搜索；独立检查反向 oracle 的边费用方向。"""
    queue = [(0, start)]
    settled = set()
    while queue:
        cost, position = heappop(queue)
        if position in settled:
            continue
        settled.add(position)
        if position == goal:
            return cost
        for neighbor in neighbors(costs, position):
            if neighbor not in settled:
                nr, nc = neighbor
                heappush(queue, (cost + costs[nr][nc], neighbor))
    return inf


def terrain_minimum(costs, goal):
    return min((value for row, cells in enumerate(costs)
                for col, value in enumerate(cells)
                if value is not None and (row, col) != goal), default=0)


def to_grid(costs):
    return [["X" if value is None else value for value in row] for row in costs]


class LookaheadHeuristicTests(unittest.TestCase):
    def assert_properties(self, costs, goal):
        minimum = terrain_minimum(costs, goal)
        estimates = [LookaheadHeuristic(costs, goal, minimum, depth)
                     for depth in range(4)]
        exact = reverse_distances(costs, goal)
        for row, cells in enumerate(costs):
            for col, cost in enumerate(cells):
                if cost is None:
                    continue
                position = (row, col)
                expected = forward_dijkstra(costs, position, goal)
                self.assertEqual(exact.get(position, inf), expected)
                values = [heuristic(position, goal) for heuristic in estimates]
                self.assertEqual(values[0], minimum * max(
                    abs(row - goal[0]) + abs(col - goal[1]) - 1, 0))
                self.assertEqual(values, sorted(values), (position, values))
                self.assertLessEqual(values[-1], expected, (position, values, expected))
                for heuristic in estimates:
                    self.assertEqual(heuristic(goal, goal), 0)
                    for neighbor in neighbors(costs, position):
                        nr, nc = neighbor
                        self.assertLessEqual(
                            heuristic(position, goal),
                            costs[nr][nc] + heuristic(neighbor, goal),
                            (position, neighbor),
                        )

    def test_forced_expensive_first_step_improves_bound(self):
        costs, goal = [[1, 8, 1, 0]], (0, 3)
        values = [LookaheadHeuristic(costs, goal, 1, depth)((0, 0), goal)
                  for depth in range(4)]
        self.assertEqual(values, [2, 9, 9, 9])

    def test_two_steps_reveal_expensive_terrain(self):
        # 回到起点也要支付起点的真实进入费，不能当作免费撤回。
        costs, goal = [[8, 1, 8, 0]], (0, 3)
        values = [LookaheadHeuristic(costs, goal, 1, depth)((0, 0), goal)
                  for depth in range(4)]
        self.assertEqual(values, [2, 2, 9, 9])

    def test_entering_goal_is_free_at_every_depth(self):
        costs, goal = [[8, 0]], (0, 1)
        for depth in range(5):
            heuristic = LookaheadHeuristic(costs, goal, 8, depth)
            self.assertEqual(heuristic((0, 0), goal), 0)
            self.assertEqual(heuristic(goal, goal), 0)

    def test_reverse_dijkstra_uses_original_edge_destination_cost(self):
        # 原图左到右需要 2；倒着搜索时也应加 2，不能误加起点的 9。
        costs, goal = [[9, 2, 0]], (0, 2)
        exact = reverse_distances(costs, goal)
        self.assertEqual(exact[(0, 0)], 2)
        self.assertEqual(exact[(0, 1)], 0)
        self.assertEqual(exact[goal], 0)
        self.assertEqual(forward_dijkstra(costs, goal, (0, 0)), 11)

    def test_zero_minimum_can_still_gain_information_from_lookahead(self):
        costs, goal = [[1, 8, 0, 0]], (0, 3)
        h0 = LookaheadHeuristic(costs, goal, 0, 0)
        h1 = LookaheadHeuristic(costs, goal, 0, 1)
        self.assertEqual(h0((0, 0), goal), 0)
        self.assertEqual(h1((0, 0), goal), 8)
        self.assert_properties(costs, goal)

    def test_zero_cost_cycles_and_disconnected_cells(self):
        costs = [[1, 0, None, 8], [0, 0, None, None], [1, 0, 1, 0]]
        goal = (2, 3)
        self.assert_properties(costs, goal)
        # 孤立格不能到达目标，所有 A* 版本也应报告无解。
        for depth in range(4):
            heuristic = LookaheadHeuristic(costs, goal, 0, depth)
            self.assertEqual(astar_weighted((0, 3), goal, to_grid(costs),
                                           heuristic=heuristic), -1)

    def test_cache_and_instrumentation_preserve_values(self):
        costs, goal = [[8, 1, 8, 0], [2, 4, 2, 2]], (0, 3)
        measured = LookaheadHeuristic(costs, goal, 1, 2, instrument=True)
        unmeasured = LookaheadHeuristic(costs, goal, 1, 2, instrument=False)
        for position in ((0, 0), (1, 0), (1, 2), goal):
            self.assertEqual(measured(position, goal), unmeasured(position, goal))
        previous_evaluations = measured.metrics.evaluations
        previous_queries = measured.metrics.queries
        previous_hits = measured.metrics.cache_hits
        measured((0, 0), goal)
        self.assertEqual(measured.metrics.evaluations, previous_evaluations)
        self.assertEqual(measured.metrics.queries, previous_queries + 1)
        self.assertGreater(measured.metrics.cache_hits, previous_hits)
        self.assertGreater(measured.cache_size, 0)
        self.assertEqual((unmeasured.metrics.queries, unmeasured.metrics.evaluations,
                          unmeasured.metrics.edge_evaluations, unmeasured.metrics.cache_hits),
                         (0, 0, 0, 0))

    def test_random_maps_bounds_consistency_and_astar_optimality(self):
        rng = random.Random(20261028)
        for case in range(50):
            rows, cols = rng.randint(1, 6), rng.randint(1, 6)
            # 轮换含零、最低费用 1、最低费用大于 1 的地形。
            weights = ((0, 1, 3, 8), (1, 3, 8), (2, 4, 9))[case % 3]
            costs = [[None if rng.random() < 0.2 else rng.choice(weights)
                      for _ in range(cols)] for _ in range(rows)]
            start, goal = (0, 0), (rows - 1, cols - 1)
            costs[0][0] = rng.choice(weights)
            costs[-1][-1] = 0
            with self.subTest(case=case):
                self.assert_properties(costs, goal)
                expected = forward_dijkstra(costs, start, goal)
                minimum = terrain_minimum(costs, goal)
                for depth in range(4):
                    heuristic = LookaheadHeuristic(costs, goal, minimum, depth)
                    stats = WeightedSearchStats()
                    actual = astar_weighted(start, goal, to_grid(costs),
                                            heuristic=heuristic, stats=stats)
                    self.assertEqual(actual, -1 if expected == inf else expected)
                    if actual != -1:
                        self.assertEqual(sum(costs[r][c] for r, c in stats.path[1:]), actual)
                        priorities = [step[3] for step in stats.trace]
                        self.assertEqual(priorities, sorted(priorities))


if __name__ == "__main__":
    unittest.main()
