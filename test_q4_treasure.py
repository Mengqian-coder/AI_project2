"""独立最短路与自底向上博弈树校验，不调用被测实现构造预期结果。"""

from dataclasses import FrozenInstanceError
import random
import unittest
from typing import cast

from q4_treasure import (
    EXAMPLE_GRID,
    Grid,
    Position,
    POINTS,
    TREASURE_VALUES,
    GameState,
    PathStats,
    SearchStats,
    alpha_beta,
    apply_action,
    astar_cost,
    compute_distances,
    first_move_values,
    minimax,
)


def cell_cost(cell: int | str) -> int:
    return cell if isinstance(cell, int) else 1


def dijkstra_distances(start: Position, grid: Grid) -> dict[Position, int]:
    """用线性选点的 Dijkstra 作 oracle，不使用启发函数或堆。"""
    pending = {start: 0}
    settled = {}
    while pending:
        position = min(pending, key=lambda item: pending[item])
        distance = pending.pop(position)
        settled[position] = distance
        row, col = position
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            neighbor = (nr, nc)
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            if grid[nr][nc] == "X" or neighbor in settled:
                continue
            candidate = distance + cell_cost(grid[nr][nc])
            if candidate < pending.get(neighbor, float("inf")):
                pending[neighbor] = candidate
    return settled


def exhaustive_game(state, distances, depth, values=TREASURE_VALUES, order="fixed"):
    """先显式展开整棵树，再倒序求值；不用递归、apply_action 或 alpha/beta。"""
    initial = (
        state.pos_a, state.pos_b, state.remaining,
        state.score_a, state.score_b, state.current_player,
    )
    nodes: list[
        tuple[
            tuple[str, str, tuple[str, ...], int, int, str],
            int,
            list[tuple[str, int]],
        ]
    ] = [(initial, depth, [])]
    cursor = 0
    while cursor < len(nodes):
        data, remaining_depth, children = nodes[cursor]
        pa, pb, remaining, sa, sb, player = data
        source = pa if player == "A" else pb
        actions = [t for t in remaining if distances[source][t] >= 0] if remaining_depth else []
        if order == "reverse":
            actions.reverse()
        elif order == "gain":
            actions = sorted(actions, key=lambda t: values[t] - distances[source][t], reverse=True)
        for target in actions:
            gain = values[target] - distances[source][target]
            left = tuple(t for t in remaining if t != target)
            successor = (
                (target, pb, left, sa + gain, sb, "B") if player == "A"
                else (pa, target, left, sa, sb + gain, "A")
            )
            children.append((target, len(nodes)))
            nodes.append((successor, remaining_depth - 1, []))
        cursor += 1

    results: list[tuple[int, tuple[str, ...]] | None] = [None] * len(nodes)
    for index in range(len(nodes) - 1, -1, -1):
        data, _, children = nodes[index]
        if not children:
            results[index] = (data[3] - data[4], ())
            continue
        # max/min 都保留第一个同分项，记录相应完整行动序列。
        scored_children = []
        for target, child_index in children:
            child_result = results[child_index]
            assert child_result is not None
            scored_children.append((target, child_result[0], child_result[1]))
        choose = max if data[5] == "A" else min
        target, score, continuation = choose(scored_children, key=lambda item: item[1])
        results[index] = (score, (target,) + continuation)
    root_result = results[0]
    assert root_result is not None
    value, line = root_result
    return (line[0] if line else None), value, line, len(nodes)


class TreasurePathTests(unittest.TestCase):
    def assert_valid_path(self, grid, start, goal, expected, stats):
        if expected < 0:
            self.assertEqual(stats.path, [])
            return
        self.assertEqual(stats.path[0], start)
        self.assertEqual(stats.path[-1], goal)
        for (r1, c1), (r2, c2) in zip(stats.path, stats.path[1:]):
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1)
            self.assertNotEqual(grid[r2][c2], "X")
        self.assertEqual(sum(cell_cost(grid[r][c]) for r, c in stats.path[1:]), expected)

    def test_all_official_pairs_against_dijkstra(self):
        original = [row[:] for row in EXAMPLE_GRID]
        distances = compute_distances()
        self.assertEqual(list(distances), list(POINTS))
        for source, start in POINTS.items():
            oracle = dijkstra_distances(start, EXAMPLE_GRID)
            for target, goal in POINTS.items():
                with self.subTest(source=source, target=target):
                    expected = oracle.get(goal, -1)
                    stats = PathStats()
                    self.assertEqual(distances[source][target], expected)
                    self.assertEqual(astar_cost(start, goal, EXAMPLE_GRID, stats=stats), expected)
                    self.assert_valid_path(EXAMPLE_GRID, start, goal, expected, stats)
        self.assertEqual(EXAMPLE_GRID, original)

    def test_destination_is_charged_and_start_is_not(self):
        grid = [[2, 9]]
        self.assertEqual(astar_cost((0, 0), (0, 1), grid), 9)
        self.assertEqual(astar_cost((0, 1), (0, 0), grid), 2)
        distances = compute_distances(grid, {"U": (0, 0), "V": (0, 1)})
        self.assertEqual(distances, {"U": {"U": 0, "V": 9}, "V": {"U": 2, "V": 0}})
        for target in ("A", "B", "T1", "T2", "T3", "T4"):
            self.assertEqual(astar_cost((0, 0), (0, 1), [[99, target]]), 1)

    def test_longer_route_can_be_cheaper_and_zero_weights_are_supported(self):
        grid = [["A", 9, "T1"], [1, 1, 1]]
        stats = PathStats()
        self.assertEqual(astar_cost((0, 0), (0, 2), grid, stats=stats), 4)
        self.assertEqual(len(stats.path), 5)
        self.assertEqual(astar_cost((0, 0), (1, 2), [[0, 0, 0], [0, 0, 0]]), 0)

    def test_unreachable_same_position_and_statistics_reset(self):
        stats = PathStats()
        self.assertEqual(astar_cost((0, 0), (0, 2), [["A", 1, "B"]], stats=stats), 2)
        self.assertGreater(stats.expanded, 0)
        self.assertEqual(astar_cost((0, 0), (0, 0), [[8]], stats=stats), 0)
        self.assertEqual((stats.expanded, stats.pushed, stats.path), (0, 1, [(0, 0)]))
        self.assertEqual(astar_cost((0, 0), (0, 2), [[1, "X", 1]], stats=stats), -1)
        self.assertEqual((stats.expanded, stats.pushed, stats.path), (1, 1, []))

    def test_bad_maps_and_endpoints_are_rejected(self):
        for grid in ([], [[]], [[1], [1, 2]], [[-1]], [[True]], [[1.5]], [["1"]], [["E"]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                astar_cost((0, 0), (0, 0), cast(Grid, grid))
        for start, goal in (((-1, 0), (0, 0)), ((0, 0), (0, 2)), ((0, 0), (0, 1))):
            with self.subTest(start=start, goal=goal), self.assertRaises(ValueError):
                astar_cost(start, goal, [[1, "X"]])

    def test_random_weighted_grids_against_dijkstra(self):
        rng = random.Random(20261102)
        for case in range(60):
            rows, cols = rng.randint(2, 6), rng.randint(2, 6)
            grid = [["X" if rng.random() < 0.22 else rng.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
            grid[0][0] = "A"
            grid[-1][-1] = "T1"
            original = [row[:] for row in grid]
            starts = [(0, 0), (rows - 1, cols - 1)]
            for start in starts:
                oracle = dijkstra_distances(start, grid)
                for goal in starts:
                    stats = PathStats()
                    with self.subTest(case=case, start=start, goal=goal):
                        expected = oracle.get(goal, -1)
                        self.assertEqual(astar_cost(start, goal, grid, stats=stats), expected)
                        self.assert_valid_path(grid, start, goal, expected, stats)
            self.assertEqual(grid, original)


class TreasureGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances = compute_distances()

    def assert_searches_agree(self, state, distances, depth, values=TREASURE_VALUES, order="fixed"):
        expected_target, expected_value, expected_line, full_visits = exhaustive_game(
            state, distances, depth, values, order,
        )
        snapshot = (state.pos_a, state.pos_b, state.remaining, state.score_a, state.score_b, state.current_player)
        old_distances = {source: dict(row) for source, row in distances.items()}
        for algorithm in (minimax, alpha_beta):
            with self.subTest(algorithm=algorithm.__name__, order=order):
                stats = SearchStats()
                result = algorithm(state, distances, depth, values=values, order=order, stats=stats)
                self.assertEqual((result.best_target, result.best_value, result.line), (expected_target, expected_value, expected_line))
                self.assertEqual(stats.visited, stats.expanded + stats.leaves)
                self.assertGreaterEqual(stats.leaves, 1)
                self.assertLessEqual(stats.visited, full_visits)
                if algorithm is minimax:
                    self.assertEqual(stats.visited, full_visits)
                    self.assertEqual((stats.cutoffs, stats.skipped_children), (0, 0))
                self.assertLessEqual(len(result.line), depth)
                replay = state
                for target in result.line:
                    replay = apply_action(replay, target, distances, values)
                self.assertEqual(replay.utility, result.best_value)
        self.assertEqual((state.pos_a, state.pos_b, state.remaining, state.score_a, state.score_b, state.current_player), snapshot)
        self.assertEqual({source: dict(row) for source, row in distances.items()}, old_distances)

    def test_official_h2_h4_and_complete_tree_counts(self):
        state = GameState()
        for depth, target, value, counts in ((2, "T2", -1, (17, 5, 12)), (4, "T3", 3, (65, 41, 24))):
            self.assert_searches_agree(state, self.distances, depth)
            plain, pruned = SearchStats(), SearchStats()
            result = minimax(state, self.distances, depth, stats=plain)
            self.assertEqual((result.best_target, result.best_value), (target, value))
            alpha_beta(state, self.distances, depth, stats=pruned)
            self.assertEqual((plain.visited, plain.expanded, plain.leaves), counts)
            self.assertLess(pruned.visited, plain.visited)
            self.assertGreater(pruned.skipped_children, 0)

    def test_state_transition_updates_only_the_current_player(self):
        state = GameState(score_a=4, score_b=2)
        after_a = apply_action(state, "T2", self.distances)
        self.assertEqual((after_a.pos_a, after_a.pos_b, after_a.current_player), ("T2", "B", "B"))
        self.assertEqual((after_a.score_a, after_a.score_b), (4 + 6 - self.distances["A"]["T2"], 2))
        self.assertEqual(after_a.remaining, ("T1", "T3", "T4"))
        after_b = apply_action(after_a, "T3", self.distances)
        self.assertEqual((after_b.pos_a, after_b.pos_b, after_b.current_player), ("T2", "T3", "A"))
        self.assertEqual((after_b.score_a, after_b.score_b), (after_a.score_a, 2 + 7 - self.distances["B"]["T3"]))
        self.assertEqual(after_b.utility, after_b.score_a - after_b.score_b)
        self.assertEqual(state.remaining, tuple(TREASURE_VALUES))
        with self.assertRaises(FrozenInstanceError):
            setattr(state, "score_a", 100)

    def test_passing_another_treasure_does_not_collect_it(self):
        grid = [["A", "T1", "T2", "B"]]
        distances = compute_distances(grid, {"A": (0, 0), "B": (0, 3), "T1": (0, 1), "T2": (0, 2)})
        state = GameState(remaining=("T1", "T2"))
        moved = apply_action(state, "T2", distances)
        self.assertEqual(moved.remaining, ("T1",))
        self.assertEqual(moved.score_a, TREASURE_VALUES["T2"] - 2)

    def test_negative_gain_is_a_legal_forced_action(self):
        distances = {"A": {"T1": 7}, "B": {"T1": 9}, "T1": {"T1": 0}}
        values = {"T1": 2}
        for player, expected in (("A", -5), ("B", 7)):
            state = GameState(remaining=("T1",), current_player=player)
            self.assert_searches_agree(state, distances, 4, values)
            result = minimax(state, distances, 4, values=values)
            self.assertEqual((result.best_target, result.best_value, result.line), ("T1", expected, ("T1",)))

    def test_depth_counts_individual_actions_and_terminal_utility(self):
        state = GameState(score_a=11, score_b=7)
        for depth in (0, 1, 2, 4, 8):
            self.assert_searches_agree(state, self.distances, depth)
        for algorithm in (minimax, alpha_beta):
            self.assertEqual(algorithm(state, self.distances, 0).best_value, 4)
            self.assertEqual(len(algorithm(state, self.distances, 1).line), 1)
            self.assertEqual(len(algorithm(state, self.distances, 2).line), 2)
            self.assertEqual(len(algorithm(state, self.distances, 8).line), 4)
            terminal = GameState(remaining=(), score_a=-3, score_b=5)
            result = algorithm(terminal, self.distances, 4)
            self.assertEqual((result.best_target, result.best_value, result.line), (None, -8, ()))

    def test_unreachable_actions_and_no_available_action(self):
        distances = {"A": {"T1": -1, "T2": 4}, "B": {"T1": -1, "T2": -1}, "T2": {"T1": -1}}
        state = GameState(remaining=("T1", "T2"), score_a=2, score_b=1)
        self.assert_searches_agree(state, distances, 4)
        for algorithm in (minimax, alpha_beta):
            result = algorithm(state, distances, 4)
            self.assertEqual((result.best_target, result.best_value, result.line), ("T2", 3, ("T2",)))
            stuck = GameState(remaining=("T1",), score_a=2, score_b=1)
            result = algorithm(stuck, distances, 4)
            self.assertEqual((result.best_target, result.best_value, result.line), (None, 1, ()))
        with self.assertRaises(ValueError):
            apply_action(state, "T1", distances)
        with self.assertRaises(ValueError):
            apply_action(state, "T3", distances)

    def test_ties_follow_search_order_and_gain_is_player_relative(self):
        values = {"T1": 5, "T2": 5}
        distances = {source: {"T1": 1, "T2": 1} for source in ("A", "B", "T1", "T2")}
        state = GameState(remaining=("T1", "T2"))
        for order, expected in (("fixed", "T1"), ("reverse", "T2"), ("gain", "T1")):
            self.assert_searches_agree(state, distances, 2, values, order)
            self.assertEqual(minimax(state, distances, 2, values=values, order=order).best_target, expected)
        distances["A"] = {"T1": 1, "T2": 4}
        distances["B"] = {"T1": 4, "T2": 1}
        # B 也按自己的即时收益降序；不能沿用 A 的位置或 U 的降序。
        b_turn = GameState(remaining=("T1", "T2"), current_player="B")
        self.assert_searches_agree(b_turn, distances, 1, values, "gain")
        self.assertEqual(minimax(b_turn, distances, 1, values=values, order="gain").best_target, "T2")

    def test_first_move_values_are_exact_for_each_branch(self):
        state = GameState()
        self.assertEqual(first_move_values(state, self.distances, 0), {})
        for depth in (2, 4):
            expected = {}
            for target in state.remaining:
                # 直接构造后继作为独立 oracle 的根，不使用 apply_action。
                child = GameState(
                    pos_a=target,
                    remaining=tuple(t for t in state.remaining if t != target),
                    score_a=TREASURE_VALUES[target] - self.distances["A"][target],
                    current_player="B",
                )
                expected[target] = exhaustive_game(child, self.distances, depth - 1)[1]
            self.assertEqual(first_move_values(state, self.distances, depth), expected)

    def test_statistics_reset_at_each_search_entry(self):
        stats = SearchStats()
        for previous in (minimax, alpha_beta):
            for following in (minimax, alpha_beta):
                previous(GameState(), self.distances, 4, stats=stats)
                result = following(GameState(score_a=7, score_b=2), self.distances, 0, stats=stats)
                self.assertEqual(result.best_value, 5)
                self.assertEqual((stats.visited, stats.expanded, stats.leaves, stats.cutoffs, stats.skipped_children), (1, 0, 1, 0, 0))

    def test_invalid_search_options_are_rejected(self):
        for algorithm in (minimax, alpha_beta):
            for depth in (-1, 1.5, True):
                with self.subTest(algorithm=algorithm.__name__, depth=depth), self.assertRaises(ValueError):
                    algorithm(GameState(), self.distances, cast(int, depth))
            with self.assertRaises(ValueError):
                algorithm(GameState(), self.distances, 2, order="unknown")
            with self.assertRaises(ValueError):
                algorithm(GameState(current_player="C"), self.distances, 2)

    def test_random_asymmetric_games_against_full_tree_oracle(self):
        rng = random.Random(20261103)
        for case in range(100):
            names = tuple(f"T{i + 1}" for i in range(rng.randint(1, 4)))
            values = {name: rng.randint(0, 12) for name in names}
            distances = {
                source: {target: (-1 if rng.random() < 0.15 else rng.randint(0, 14)) for target in names}
                for source in ("A", "B") + names
            }
            state = GameState(
                pos_a=rng.choice(("A",) + names), pos_b=rng.choice(("B",) + names),
                remaining=names, score_a=rng.randint(-10, 10), score_b=rng.randint(-10, 10),
                current_player=rng.choice(("A", "B")),
            )
            depth = rng.randint(0, 6)
            for order in ("fixed", "reverse", "gain"):
                with self.subTest(case=case, depth=depth, order=order):
                    self.assert_searches_agree(state, distances, depth, values, order)


if __name__ == "__main__":
    unittest.main()
