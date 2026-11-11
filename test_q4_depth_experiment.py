"""Checks for the shallow-search experiment and its decision-loss metric.

The reference evaluator scores action permutations directly, without calling
the game's transition function or either production game-search algorithm.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import permutations
import random
import unittest
from unittest.mock import patch

import q4_treasure as game
from q4_depth_experiment import (
    ExperimentCase,
    TIE_POLICIES,
    analyze_case,
    evaluate_tie_policies,
    make_cases,
    oracle_action_values,
    select_tied_action,
    summarize,
)


TARGETS = ("T1", "T2", "T3", "T4")


def reference_line_value(line, distances, values):
    """Score one legal sequence, independently of GameState/apply_action."""
    positions = ["A", "B"]
    used = set()
    utility = 0
    for index, target in enumerate(line):
        if target not in TARGETS or target in used:
            raise AssertionError(f"illegal action sequence: {line!r}")
        player = index % 2
        cost = distances[positions[player]][target]
        if cost < 0:
            raise AssertionError(f"unreachable action sequence: {line!r}")
        gain = values[target] - cost
        utility += gain if player == 0 else -gain
        positions[player] = target
        used.add(target)
    return utility


def reference_action_values(distances, values, depth):
    """Enumerate leaves, then reduce their prefixes from the bottom up."""
    if depth == 0:
        return {}
    depth = min(depth, len(TARGETS))
    frontier = {
        line: reference_line_value(line, distances, values)
        for line in permutations(TARGETS, depth)
    }
    for prefix_length in range(depth - 1, 0, -1):
        buckets = {}
        for line, value in frontier.items():
            buckets.setdefault(line[:prefix_length], []).append(value)
        reducer = max if prefix_length % 2 == 0 else min
        frontier = {prefix: reducer(scores) for prefix, scores in buckets.items()}
    return {target: frontier[(target,)] for target in TARGETS}


class DepthExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances = game.compute_distances()

    def case(self, values, name="fixture"):
        return ExperimentCase(
            name=name,
            family="test",
            values=dict(zip(TARGETS, values)),
        )

    def test_official_decision_loss_uses_full_horizon_values(self):
        row = analyze_case(
            ExperimentCase("official", "official", dict(game.TREASURE_VALUES)),
            self.distances,
        )
        self.assertEqual(row["shallow"]["depth"], 2)
        self.assertEqual(row["full"]["depth"], 4)
        self.assertEqual(row["shallow"]["chosen_target"], "T2")
        self.assertEqual(row["shallow"]["predicted_value"], -1)
        self.assertEqual(row["full"]["chosen_target"], "T3")
        self.assertEqual(row["full"]["predicted_value"], 3)
        self.assertEqual(row["full"]["action_values"],
                         {"T1": -5, "T2": 1, "T3": 3, "T4": -7})
        self.assertEqual(row["shallow"]["full_value_after_chosen"], 1)
        # Comparing the H=2 prediction -1 with V4=3 would give the wrong loss 4.
        self.assertEqual(row["regret"], 2)
        self.assertTrue(row["mistake"])
        self.assertEqual(row["tie_class"], "all_wrong")

    def test_oracle_matches_independent_permutation_evaluator(self):
        rng = random.Random(20261103)
        for _ in range(25):
            values = {target: rng.randint(1, 20) for target in TARGETS}
            for depth in range(5):
                with self.subTest(values=values, depth=depth):
                    self.assertEqual(
                        oracle_action_values(self.distances, values, depth),
                        reference_action_values(self.distances, values, depth),
                    )

    def test_oracle_handles_directed_distances_and_second_moves(self):
        rng = random.Random(3819)
        labels = ("A", "B") + TARGETS
        for _ in range(6):
            distances = {
                source: {
                    target: 0 if source == target else rng.randint(1, 15)
                    for target in labels
                }
                for source in labels
            }
            # Floyd-Warshall makes these a valid directed shortest-cost matrix.
            for via in labels:
                for source in labels:
                    for target in labels:
                        distances[source][target] = min(
                            distances[source][target],
                            distances[source][via] + distances[via][target],
                        )
            values = {target: rng.randint(1, 20) for target in TARGETS}
            for depth in (2, 4):
                self.assertEqual(
                    oracle_action_values(distances, values, depth),
                    reference_action_values(distances, values, depth),
                )

    def test_different_first_moves_can_both_be_optimal(self):
        row = analyze_case(self.case((1, 1, 1, 9)), self.distances)
        self.assertEqual(row["shallow"]["chosen_target"], "T4")
        self.assertEqual(row["full"]["chosen_target"], "T2")
        self.assertEqual(row["full"]["optimal_targets"], ["T2", "T4"])
        self.assertTrue(row["different_first_move"])
        self.assertEqual(row["regret"], 0)
        self.assertFalse(row["mistake"])
        self.assertEqual(row["tie_class"], "all_correct")

    def test_mixed_shallow_tie_records_possible_loss(self):
        row = analyze_case(self.case((1, 1, 1, 5)), self.distances)
        self.assertEqual(row["shallow"]["optimal_targets"], ["T2", "T4"])
        self.assertEqual(row["shallow"]["chosen_target"], "T2")
        self.assertEqual(row["tie_class"], "mixed")
        self.assertEqual(row["best_tie_regret"], 0)
        self.assertEqual(row["worst_tie_regret"], 4)
        self.assertEqual(row["regret"], 0)
        self.assertFalse(row["mistake"])

    def test_every_shallow_tied_choice_can_be_wrong(self):
        row = analyze_case(self.case((4, 1, 2, 5)), self.distances)
        self.assertEqual(row["shallow"]["optimal_targets"], ["T2", "T4"])
        self.assertEqual(row["full"]["optimal_targets"], ["T3"])
        self.assertEqual(row["tie_class"], "all_wrong")
        self.assertEqual(row["best_tie_regret"], 2)
        self.assertEqual(row["worst_tie_regret"], 6)
        self.assertEqual(row["regret"], 2)
        self.assertTrue(row["mistake"])

    def test_every_shallow_tied_choice_can_be_correct(self):
        row = analyze_case(self.case((1, 1, 1, 1)), self.distances)
        self.assertEqual(row["shallow"]["optimal_targets"], ["T2", "T3"])
        self.assertEqual(row["tie_class"], "all_correct")
        self.assertEqual(row["best_tie_regret"], 0)
        self.assertEqual(row["worst_tie_regret"], 0)

    def test_reported_lines_replay_and_forced_first_step_is_preserved(self):
        rng = random.Random(168)
        fixtures = [dict(game.TREASURE_VALUES)] + [
            {target: rng.randint(1, 20) for target in TARGETS}
            for _ in range(20)
        ]
        for values in fixtures:
            case = ExperimentCase("replay", "test", values)
            row = analyze_case(case, self.distances)
            for section, length in (("shallow", 2), ("full", 4)):
                result = row[section]
                with self.subTest(values=values, section=section):
                    self.assertIsInstance(result["line"], list)
                    self.assertEqual(len(result["line"]), length)
                    self.assertEqual(result["line"][0], result["chosen_target"])
                    self.assertEqual(
                        reference_line_value(result["line"], self.distances, values),
                        result["predicted_value"],
                    )
                    self.assertGreater(result["visited"], 0)
            forced_line = row["shallow"]["full_line_after_chosen"]
            self.assertEqual(len(forced_line), 4)
            self.assertEqual(forced_line[0], row["shallow"]["chosen_target"])
            self.assertEqual(
                reference_line_value(forced_line, self.distances, values),
                row["shallow"]["full_value_after_chosen"],
            )

    def test_random_metrics_and_tie_classes_follow_exact_action_values(self):
        rng = random.Random(6821)
        for index in range(30):
            values = {target: rng.randint(1, 20) for target in TARGETS}
            row = analyze_case(ExperimentCase(str(index), "test", values), self.distances)
            q2 = reference_action_values(self.distances, values, 2)
            q4 = reference_action_values(self.distances, values, 4)
            shallow_best = [target for target in TARGETS if q2[target] == max(q2.values())]
            full_best = [target for target in TARGETS if q4[target] == max(q4.values())]
            regrets = [max(q4.values()) - q4[target] for target in shallow_best]
            expected_class = (
                "all_correct" if max(regrets) == 0
                else "all_wrong" if min(regrets) > 0
                else "mixed"
            )
            with self.subTest(values=values):
                self.assertEqual(row["shallow"]["action_values"], q2)
                self.assertEqual(row["full"]["action_values"], q4)
                self.assertEqual(row["shallow"]["optimal_targets"], shallow_best)
                self.assertEqual(row["full"]["optimal_targets"], full_best)
                self.assertEqual(row["best_tie_regret"], min(regrets))
                self.assertEqual(row["worst_tie_regret"], max(regrets))
                self.assertEqual(row["regret"], regrets[0])
                self.assertEqual(row["mistake"], regrets[0] > 0)
                self.assertEqual(row["tie_class"], expected_class)

    def test_case_families_and_factorial_grid_are_complete_and_reproducible(self):
        self.assertEqual(len(make_cases()), 1 + 4 * 20 + 20 * 20 + 1000)
        cases = make_cases(value_min=2, value_max=4, random_cases=7, seed=47)
        self.assertEqual(cases, make_cases(value_min=2, value_max=4, random_cases=7, seed=47))
        self.assertEqual(len(cases), 1 + 4 * 3 + 3 * 3 + 7)
        self.assertEqual(Counter(case.family for case in cases), {
            "official": 1,
            "scan_T1": 3, "scan_T2": 3, "scan_T3": 3, "scan_T4": 3,
            "grid_T2_T3": 9, "random": 7,
        })
        for target in TARGETS:
            scans = [case for case in cases if case.family == f"scan_{target}"]
            self.assertEqual(sorted(case.values[target] for case in scans), [2, 3, 4])
            for case in scans:
                for other in TARGETS:
                    if other != target:
                        self.assertEqual(case.values[other], game.TREASURE_VALUES[other])
        grid = [case for case in cases if case.family == "grid_T2_T3"]
        self.assertEqual(
            {(case.values["T2"], case.values["T3"]) for case in grid},
            {(second, third) for second in range(2, 5) for third in range(2, 5)},
        )
        for case in grid:
            self.assertEqual(case.values["T1"], game.TREASURE_VALUES["T1"])
            self.assertEqual(case.values["T4"], game.TREASURE_VALUES["T4"])
        random_values = [case.values for case in cases if case.family == "random"]
        self.assertTrue(all(2 <= value <= 4 for values in random_values for value in values.values()))
        changed_seed = make_cases(value_min=2, value_max=4, random_cases=7, seed=48)
        self.assertNotEqual(random_values, [case.values for case in changed_seed if case.family == "random"])

    def test_analysis_does_not_mutate_inputs_and_can_skip_verification(self):
        values = dict(game.TREASURE_VALUES)
        distances = deepcopy(self.distances)
        case = ExperimentCase("immutability", "test", values, {"T1": 9}, 20261103)
        before = deepcopy((case, distances, game.TREASURE_VALUES))
        verified = analyze_case(case, distances)
        unchecked = analyze_case(case, distances, verify=False)
        self.assertEqual(verified, unchecked)
        self.assertEqual((case, distances, game.TREASURE_VALUES), before)
        self.assertEqual(verified["name"], case.name)
        self.assertEqual(verified["family"], case.family)
        self.assertEqual(verified["values"], case.values)
        self.assertEqual(verified["parameters"], case.parameters)
        self.assertEqual(verified["seed"], case.seed)

    def test_verification_rejects_disagreement_with_independent_oracle(self):
        def inconsistent_oracle(distances, values, depth):
            return {
                target: value + 99
                for target, value in reference_action_values(distances, values, depth).items()
            }

        with patch("q4_depth_experiment.oracle_action_values", side_effect=inconsistent_oracle):
            with self.assertRaises(AssertionError):
                analyze_case(self.case((9, 6, 7, 7)), self.distances)

    def test_tie_rules_filter_on_h2_before_applying_preference(self):
        # T2 is nearest and T4 has the largest immediate gain, but neither
        # may replace the unique H=2 optimum T1.
        q2 = {"T1": 3, "T2": 0, "T3": 1, "T4": 2}
        values = {"T1": 1, "T2": 1, "T3": 1, "T4": 100}
        for policy in TIE_POLICIES:
            with self.subTest(policy=policy):
                self.assertEqual(select_tied_action(q2, self.distances, values, policy), "T1")

    def test_tie_rules_have_canonical_secondary_order(self):
        # Dict insertion order must not silently redefine the tie rule.
        q2 = {target: 0 for target in reversed(TARGETS)}
        distances = deepcopy(self.distances)
        distances["A"] = {target: 4 for target in TARGETS}
        values = {target: 10 for target in TARGETS}
        expected = {"default": "T1", "reverse": "T4", "nearest": "T1", "gain": "T1"}
        before = deepcopy((q2, distances, values))
        for policy, target in expected.items():
            self.assertEqual(select_tied_action(q2, distances, values, policy), target)
        self.assertEqual((q2, distances, values), before)

    def test_full_horizon_scores_cannot_change_tie_policy_selection(self):
        q2 = {"T1": 5, "T2": 5, "T3": 0, "T4": -1}
        values = dict(game.TREASURE_VALUES)
        first = evaluate_tie_policies(
            q2, {"T1": 100, "T2": -100, "T3": 0, "T4": 0}, self.distances, values,
        )
        second = evaluate_tie_policies(
            q2, {"T1": -100, "T2": 100, "T3": 0, "T4": 0}, self.distances, values,
        )
        self.assertEqual(
            {policy: result["chosen_target"] for policy, result in first.items()},
            {policy: result["chosen_target"] for policy, result in second.items()},
        )
        # Q4 changes the measured loss, while the shallow choice stays fixed.
        self.assertEqual(first["default"]["regret"], 0)
        self.assertEqual(second["default"]["regret"], 200)

    def test_threshold_tie_policies_and_losses(self):
        row = analyze_case(self.case((9, 5, 7, 7)), self.distances)
        expected = {
            "default": ("T2", 2), "reverse": ("T3", 0),
            "nearest": ("T2", 2), "gain": ("T2", 2),
        }
        for policy, (target, regret) in expected.items():
            result = row["tie_policies"][policy]
            self.assertEqual(result["chosen_target"], target)
            self.assertEqual(result["regret"], regret)
            self.assertEqual(result["predicted_value"], row["shallow"]["predicted_value"])
        self.assertEqual(row["tie_policies"]["default"]["regret"], row["regret"])

    def test_factorial_tie_policy_results_match_report(self):
        # Recompute all 400 combinations rather than loading saved results.
        cases = [case for case in make_cases(random_cases=0) if case.family == "grid_T2_T3"]
        rows = [analyze_case(case, self.distances, verify=False) for case in cases]
        result = summarize(rows)["grid_T2_T3"]
        self.assertEqual(result["cases"], 400)
        expected = {
            "default": (27, 0.290, 0), "reverse": (21, 0.190, 6),
            "nearest": (20, 0.170, 7), "gain": (23, 0.230, 4),
        }
        for policy, (mistakes, mean_regret, repaired) in expected.items():
            stats = result["tie_policies"][policy]
            with self.subTest(policy=policy):
                self.assertEqual(stats["mistakes"], mistakes)
                self.assertAlmostEqual(stats["mean_regret"], mean_regret)
                self.assertEqual(stats["repaired_mistakes"], repaired)
                self.assertEqual(stats["introduced_mistakes"], 0)


if __name__ == "__main__":
    unittest.main()
