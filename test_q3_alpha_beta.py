"""用独立的自底向上求值验证 Minimax、剪枝行为和搜索统计。"""

import random
import unittest

from q3_alpha_beta import (
    SearchStats,
    TreeNode,
    alpha_beta,
    build_example_tree,
    minimax,
)


def branch(*children):
    """用数字创建叶子，用 TreeNode 创建子树。"""
    return TreeNode(children=[
        child if isinstance(child, TreeNode) else TreeNode(value=child)
        for child in children
    ])


def bottom_up_value(root):
    """先展开所有路径，再倒序求值，不使用递归或 alpha/beta。"""
    occurrences = [(root, 0, [])]
    index = 0
    while index < len(occurrences):
        node, depth, child_indexes = occurrences[index]
        for child in node.children:
            child_indexes.append(len(occurrences))
            occurrences.append((child, depth + 1, []))
        index += 1

    scores = [None] * len(occurrences)
    for index in range(len(occurrences) - 1, -1, -1):
        node, depth, child_indexes = occurrences[index]
        if not child_indexes:
            if node.value is None:
                raise ValueError("A leaf must have a value.")
            scores[index] = node.value
        else:
            child_scores = [scores[child_index] for child_index in child_indexes]
            scores[index] = max(child_scores) if depth % 2 == 0 else min(child_scores)
    return scores[0], len(occurrences)


def snapshot(root):
    """同时记录节点值、孩子顺序、共享关系及原 children 列表身份。"""
    records = []
    pending = [root]
    seen = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        records.append((
            id(node), node.value, id(node.children),
            tuple(id(child) for child in node.children),
        ))
        pending.extend(node.children)
    return sorted(records)


class AlphaBetaTests(unittest.TestCase):
    def assert_both_agree(self, root, expected=None):
        original = snapshot(root)
        oracle, full_visits = bottom_up_value(root)
        if expected is not None:
            self.assertEqual(oracle, expected)

        for search in (minimax, alpha_beta):
            stats = SearchStats()
            with self.subTest(search=search.__name__):
                self.assertEqual(search(root), oracle)
                self.assertEqual(search(root, stats=stats), oracle)
                self.assertEqual(snapshot(root), original)
                self.assertEqual(stats.visited, stats.expanded + stats.leaves)
                self.assertTrue(all(isinstance(event, str) for event in stats.trace))
                self.assertLessEqual(stats.visited, full_visits)
                if search is minimax:
                    self.assertEqual(stats.visited, full_visits)
                    self.assertEqual((stats.cutoffs, stats.skipped_children), (0, 0))

    def test_official_example_and_statistics(self):
        root = build_example_tree()
        # PDF 构建代码中的 MAX(6, 9) 被两个 MIN 节点共同引用。
        self.assertIs(root.children[0].children[1], root.children[1].children[0])
        self.assert_both_agree(root, 5)

        plain = SearchStats()
        pruned = SearchStats()
        self.assertEqual(minimax(root, stats=plain), 5)
        self.assertEqual(alpha_beta(root, stats=pruned), 5)
        self.assertEqual((plain.visited, plain.expanded, plain.leaves), (15, 7, 8))
        self.assertEqual((pruned.visited, pruned.expanded, pruned.leaves), (14, 7, 7))
        self.assertEqual((pruned.cutoffs, pruned.skipped_children), (1, 1))

    def test_single_leaf_including_zero_and_negative_value(self):
        for value in (-7, 0, 13):
            with self.subTest(value=value):
                root = TreeNode(value=value)
                self.assert_both_agree(root, value)
                for search in (minimax, alpha_beta):
                    stats = SearchStats()
                    self.assertEqual(search(root, stats=stats), value)
                    self.assertEqual((stats.visited, stats.expanded, stats.leaves), (1, 0, 1))
                    self.assertEqual((stats.cutoffs, stats.skipped_children), (0, 0))

    def test_negative_values_zero_and_ties(self):
        self.assert_both_agree(branch(branch(-3, 0, -3), branch(-2, -2), branch(0, 0)), 0)
        self.assert_both_agree(branch(branch(-9, -7), branch(-4, -6)), -6)
        self.assert_both_agree(branch(branch(4, 4), branch(4, 4)), 4)

    def test_max_node_cuts_off_remaining_child(self):
        # MIN 已经有 5；第二个 MAX 看见 6 后，无需读取 9。
        root = branch(branch(branch(3, 5), branch(6, 9)))
        self.assert_both_agree(root, 5)
        stats = SearchStats()
        self.assertEqual(alpha_beta(root, stats=stats), 5)
        self.assertEqual((stats.visited, stats.expanded, stats.leaves), (7, 4, 3))
        self.assertEqual((stats.cutoffs, stats.skipped_children), (1, 1))

    def test_min_node_cuts_off_multiple_remaining_children(self):
        # 根 MAX 已经有 5；第二个 MIN 看见 4 后，跳过两个直接子树。
        # skipped_children 只记 2 个直接孩子，不把它们的后代也相加。
        root = branch(branch(5, 7), branch(4, branch(100, 200), branch(300, 400)))
        self.assert_both_agree(root, 5)
        stats = SearchStats()
        self.assertEqual(alpha_beta(root, stats=stats), 5)
        self.assertEqual((stats.visited, stats.expanded, stats.leaves), (6, 3, 3))
        self.assertEqual((stats.cutoffs, stats.skipped_children), (1, 2))

    def test_equality_is_enough_to_prune(self):
        root = branch(branch(5), branch(5, 100))
        self.assert_both_agree(root, 5)
        stats = SearchStats()
        self.assertEqual(alpha_beta(root, stats=stats), 5)
        self.assertEqual((stats.visited, stats.leaves), (5, 2))
        self.assertEqual((stats.cutoffs, stats.skipped_children), (1, 1))

    def test_no_cutoff_is_counted_after_the_last_child(self):
        # 即使最后一个孩子让 alpha >= beta，也没有实际省略搜索。
        for root, expected_visits in (
            (branch(branch(5), branch(4)), 5),
            (branch(branch(branch(5), branch(6))), 6),
        ):
            with self.subTest(expected_visits=expected_visits):
                self.assert_both_agree(root, 5)
                stats = SearchStats()
                self.assertEqual(alpha_beta(root, stats=stats), 5)
                self.assertEqual(stats.visited, expected_visits)
                self.assertEqual((stats.cutoffs, stats.skipped_children), (0, 0))

    def test_inherited_alpha_does_not_prune_max_children_too_early(self):
        # 右侧 MAX(1, 2) 继承 alpha=5、beta=9，仍应完整读取 1、2。
        # 它的局部值是 2；上层 MIN 得到 2，根最后保留已有的 5。
        root = branch(
            branch(branch(5)),
            branch(branch(6, 9), branch(1, 2)),
        )
        self.assert_both_agree(root, 5)
        stats = SearchStats()
        self.assertEqual(alpha_beta(root, stats=stats), 5)
        self.assertEqual((stats.visited, stats.expanded, stats.leaves), (11, 6, 5))
        self.assertEqual((stats.cutoffs, stats.skipped_children), (0, 0))

    def test_shared_node_is_evaluated_once_per_path(self):
        # 把最小值放在最后，保证第二次访问也不会提前跳过孩子。
        shared = branch(8, 2)
        root = branch(shared, shared)
        self.assert_both_agree(root, 2)
        for search in (minimax, alpha_beta):
            stats = SearchStats()
            self.assertEqual(search(root, stats=stats), 2)
            self.assertEqual((stats.visited, stats.expanded, stats.leaves), (7, 3, 4))
            self.assertIs(root.children[0], root.children[1])

    def test_statistics_are_reset_for_each_entry_call(self):
        stats = SearchStats()
        for previous_search in (minimax, alpha_beta):
            for next_search in (minimax, alpha_beta):
                with self.subTest(previous=previous_search.__name__, next=next_search.__name__):
                    previous_search(build_example_tree(), stats=stats)
                    stats.trace.append("sentinel from previous search")
                    self.assertEqual(next_search(TreeNode(value=0), stats=stats), 0)
                    self.assertEqual((stats.visited, stats.expanded, stats.leaves), (1, 0, 1))
                    self.assertEqual((stats.cutoffs, stats.skipped_children), (0, 0))
                    self.assertNotIn("sentinel from previous search", stats.trace)

    def test_leaf_without_value_is_rejected(self):
        for search in (minimax, alpha_beta):
            with self.subTest(search=search.__name__):
                with self.assertRaises(ValueError):
                    search(TreeNode())
                with self.assertRaises(ValueError):
                    search(branch(TreeNode()))

    def test_random_trees_against_bottom_up_oracle(self):
        rng = random.Random(20261030)

        def make_tree(depth):
            if depth == 0 or rng.random() < 0.28:
                return TreeNode(value=rng.randint(-30, 30))
            return TreeNode(children=[make_tree(depth - 1) for _ in range(rng.randint(1, 4))])

        for case in range(150):
            with self.subTest(case=case):
                self.assert_both_agree(make_tree(rng.randint(1, 5)))


if __name__ == "__main__":
    unittest.main()
