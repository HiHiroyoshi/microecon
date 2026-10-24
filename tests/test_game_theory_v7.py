"""microecon.game_theory v0.7.0 の理論整合性テスト.

Lemke-Howson アルゴリズム（一般形ゲーム）、連続戦略ゲーム（Cournot 複占）、
展開型ゲーム（後方帰納法による部分ゲーム完全均衡）、および不変データクラスを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from microecon.game_theory import (
    ContinuousGame,
    ExtensiveFormGame,
    ExtensiveFormNode,
    LemkeHowsonSolver,
    MixedStrategyProfile,
    NashSolver,
    NormalFormGame,
)

APPROX_ABS = 1e-4
INF = float("inf")


# --------------------------------------------------------------------------
# 1. 不変データクラス
# --------------------------------------------------------------------------


class TestImmutableDataclasses:
    def test_mixed_profile_equality_does_not_raise(self) -> None:
        lhs = MixedStrategyProfile(
            probabilities=(np.array([0.5, 0.5]), np.array([0.25, 0.75])),
            expected_payoffs=(1.0, 2.0),
        )
        rhs = MixedStrategyProfile(
            probabilities=(np.array([0.5, 0.5]), np.array([0.25, 0.75])),
            expected_payoffs=(1.0, 2.0),
        )
        other = MixedStrategyProfile(
            probabilities=(np.array([0.5, 0.5]), np.array([0.75, 0.25])),
            expected_payoffs=(1.0, 2.0),
        )
        assert lhs == rhs
        assert lhs != other
        assert lhs != "not a profile"

    def test_mixed_profile_copies_and_freezes_arrays(self) -> None:
        source = np.array([0.5, 0.5])
        profile = MixedStrategyProfile(
            probabilities=(source, source), expected_payoffs=(0.0, 0.0)
        )
        source[0] = 999.0
        assert profile.probabilities[0][0] == pytest.approx(0.5, abs=APPROX_ABS)
        with pytest.raises(ValueError, match="read-only"):
            profile.probabilities[0][0] = 1.0

    def test_extensive_node_children_are_read_only(self) -> None:
        children = {"L": "a", "R": "b"}
        node = ExtensiveFormNode("root", player_idx=0, actions=("L", "R"), children=children)
        children["L"] = "mutated"
        assert node.children is not None
        assert node.children["L"] == "a"
        with pytest.raises(TypeError):
            node.children["L"] = "c"  # type: ignore[index]


# --------------------------------------------------------------------------
# 2. Lemke-Howson アルゴリズム
# --------------------------------------------------------------------------


class TestLemkeHowsonMatchingPennies:
    """コインの表裏合わせ（2x2 ゼロサム）: 解析解 (0.5, 0.5) との一致."""

    @pytest.fixture
    def solver(self) -> LemkeHowsonSolver:
        matrix_a = np.array([[1.0, -1.0], [-1.0, 1.0]])
        return LemkeHowsonSolver(matrix_a, -matrix_a)

    def test_unique_mixed_equilibrium(self, solver: LemkeHowsonSolver) -> None:
        p, q = solver.solve(initial_dropped_label=0)
        assert tuple(p) == pytest.approx((0.5, 0.5), abs=APPROX_ABS)
        assert tuple(q) == pytest.approx((0.5, 0.5), abs=APPROX_ABS)

    def test_expected_payoffs_use_original_matrices(self, solver: LemkeHowsonSolver) -> None:
        p, q = solver.solve(initial_dropped_label=0)
        assert solver.expected_payoffs(p, q) == pytest.approx((0.0, 0.0), abs=APPROX_ABS)

    @pytest.mark.parametrize("label", [-1, 4, 100])
    def test_invalid_initial_label_raises(self, solver: LemkeHowsonSolver, label: int) -> None:
        with pytest.raises(ValueError):
            solver.solve(initial_dropped_label=label)


class TestLemkeHowsonBattleOfTheSexes:
    """男女の争い: どの初期ラベルからも純粋均衡に到達し、指数 -1 の
    混合均衡 (2/3, 1/3) には到達しない（アルゴリズムの理論的限界の文書化）."""

    MATRIX_A = np.array([[2.0, 0.0], [0.0, 1.0]])
    MATRIX_B = np.array([[1.0, 0.0], [0.0, 2.0]])

    @pytest.mark.parametrize("label", [0, 1, 2, 3])
    def test_reaches_only_pure_equilibria(self, label: int) -> None:
        solver = LemkeHowsonSolver(self.MATRIX_A, self.MATRIX_B)
        p, q = solver.solve(initial_dropped_label=label)
        pure_equilibria = [((1.0, 0.0), (1.0, 0.0)), ((0.0, 1.0), (0.0, 1.0))]
        assert any(
            tuple(p) == pytest.approx(pe_p, abs=APPROX_ABS)
            and tuple(q) == pytest.approx(pe_q, abs=APPROX_ABS)
            for pe_p, pe_q in pure_equilibria
        )
        assert p[0] != pytest.approx(2.0 / 3.0, abs=APPROX_ABS)

    def test_payoffs_are_unscaled(self) -> None:
        solver = LemkeHowsonSolver(self.MATRIX_A, self.MATRIX_B)
        p, q = solver.solve(initial_dropped_label=0)
        # A, B は 0 を含むため内部で k=1 の正値化が行われるが、利得は元の行列で評価される
        assert solver.expected_payoffs(p, q) == pytest.approx((2.0, 1.0), abs=APPROX_ABS)


class TestLemkeHowsonDegenerate2x3:
    """非正方・非対称 2x3 退化ゲーム: 辞書式ピボットによる正常終了と
    線分状均衡 (p*=(2/3,1/3), q0 = q1 + 2 q2) の代表頂点の取得."""

    MATRIX_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
    MATRIX_B = np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 0.0]])

    @pytest.mark.parametrize("label", range(5))
    def test_representative_vertex(self, label: int) -> None:
        solver = LemkeHowsonSolver(self.MATRIX_A, self.MATRIX_B)
        p, q = solver.solve(initial_dropped_label=label)
        assert tuple(p) == pytest.approx((2.0 / 3.0, 1.0 / 3.0), abs=APPROX_ABS)
        assert q.sum() == pytest.approx(1.0, abs=APPROX_ABS)
        assert q[0] == pytest.approx(q[1] + 2.0 * q[2], abs=APPROX_ABS)
        vertices = [(0.5, 0.5, 0.0), (2.0 / 3.0, 0.0, 1.0 / 3.0)]
        assert any(tuple(q) == pytest.approx(v, abs=APPROX_ABS) for v in vertices)

    def test_nash_solver_returns_single_representative(self) -> None:
        game = NormalFormGame(payoff_matrices=(self.MATRIX_A, self.MATRIX_B))
        res = NashSolver(game).solve()
        assert res.pure_nash_equilibria == ()
        assert len(res.mixed_nash_equilibria) == 1
        profile = res.mixed_nash_equilibria[0]
        assert profile.expected_payoffs == pytest.approx(
            (2.0 / 3.0, 2.0 / 3.0), abs=APPROX_ABS
        )


class TestNashSolverMxN:
    def test_rock_paper_scissors(self) -> None:
        matrix_a = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        game = NormalFormGame(payoff_matrices=(matrix_a, -matrix_a))
        res = NashSolver(game).solve()
        assert res.is_zero_sum is True
        assert res.pure_nash_equilibria == ()
        assert len(res.mixed_nash_equilibria) == 1
        profile = res.mixed_nash_equilibria[0]
        for probs in profile.probabilities:
            assert tuple(probs) == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=APPROX_ABS)
        assert profile.expected_payoffs == pytest.approx((0.0, 0.0), abs=APPROX_ABS)

    def test_2x2_partial_boundary_solution_is_not_reported(self) -> None:
        """2x2 解析解は完全混合 (0 < p, q < 1) のみを返す（囚人のジレンマでは空）."""
        matrix_a = np.array([[-1.0, -3.0], [0.0, -2.0]])
        matrix_b = np.array([[-1.0, 0.0], [-3.0, -2.0]])
        game = NormalFormGame(payoff_matrices=(matrix_a, matrix_b))
        assert NashSolver(game).find_mixed_nash() == ()


# --------------------------------------------------------------------------
# 3. 連続戦略ゲーム（Cournot 複占）
# --------------------------------------------------------------------------


def _cournot(a: float, b: float, c1: float, c2: float) -> ContinuousGame:
    q1, q2 = sp.symbols("q1 q2")
    price = a - b * (q1 + q2)
    return ContinuousGame(
        payoff_expressions=((price - c1) * q1, (price - c2) * q2),
        strategy_variables=(q1, q2),
        strategy_bounds=((0.0, INF), (0.0, INF)),
    )


class TestCournotDuopoly:
    def test_interior_solution(self) -> None:
        results = _cournot(a=100.0, b=1.0, c1=10.0, c2=10.0).find_pure_nash()
        assert len(results) == 1
        res = results[0]
        assert res.is_corner is False
        assert res.strategies == pytest.approx((30.0, 30.0), abs=APPROX_ABS)
        assert res.payoffs == pytest.approx((900.0, 900.0), abs=APPROX_ABS)

    def test_corner_solution(self) -> None:
        a, b = 100.0, 1.0
        results = _cournot(a=a, b=b, c1=10.0, c2=120.0).find_pure_nash()
        assert len(results) == 1
        res = results[0]
        assert res.is_corner is True
        q1_star, q2_star = res.strategies
        assert q1_star == pytest.approx(45.0, abs=APPROX_ABS)
        assert q2_star == pytest.approx(0.0, abs=APPROX_ABS)
        assert a - b * (q1_star + q2_star) == pytest.approx(55.0, abs=APPROX_ABS)
        assert res.payoffs == pytest.approx((2025.0, 0.0), abs=APPROX_ABS)

    def test_best_response_expression(self) -> None:
        game = _cournot(a=100.0, b=1.0, c1=10.0, c2=10.0)
        q1, q2 = game.strategy_variables
        br1 = game.get_best_response_expression(0)
        assert sp.simplify(br1 - (45 - q2 / 2)) == 0
        br2 = game.get_best_response_expression(1)
        assert sp.simplify(br2 - (45 - q1 / 2)) == 0

    def test_best_response_with_symbolic_parameters(self) -> None:
        a, b, c = sp.symbols("a b c", positive=True)
        q1, q2 = sp.symbols("q1 q2")
        price = a - b * (q1 + q2)
        game = ContinuousGame(
            ((price - c) * q1, (price - c) * q2), (q1, q2), ((0.0, INF), (0.0, INF))
        )
        br1 = game.get_best_response_expression(0)
        assert sp.simplify(br1 - (a - c - b * q2) / (2 * b)) == 0
        with pytest.raises(ValueError):
            game.find_pure_nash()

    def test_finite_upper_bound_capacity_constraint(self) -> None:
        """プレイヤー1に生産能力上限 20 がある場合、上限での角点解となる."""
        q1, q2 = sp.symbols("q1 q2")
        price = 100 - (q1 + q2)
        game = ContinuousGame(
            ((price - 10) * q1, (price - 10) * q2), (q1, q2), ((0.0, 20.0), (0.0, INF))
        )
        results = game.find_pure_nash()
        assert len(results) == 1
        assert results[0].is_corner is True
        assert results[0].strategies == pytest.approx((20.0, 35.0), abs=APPROX_ABS)

    def test_invalid_player_idx(self) -> None:
        with pytest.raises(ValueError):
            _cournot(100.0, 1.0, 10.0, 10.0).get_best_response_expression(2)

    def test_invalid_bounds(self) -> None:
        q1, q2 = sp.symbols("q1 q2")
        with pytest.raises(ValueError):
            ContinuousGame((q1, q2), (q1, q2), ((1.0, 0.0), (0.0, 1.0)))


# --------------------------------------------------------------------------
# 4. 展開型ゲーム（後方帰納法）
# --------------------------------------------------------------------------


def _entry_game_nodes() -> dict[str, ExtensiveFormNode]:
    nodes = [
        ExtensiveFormNode(
            "root",
            player_idx=0,
            actions=("Enter", "Out"),
            children={"Enter": "n_enter", "Out": "n_out"},
        ),
        ExtensiveFormNode("n_out", player_idx=None, actions=(), payoffs=(0.0, 300.0)),
        ExtensiveFormNode(
            "n_enter",
            player_idx=1,
            actions=("Fight", "Acquiesce"),
            children={"Fight": "n_fight", "Acquiesce": "n_acq"},
        ),
        ExtensiveFormNode("n_fight", player_idx=None, actions=(), payoffs=(-100.0, 0.0)),
        ExtensiveFormNode("n_acq", player_idx=None, actions=(), payoffs=(100.0, 100.0)),
    ]
    return {node.node_id: node for node in nodes}


class TestMarketEntryGame:
    @pytest.fixture
    def game(self) -> ExtensiveFormGame:
        return ExtensiveFormGame("root", _entry_game_nodes(), num_players=2)

    def test_spne_removes_incredible_threat(self, game: ExtensiveFormGame) -> None:
        res = game.solve_spne(tie_breaking_rule="first")
        assert res.equilibrium_strategies[0]["root"] == "Enter"
        assert res.equilibrium_strategies[1]["n_enter"] == "Acquiesce"
        assert res.payoffs == pytest.approx((100.0, 100.0), abs=APPROX_ABS)

    def test_strategies_are_read_only(self, game: ExtensiveFormGame) -> None:
        res = game.solve_spne(tie_breaking_rule="first")
        with pytest.raises((TypeError, AttributeError)):
            res.equilibrium_strategies[0]["root"] = "Out"  # type: ignore[index]
        with pytest.raises((TypeError, AttributeError)):
            res.equilibrium_strategies[2] = {}  # type: ignore[index]

    def test_unsupported_tie_breaking_rule(self, game: ExtensiveFormGame) -> None:
        with pytest.raises(ValueError, match="Unsupported tie_breaking_rule 'random'"):
            game.solve_spne(tie_breaking_rule="random")


class TestTieBreaking:
    def test_first_action_selected_on_tie(self) -> None:
        nodes = {
            "root": ExtensiveFormNode(
                "root", player_idx=0, actions=("B", "A"), children={"B": "b", "A": "a"}
            ),
            "a": ExtensiveFormNode("a", player_idx=None, actions=(), payoffs=(1.0, 0.0)),
            "b": ExtensiveFormNode("b", player_idx=None, actions=(), payoffs=(1.0, 5.0)),
        }
        res = ExtensiveFormGame("root", nodes).solve_spne()
        assert res.equilibrium_strategies[0]["root"] == "B"
        assert res.payoffs == pytest.approx((1.0, 5.0), abs=APPROX_ABS)


class TestExtensiveFormValidation:
    def test_missing_root(self) -> None:
        with pytest.raises(ValueError):
            ExtensiveFormGame("nope", _entry_game_nodes())

    def test_node_with_two_parents(self) -> None:
        nodes = _entry_game_nodes()
        nodes["root"] = ExtensiveFormNode(
            "root",
            player_idx=0,
            actions=("Enter", "Out"),
            children={"Enter": "n_enter", "Out": "n_acq"},
        )
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)

    def test_root_with_parent_cycle(self) -> None:
        nodes = _entry_game_nodes()
        nodes["n_enter"] = ExtensiveFormNode(
            "n_enter",
            player_idx=1,
            actions=("Fight", "Acquiesce"),
            children={"Fight": "root", "Acquiesce": "n_acq"},
        )
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)

    def test_unreachable_cycle(self) -> None:
        nodes = _entry_game_nodes()
        nodes["x"] = ExtensiveFormNode("x", player_idx=0, actions=("go",), children={"go": "y"})
        nodes["y"] = ExtensiveFormNode("y", player_idx=1, actions=("go",), children={"go": "x"})
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)

    def test_isolated_node(self) -> None:
        nodes = _entry_game_nodes()
        nodes["orphan"] = ExtensiveFormNode(
            "orphan", player_idx=None, actions=(), payoffs=(0.0, 0.0)
        )
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)

    def test_player_idx_out_of_range(self) -> None:
        nodes = _entry_game_nodes()
        nodes["n_enter"] = ExtensiveFormNode(
            "n_enter",
            player_idx=2,
            actions=("Fight", "Acquiesce"),
            children={"Fight": "n_fight", "Acquiesce": "n_acq"},
        )
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes, num_players=2)

    def test_terminal_payoff_length_mismatch(self) -> None:
        nodes = _entry_game_nodes()
        nodes["n_out"] = ExtensiveFormNode("n_out", player_idx=None, actions=(), payoffs=(0.0,))
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)

    def test_actions_children_mismatch(self) -> None:
        nodes = _entry_game_nodes()
        nodes["root"] = ExtensiveFormNode(
            "root",
            player_idx=0,
            actions=("Enter", "Stay"),
            children={"Enter": "n_enter", "Out": "n_out"},
        )
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)

    def test_unknown_child(self) -> None:
        nodes = _entry_game_nodes()
        nodes["root"] = ExtensiveFormNode(
            "root",
            player_idx=0,
            actions=("Enter", "Out"),
            children={"Enter": "n_enter", "Out": "missing"},
        )
        with pytest.raises(ValueError):
            ExtensiveFormGame("root", nodes)
