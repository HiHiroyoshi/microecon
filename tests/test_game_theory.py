"""microecon.game_theory の単体テスト（戦略型ゲーム・ナッシュ均衡分析）."""

from __future__ import annotations

import numpy as np
import pytest

from microecon.game_theory.nash_solver import NashSolver
from microecon.game_theory.normal_form import NormalFormGame

APPROX_ABS = 1e-4


# --------------------------------------------------------------------------
# 1. 囚人のジレンマ（厳密支配戦略・純粋戦略ナッシュ均衡）
# --------------------------------------------------------------------------


class TestPrisonersDilemma:
    """A=[[-1,-3],[0,-2]], B=[[-1,0],[-3,-2]] における
    支配戦略 (Defect, index=1) と純粋戦略ナッシュ均衡 (1,1) の検証."""

    @pytest.fixture
    def game(self) -> NormalFormGame:
        matrix_a = np.array([[-1.0, -3.0], [0.0, -2.0]])
        matrix_b = np.array([[-1.0, 0.0], [-3.0, -2.0]])
        return NormalFormGame(payoff_matrices=(matrix_a, matrix_b))

    def test_has_strict_dominant_strategy(self, game: NormalFormGame) -> None:
        assert game.has_dominant_strategy(0, strict=True) is True

    def test_dominant_strategy_index(self, game: NormalFormGame) -> None:
        assert game.get_dominant_strategy(0, strict=True) == 1

    def test_pure_nash_equilibrium(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        pure_nash = solver.find_pure_nash()
        assert len(pure_nash) == 1
        assert pure_nash[0].strategy_indices == (1, 1)
        assert pure_nash[0].payoffs[0] == pytest.approx(-2.0, abs=APPROX_ABS)
        assert pure_nash[0].payoffs[1] == pytest.approx(-2.0, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 2. 弱支配戦略のテストゲーム
# --------------------------------------------------------------------------


class TestWeakDominance:
    """A=[[2,2],[2,1]], B=[[1,1],[1,1]] における弱支配戦略 (index=0) の検証."""

    @pytest.fixture
    def game(self) -> NormalFormGame:
        matrix_a = np.array([[2.0, 2.0], [2.0, 1.0]])
        matrix_b = np.array([[1.0, 1.0], [1.0, 1.0]])
        return NormalFormGame(payoff_matrices=(matrix_a, matrix_b))

    def test_no_strict_dominant_strategy(self, game: NormalFormGame) -> None:
        assert game.has_dominant_strategy(0, strict=True) is False

    def test_weak_dominant_strategy_exists(self, game: NormalFormGame) -> None:
        assert game.has_dominant_strategy(0, strict=False) is True
        assert game.get_dominant_strategy(0, strict=False) == 0


# --------------------------------------------------------------------------
# 3. コインの表裏合わせ（ゼロサム・混合戦略ナッシュ均衡）
# --------------------------------------------------------------------------


class TestMatchingPennies:
    """A=[[1,-1],[-1,1]], B=[[-1,1],[1,-1]] における
    ゼロサム性・純粋戦略均衡不在・混合戦略均衡 (0.5, 0.5) の検証."""

    @pytest.fixture
    def game(self) -> NormalFormGame:
        matrix_a = np.array([[1.0, -1.0], [-1.0, 1.0]])
        matrix_b = np.array([[-1.0, 1.0], [1.0, -1.0]])
        return NormalFormGame(payoff_matrices=(matrix_a, matrix_b))

    def test_is_zero_sum(self, game: NormalFormGame) -> None:
        assert game.is_zero_sum is True

    def test_no_pure_nash_equilibrium(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        assert len(solver.find_pure_nash()) == 0

    def test_mixed_nash_equilibrium(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        mixed_nash = solver.find_mixed_nash()
        assert len(mixed_nash) == 1
        profile = mixed_nash[0]
        assert profile.probabilities[0][0] == pytest.approx(0.5, abs=APPROX_ABS)
        assert profile.probabilities[1][0] == pytest.approx(0.5, abs=APPROX_ABS)
        assert profile.expected_payoffs[0] == pytest.approx(0.0, abs=APPROX_ABS)
        assert profile.expected_payoffs[1] == pytest.approx(0.0, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 4. 男女の争い（複数の純粋戦略均衡・混合戦略均衡）
# --------------------------------------------------------------------------


class TestBattleOfTheSexes:
    """A=[[2,0],[0,1]], B=[[1,0],[0,2]] における
    純粋戦略均衡2件・混合戦略均衡 (p*=2/3, q*=1/3) の検証."""

    @pytest.fixture
    def game(self) -> NormalFormGame:
        matrix_a = np.array([[2.0, 0.0], [0.0, 1.0]])
        matrix_b = np.array([[1.0, 0.0], [0.0, 2.0]])
        return NormalFormGame(payoff_matrices=(matrix_a, matrix_b))

    def test_pure_nash_equilibria(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        pure_nash = solver.find_pure_nash()
        assert len(pure_nash) == 2
        profiles_by_index = {profile.strategy_indices: profile for profile in pure_nash}
        assert profiles_by_index[(0, 0)].payoffs == pytest.approx(
            (2.0, 1.0), abs=APPROX_ABS
        )
        assert profiles_by_index[(1, 1)].payoffs == pytest.approx(
            (1.0, 2.0), abs=APPROX_ABS
        )

    def test_mixed_nash_equilibrium(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        mixed_nash = solver.find_mixed_nash()
        assert len(mixed_nash) == 1
        profile = mixed_nash[0]
        assert profile.probabilities[0][0] == pytest.approx(2.0 / 3.0, abs=APPROX_ABS)
        assert profile.probabilities[1][0] == pytest.approx(1.0 / 3.0, abs=APPROX_ABS)
        assert profile.expected_payoffs[0] == pytest.approx(2.0 / 3.0, abs=APPROX_ABS)
        assert profile.expected_payoffs[1] == pytest.approx(2.0 / 3.0, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 5. 3x3調整ゲーム（v0.7.0: Lemke-Howson による M x N 混合戦略探索）
# --------------------------------------------------------------------------


class TestThreeByThreeCoordinationGame:
    """対角成分に利得を持つ3x3調整ゲームにおいて、v0.7.0 以降の
    find_mixed_nash() は NotImplementedError を送出せず Lemke-Howson で探索し、
    全初期ラベルが純粋戦略均衡に到達するため mixed_nash_equilibria が
    空タプルとなることの検証（純粋戦略均衡は pure_nash_equilibria にのみ集約）."""

    @pytest.fixture
    def game(self) -> NormalFormGame:
        matrix = np.diag([3.0, 2.0, 1.0])
        return NormalFormGame(payoff_matrices=(matrix, matrix))

    def test_find_mixed_nash_excludes_pure_profiles(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        assert solver.find_mixed_nash() == ()

    def test_solve_mixed_nash_is_empty(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        res = solver.solve()
        assert res.mixed_nash_equilibria == ()

    def test_pure_nash_equilibria_on_diagonal(self, game: NormalFormGame) -> None:
        solver = NashSolver(game)
        res = solver.solve()
        pure_nash = res.pure_nash_equilibria
        assert len(pure_nash) == 3
        profiles_by_index = {profile.strategy_indices: profile for profile in pure_nash}
        assert profiles_by_index[(0, 0)].payoffs == pytest.approx(
            (3.0, 3.0), abs=APPROX_ABS
        )
        assert profiles_by_index[(1, 1)].payoffs == pytest.approx(
            (2.0, 2.0), abs=APPROX_ABS
        )
        assert profiles_by_index[(2, 2)].payoffs == pytest.approx(
            (1.0, 1.0), abs=APPROX_ABS
        )


# --------------------------------------------------------------------------
# 6. 仕様境界・不変性テスト
# --------------------------------------------------------------------------


class TestMixedStrategyProfileImmutability:
    """MixedStrategyProfile.probabilities の各配列が読み取り専用であることの検証."""

    def test_probability_array_is_read_only(self) -> None:
        matrix_a = np.array([[1.0, -1.0], [-1.0, 1.0]])
        matrix_b = np.array([[-1.0, 1.0], [1.0, -1.0]])
        game = NormalFormGame(payoff_matrices=(matrix_a, matrix_b))
        solver = NashSolver(game)
        res = solver.solve()

        with pytest.raises(ValueError, match="read-only"):
            res.mixed_nash_equilibria[0].probabilities[0][0] = 999.0
