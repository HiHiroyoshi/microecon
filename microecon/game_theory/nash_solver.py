"""2人戦略型ゲームにおける純粋・混合戦略ナッシュ均衡の探索."""

from __future__ import annotations

import numpy as np

from microecon.game_theory.dataclasses import (
    MixedStrategyProfile,
    NormalFormGameResult,
    PureStrategyProfile,
)
from microecon.game_theory.normal_form import NormalFormGame

_ATOL = 1e-6


class NashSolver:
    """:class:`~microecon.game_theory.normal_form.NormalFormGame` に対する
    ナッシュ均衡ソルバー.

    純粋戦略ナッシュ均衡は最適反応（Best Response）比較により任意の
    :math:`m \\times n` サイズのゲームに対応する。混合戦略ナッシュ均衡は
    v0.6.0 時点では :math:`2 \\times 2` ゲームの解析解（無差別条件）による
    算出のみをサポートする。
    """

    def __init__(self, game: NormalFormGame) -> None:
        self.game = game

    def find_pure_nash(self) -> tuple[PureStrategyProfile, ...]:
        """最適反応比較により純粋戦略ナッシュ均衡を列挙する."""
        matrix_a, matrix_b = self.game.payoff_matrices
        n_rows, n_cols = matrix_a.shape

        is_best_response_row = np.zeros((n_rows, n_cols), dtype=bool)
        for j in range(n_cols):
            column = matrix_a[:, j]
            is_best_response_row[:, j] = column >= (column.max() - _ATOL)

        is_best_response_col = np.zeros((n_rows, n_cols), dtype=bool)
        for i in range(n_rows):
            row = matrix_b[i, :]
            is_best_response_col[i, :] = row >= (row.max() - _ATOL)

        equilibria = []
        for i in range(n_rows):
            for j in range(n_cols):
                if is_best_response_row[i, j] and is_best_response_col[i, j]:
                    equilibria.append(
                        PureStrategyProfile(
                            strategy_indices=(i, j),
                            payoffs=(float(matrix_a[i, j]), float(matrix_b[i, j])),
                        )
                    )
        return tuple(equilibria)

    def find_mixed_nash(self) -> tuple[MixedStrategyProfile, ...]:
        """混合戦略ナッシュ均衡を算出する（v0.6.0時点では2x2ゲームのみ対応）."""
        if self.game.n_strategies != (2, 2):
            raise NotImplementedError(
                "v0.6.0 only supports mixed strategy Nash equilibrium for 2x2 games. "
                "Arbitrary N x M games will be supported in v0.7.0 via Lemke-Howson "
                "algorithm."
            )
        return self._solve_2x2_mixed_nash()

    def _solve_2x2_mixed_nash(self) -> tuple[MixedStrategyProfile, ...]:
        """無差別条件による2x2ゲームの解析解を返す.

        プレイヤー1の戦略0を選ぶ確率を ``p``、プレイヤー2の戦略0を選ぶ確率を
        ``q`` とすると、相手を無差別にする ``p*``, ``q*`` は次式で与えられる。

        ``q* = (a22 - a12) / (a11 - a12 - a21 + a22)``
        ``p* = (b22 - b21) / (b11 - b12 - b21 + b22)``
        """
        matrix_a, matrix_b = self.game.payoff_matrices
        a11, a12 = matrix_a[0, 0], matrix_a[0, 1]
        a21, a22 = matrix_a[1, 0], matrix_a[1, 1]
        b11, b12 = matrix_b[0, 0], matrix_b[0, 1]
        b21, b22 = matrix_b[1, 0], matrix_b[1, 1]

        denom_p = b11 - b12 - b21 + b22
        denom_q = a11 - a12 - a21 + a22
        if abs(denom_p) <= _ATOL or abs(denom_q) <= _ATOL:
            return ()

        p_star = (b22 - b21) / denom_p
        q_star = (a22 - a12) / denom_q
        if not (-_ATOL <= p_star <= 1 + _ATOL) or not (-_ATOL <= q_star <= 1 + _ATOL):
            return ()

        p_star = float(np.clip(p_star, 0.0, 1.0))
        q_star = float(np.clip(q_star, 0.0, 1.0))

        p_vec = np.array([p_star, 1.0 - p_star])
        q_vec = np.array([q_star, 1.0 - q_star])

        profile = MixedStrategyProfile(
            probabilities=(p_vec, q_vec),
            expected_payoffs=(
                self.game.get_expected_payoff(0, (p_vec, q_vec)),
                self.game.get_expected_payoff(1, (p_vec, q_vec)),
            ),
        )
        return (profile,)

    def solve(self) -> NormalFormGameResult:
        """純粋・混合戦略ナッシュ均衡をまとめて算出する.

        混合戦略均衡が :math:`2 \\times 2` 以外のゲームサイズに起因して
        ``NotImplementedError`` を送出した場合は、これを内部で捕捉し
        ``mixed_nash_equilibria = ()`` としてフォールバックする。
        """
        pure_nash_equilibria = self.find_pure_nash()
        try:
            mixed_nash_equilibria = self.find_mixed_nash()
        except NotImplementedError:
            mixed_nash_equilibria = ()

        return NormalFormGameResult(
            pure_nash_equilibria=pure_nash_equilibria,
            mixed_nash_equilibria=mixed_nash_equilibria,
            is_zero_sum=self.game.is_zero_sum,
        )
