"""2人戦略型ゲームにおける純粋・混合戦略ナッシュ均衡の探索."""

from __future__ import annotations

import numpy as np

from microecon.game_theory.dataclasses import (
    MixedStrategyProfile,
    NormalFormGameResult,
    PureStrategyProfile,
)
from microecon.game_theory.lemke_howson import LemkeHowsonSolver
from microecon.game_theory.normal_form import NormalFormGame

_ATOL = 1e-6
_SUPPORT_TOL = 1e-6
_DEDUP_TOL = 1e-4


class NashSolver:
    """:class:`~microecon.game_theory.normal_form.NormalFormGame` に対する
    ナッシュ均衡ソルバー.

    純粋戦略ナッシュ均衡は最適反応（Best Response）比較により任意の
    :math:`m \\times n` サイズのゲームに対応する。

    混合戦略ナッシュ均衡（純粋戦略ではない、すなわち少なくとも1人のプレイヤーが
    複数の戦略に正の確率を置くプロファイル）は次のように算出する。

    - :math:`2 \\times 2` ゲーム: 無差別条件による解析解で完全混合戦略
      (:math:`0 < p, q < 1`) を算出する。
    - それ以外の :math:`M \\times N` ゲーム: 全初期ラベル
      (:math:`0, \\dots, M+N-1`) で
      :class:`~microecon.game_theory.lemke_howson.LemkeHowsonSolver` を実行し、
      純粋戦略を除外した上で重複しない（許容誤差 ``1e-4``）解のみを返す。

    Notes:
        Lemke-Howson アルゴリズムは「少なくとも1つのナッシュ均衡」の発見を
        保証するものであり、一般の :math:`M \\times N` ゲームにおける全ナッシュ
        均衡の完全列挙は保証しない。ラベルの軌跡によっては到達しない均衡が
        存在し得る。また退化ゲーム等で均衡が連続体（線分等）となる場合は、
        Lemke-Howson が到達した連続体の代表点（頂点解）のみを返却する。
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
        """純粋戦略ではない混合戦略ナッシュ均衡を算出する.

        :math:`2 \\times 2` ゲームでは解析解、それ以外では Lemke-Howson
        アルゴリズムの全初期ラベル探索を用いる（網羅性の限界はクラスの Notes を参照）。
        """
        if self.game.n_strategies == (2, 2):
            return self._solve_2x2_mixed_nash()
        return self._solve_lemke_howson_mixed_nash()

    def _solve_lemke_howson_mixed_nash(self) -> tuple[MixedStrategyProfile, ...]:
        """全初期ラベルで Lemke-Howson を実行し、非純粋かつ重複しない解を返す."""
        matrix_a, matrix_b = self.game.payoff_matrices
        lh_solver = LemkeHowsonSolver(matrix_a, matrix_b)

        profiles: list[MixedStrategyProfile] = []
        for label in range(lh_solver.n_labels):
            p, q = lh_solver.solve(initial_dropped_label=label)
            if _is_pure(p) and _is_pure(q):
                continue
            if any(
                np.allclose(p, existing.probabilities[0], atol=_DEDUP_TOL)
                and np.allclose(q, existing.probabilities[1], atol=_DEDUP_TOL)
                for existing in profiles
            ):
                continue
            profiles.append(
                MixedStrategyProfile(
                    probabilities=(p, q),
                    expected_payoffs=lh_solver.expected_payoffs(p, q),
                )
            )
        return tuple(profiles)

    def _solve_2x2_mixed_nash(self) -> tuple[MixedStrategyProfile, ...]:
        """無差別条件による2x2ゲームの解析解を返す.

        プレイヤー1の戦略0を選ぶ確率を ``p``、プレイヤー2の戦略0を選ぶ確率を
        ``q`` とすると、相手を無差別にする ``p*``, ``q*`` は次式で与えられる。

        ``q* = (a22 - a12) / (a11 - a12 - a21 + a22)``
        ``p* = (b22 - b21) / (b11 - b12 - b21 + b22)``

        完全混合戦略 (``0 < p*, q* < 1``) のみを返し、境界解は返さない。
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

        p_star = float((b22 - b21) / denom_p)
        q_star = float((a22 - a12) / denom_q)
        if not (_ATOL < p_star < 1 - _ATOL) or not (_ATOL < q_star < 1 - _ATOL):
            return ()

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
        """純粋・混合戦略ナッシュ均衡をまとめて算出する."""
        return NormalFormGameResult(
            pure_nash_equilibria=self.find_pure_nash(),
            mixed_nash_equilibria=self.find_mixed_nash(),
            is_zero_sum=self.game.is_zero_sum,
        )


def _is_pure(probabilities: np.ndarray) -> bool:
    """正の確率（``> 1e-6``）を持つ戦略がちょうど1つであれば純粋戦略とみなす."""
    return int(np.count_nonzero(probabilities > _SUPPORT_TOL)) == 1
