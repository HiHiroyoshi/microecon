"""Lemke-Howson アルゴリズムによる双行列ゲームのナッシュ均衡探索.

利得行列 ``(A, B)`` (``M x N``) の双行列ゲームについて、2つの最適反応多面体

``P = {x >= 0 : B^T x <= 1}``,  ``Q = {y >= 0 : A y <= 1}``

上の相補的ピボット操作（線形相補性問題 LCP の Tableau 法）により、1つの
ナッシュ均衡に対応する頂点ペア ``(x, y)`` を求める。ラベルは
``0, ..., M-1`` がプレイヤー1の戦略、``M, ..., M+N-1`` がプレイヤー2の戦略に
対応する。

Notes:
    Lemke-Howson アルゴリズムが保証するのは「少なくとも1つのナッシュ均衡」の
    発見であり、一般の ``M x N`` ゲームにおける全ナッシュ均衡の完全列挙は
    保証しない。初期ラベルの軌跡によっては到達しない均衡（例: 男女の争いにおける
    指数 -1 の完全混合均衡）が存在し得る。
"""

from __future__ import annotations

import numpy as np

_SUPPORT_TOL = 1e-6
_BEST_RESPONSE_TOL = 1e-4
_PIVOT_TOL = 1e-12


class LemkeHowsonSolver:
    """任意サイズ ``M x N`` の双行列ゲームに対する Lemke-Howson ソルバー.

    退化ゲームにおける無限巡回（cycling）を防止するため、Tableau の行選択には
    辞書式最小比テスト (Lexicographic Minimum Ratio Test) を適用する。
    """

    def __init__(self, payoff_matrix_a: np.ndarray, payoff_matrix_b: np.ndarray) -> None:
        matrix_a = np.asarray(payoff_matrix_a, dtype=float)
        matrix_b = np.asarray(payoff_matrix_b, dtype=float)
        if matrix_a.ndim != 2 or matrix_b.ndim != 2:
            raise ValueError("Payoff matrices must be 2-dimensional arrays.")
        if matrix_a.shape != matrix_b.shape:
            raise ValueError(
                "Payoff matrices for both players must share the same shape, "
                f"got {matrix_a.shape} and {matrix_b.shape}."
            )
        if matrix_a.size == 0:
            raise ValueError("Payoff matrices must be non-empty.")
        self.payoff_matrix_a = matrix_a
        self.payoff_matrix_b = matrix_b

    @property
    def n_labels(self) -> int:
        """ラベル総数 ``M + N`` を返す."""
        n_rows, n_cols = self.payoff_matrix_a.shape
        return int(n_rows + n_cols)

    def solve(self, initial_dropped_label: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """初期ラベル ``initial_dropped_label`` から出発してナッシュ均衡 ``(p*, q*)`` を求める.

        Args:
            initial_dropped_label: 最初に落とす（Tableau に導入する）ラベル。
                ``0 <= label < M + N`` でなければならない。

        Returns:
            プレイヤー1・プレイヤー2の確率分布ベクトル ``(p*, q*)``。

        Raises:
            ValueError: ``initial_dropped_label`` が範囲外の場合。
            RuntimeError: ピボット操作が破綻した場合、または得られた解が
                最適反応条件の事後検証を満たさない場合。
        """
        n_rows, n_cols = self.payoff_matrix_a.shape
        if not 0 <= initial_dropped_label < n_rows + n_cols:
            raise ValueError(
                f"initial_dropped_label must satisfy 0 <= label < {n_rows + n_cols}, "
                f"got {initial_dropped_label}."
            )

        scaled_a, scaled_b = self._positive_scaled_matrices()

        # 列 0..M-1 は x (ラベル 0..M-1)、列 M..M+N-1 はスラック (ラベル M..M+N-1)。
        row_tableau = np.hstack([scaled_b.T, np.eye(n_cols), np.ones((n_cols, 1))])
        row_basis = list(range(n_rows, n_rows + n_cols))
        row_identity_cols = list(range(n_rows, n_rows + n_cols))

        # 列 0..M-1 はスラック (ラベル 0..M-1)、列 M..M+N-1 は y (ラベル M..M+N-1)。
        col_tableau = np.hstack([np.eye(n_rows), scaled_a, np.ones((n_rows, 1))])
        col_basis = list(range(n_rows))
        col_identity_cols = list(range(n_rows))

        # 落としたラベルは x 側 (ラベル < M) なら row_tableau、y 側なら col_tableau に
        # 導入する。以降は退出したラベルを他方の Tableau に導入する操作を交互に繰り返す。
        entering = initial_dropped_label
        use_row_tableau = initial_dropped_label < n_rows
        max_iterations = 10_000 + 100 * (n_rows + n_cols) ** 2
        for _ in range(max_iterations):
            if use_row_tableau:
                leaving = self._pivot(row_tableau, row_basis, row_identity_cols, entering)
            else:
                leaving = self._pivot(col_tableau, col_basis, col_identity_cols, entering)
            if leaving == initial_dropped_label:
                break
            entering = leaving
            use_row_tableau = not use_row_tableau
        else:
            raise RuntimeError("Lemke-Howson pivoting did not terminate.")

        x = self._extract_values(row_tableau, row_basis, range(n_rows))
        y = self._extract_values(col_tableau, col_basis, range(n_rows, n_rows + n_cols))
        if x.sum() <= _PIVOT_TOL or y.sum() <= _PIVOT_TOL:
            raise RuntimeError("Lemke-Howson pivoting reached the artificial equilibrium.")

        p = x / x.sum()
        q = y / y.sum()
        self._verify_best_responses(p, q)
        return p, q

    def expected_payoffs(self, p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        """スケーリング前の元の利得行列で期待利得 ``(p^T A q, p^T B q)`` を返す."""
        return (
            float(p @ self.payoff_matrix_a @ q),
            float(p @ self.payoff_matrix_b @ q),
        )

    def _positive_scaled_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """全要素を正にする定数 ``k`` を加えた利得行列を返す（内部計算専用）."""
        min_value = float(min(self.payoff_matrix_a.min(), self.payoff_matrix_b.min()))
        k = abs(min_value) + 1.0 if min_value <= 0.0 else 0.0
        return self.payoff_matrix_a + k, self.payoff_matrix_b + k

    @staticmethod
    def _pivot(
        tableau: np.ndarray,
        basis: list[int],
        identity_cols: list[int],
        entering: int,
    ) -> int:
        """ラベル ``entering`` の列で Tableau をピボットし、退出したラベルを返す.

        行選択は辞書式最小比テストにより一意に決定する。最小比率
        ``rhs_r / coeff_r`` が同値の行が複数ある場合、初期単位行列に対応する列の
        係数比 ``tableau[r, c] / coeff_r`` を左の列から順に辞書式比較する。
        単位行列部分の各行は線形独立であるため、最小行は必ず一意に定まる。
        """
        column = tableau[:, entering]
        candidates = [r for r in range(tableau.shape[0]) if column[r] > _PIVOT_TOL]
        if not candidates:
            raise RuntimeError(
                f"No valid pivot row for entering label {entering} (unbounded ray)."
            )

        def lex_key(r: int) -> tuple[float, ...]:
            ratios = [tableau[r, -1] / column[r]]
            ratios.extend(tableau[r, c] / column[r] for c in identity_cols)
            return tuple(ratios)

        pivot_row = candidates[0]
        best_key = lex_key(pivot_row)
        for r in candidates[1:]:
            key = lex_key(r)
            if _lex_less(key, best_key):
                pivot_row, best_key = r, key

        tableau[pivot_row, :] /= tableau[pivot_row, entering]
        for r in range(tableau.shape[0]):
            if r != pivot_row and tableau[r, entering] != 0.0:
                tableau[r, :] -= tableau[r, entering] * tableau[pivot_row, :]

        leaving = basis[pivot_row]
        basis[pivot_row] = entering
        return leaving

    @staticmethod
    def _extract_values(
        tableau: np.ndarray, basis: list[int], labels: range
    ) -> np.ndarray:
        """基底変数の値（右辺列）から ``labels`` に対応する変数ベクトルを取り出す."""
        values = np.zeros(len(labels))
        for row, label in enumerate(basis):
            if label in labels:
                values[label - labels.start] = max(float(tableau[row, -1]), 0.0)
        return values

    def _verify_best_responses(self, p: np.ndarray, q: np.ndarray) -> None:
        """支持集合上の戦略が最適反応になっていることを事後検証する."""
        row_payoffs = self.payoff_matrix_a @ q
        col_payoffs = p @ self.payoff_matrix_b
        row_ok = np.all(
            row_payoffs[p > _SUPPORT_TOL] >= row_payoffs.max() - _BEST_RESPONSE_TOL
        )
        col_ok = np.all(
            col_payoffs[q > _SUPPORT_TOL] >= col_payoffs.max() - _BEST_RESPONSE_TOL
        )
        if not (row_ok and col_ok):
            raise RuntimeError(
                "Lemke-Howson result violates the best-response condition: "
                f"p={p}, q={q}."
            )


def _lex_less(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> bool:
    """浮動小数点誤差を許容した辞書式比較 ``lhs < rhs``."""
    for a, b in zip(lhs, rhs):
        if a < b - _PIVOT_TOL:
            return True
        if a > b + _PIVOT_TOL:
            return False
    return False
