"""2人戦略型ゲーム（Normal Form Game）を表現するファサードクラス."""

from __future__ import annotations

import numpy as np

_ATOL = 1e-6


class NormalFormGame:
    """2人有限戦略型ゲーム.

    利得行列のペア ``(A, B)`` によってゲームを定義する。``A[i, j]`` はプレイヤー1が
    戦略 ``i``、プレイヤー2が戦略 ``j`` を選んだ場合のプレイヤー1の利得を、
    ``B[i, j]`` は同じ戦略プロファイルにおけるプレイヤー2の利得を表す。
    """

    def __init__(
        self,
        payoff_matrices: tuple[np.ndarray, np.ndarray],
        player_names: tuple[str, str] = ("Player 1", "Player 2"),
        strategy_names: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
    ) -> None:
        matrix_a = np.asarray(payoff_matrices[0], dtype=float)
        matrix_b = np.asarray(payoff_matrices[1], dtype=float)
        if matrix_a.ndim != 2 or matrix_b.ndim != 2:
            raise ValueError("Payoff matrices must be 2-dimensional arrays.")
        if matrix_a.shape != matrix_b.shape:
            raise ValueError(
                "Payoff matrices for both players must share the same shape, "
                f"got {matrix_a.shape} and {matrix_b.shape}."
            )
        self.payoff_matrices: tuple[np.ndarray, np.ndarray] = (matrix_a, matrix_b)
        self.player_names = player_names
        self.strategy_names = strategy_names

    @property
    def n_strategies(self) -> tuple[int, int]:
        """(プレイヤー1の戦略数, プレイヤー2の戦略数) を返す."""
        n_rows, n_cols = self.payoff_matrices[0].shape
        return (int(n_rows), int(n_cols))

    def get_payoff(self, player_idx: int, strategy_profile: tuple[int, int]) -> float:
        """戦略プロファイル ``strategy_profile`` におけるプレイヤー ``player_idx`` の利得を返す."""
        i, j = strategy_profile
        return float(self.payoff_matrices[player_idx][i, j])

    def get_expected_payoff(
        self, player_idx: int, mixed_strategies: tuple[np.ndarray, np.ndarray]
    ) -> float:
        """混合戦略の組 ``(p, q)`` における期待利得 ``p^T A q`` を返す."""
        p, q = mixed_strategies
        matrix = self.payoff_matrices[player_idx]
        return float(np.asarray(p, dtype=float) @ matrix @ np.asarray(q, dtype=float))

    def get_dominant_strategy(self, player_idx: int, strict: bool = True) -> int | None:
        """プレイヤー ``player_idx`` の支配戦略のインデックスを返す。存在しなければ ``None``."""
        n_own = self.n_strategies[player_idx]
        for candidate in range(n_own):
            if self._dominates_all_others(player_idx, candidate, strict):
                return candidate
        return None

    def has_dominant_strategy(self, player_idx: int, strict: bool = True) -> bool:
        """プレイヤー ``player_idx`` に支配戦略が存在するか否かを返す."""
        return self.get_dominant_strategy(player_idx, strict) is not None

    def _dominates_all_others(self, player_idx: int, candidate: int, strict: bool) -> bool:
        payoff_matrix = self._own_payoff_matrix(player_idx)
        n_own = payoff_matrix.shape[0]
        for other in range(n_own):
            if other == candidate:
                continue
            diff = payoff_matrix[candidate, :] - payoff_matrix[other, :]
            if strict:
                if not np.all(diff > _ATOL):
                    return False
            else:
                if not np.all(diff >= -_ATOL):
                    return False
                if not np.any(diff > _ATOL):
                    return False
        return True

    def _own_payoff_matrix(self, player_idx: int) -> np.ndarray:
        """プレイヤー ``player_idx`` を「行プレイヤー」（行=自身の戦略、列=相手の戦略）
        とみなした利得行列を返す。
        """
        matrix = self.payoff_matrices[player_idx]
        return matrix if player_idx == 0 else matrix.T

    @property
    def is_zero_sum(self) -> bool:
        """``A + B = 0`` (``atol=1e-6``) を満たすゼロサムゲームか否かを返す."""
        matrix_a, matrix_b = self.payoff_matrices
        return bool(np.allclose(matrix_a + matrix_b, 0.0, atol=_ATOL))
