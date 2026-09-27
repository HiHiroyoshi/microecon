"""戦略型ゲーム（Normal Form Game）の解析結果を保持する不変dataclass群."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    """配列を複製し、要素の書き換えを不可能 (read-only) にする."""
    readonly_arr = arr.copy()
    readonly_arr.flags.writeable = False
    return readonly_arr


@dataclass(frozen=True)
class PureStrategyProfile:
    """純粋戦略プロファイル.

    Attributes:
        strategy_indices: 各プレイヤーの選択した戦略インデックス
            (プレイヤー1の戦略インデックス, プレイヤー2の戦略インデックス)。
        payoffs: 各プレイヤーの利得 (プレイヤー1の利得, プレイヤー2の利得)。
    """

    strategy_indices: tuple[int, ...]
    payoffs: tuple[float, ...]


@dataclass(frozen=True)
class MixedStrategyProfile:
    """混合戦略プロファイル.

    Attributes:
        probabilities: 各プレイヤーの確率分布ベクトル
            (プレイヤー1の確率分布ベクトル, プレイヤー2の確率分布ベクトル)。
            保持されるNumPy配列は複製された上で書き換え不可 (read-only) 化される。
        expected_payoffs: 各プレイヤーの期待利得
            (プレイヤー1の期待利得, プレイヤー2の期待利得)。
    """

    probabilities: tuple[np.ndarray, ...]
    expected_payoffs: tuple[float, ...]

    def __post_init__(self) -> None:
        readonly_probs = tuple(_make_readonly(p) for p in self.probabilities)
        object.__setattr__(self, "probabilities", readonly_probs)


@dataclass(frozen=True)
class NormalFormGameResult:
    """戦略型ゲームのナッシュ均衡解析結果.

    Attributes:
        pure_nash_equilibria: 純粋戦略ナッシュ均衡の一覧。
        mixed_nash_equilibria: 混合戦略ナッシュ均衡の一覧
            (2x2以外のゲームでは空タプル)。
        is_zero_sum: ゼロサムゲームか否か。
    """

    pure_nash_equilibria: tuple[PureStrategyProfile, ...]
    mixed_nash_equilibria: tuple[MixedStrategyProfile, ...]
    is_zero_sum: bool
