"""ゲーム理論モジュールの解析結果・構造を保持する不変dataclass群.

``dict`` などの可変オブジェクトは :class:`types.MappingProxyType` により、
``np.ndarray`` は ``flags.writeable = False`` により読み取り専用化される。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _make_readonly_arr(arr: np.ndarray) -> np.ndarray:
    """配列を ``float`` 型で複製し、要素の書き換えを不可能 (read-only) にする."""
    readonly_arr = np.array(arr, dtype=float, copy=True)
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


@dataclass(frozen=True, eq=False)
class MixedStrategyProfile:
    """混合戦略プロファイル.

    NumPy配列同士の ``==`` は要素ごとの真偽値配列を返し、dataclass自動生成の
    ``__eq__`` では ``ValueError: truth value of an array is ambiguous`` を
    引き起こすため、``eq=False`` とした上で :func:`numpy.array_equal` による
    ``__eq__`` を明示的に実装する。

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
        readonly_probs = tuple(_make_readonly_arr(p) for p in self.probabilities)
        object.__setattr__(self, "probabilities", readonly_probs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixedStrategyProfile):
            return False
        if len(self.probabilities) != len(other.probabilities):
            return False
        for p1, p2 in zip(self.probabilities, other.probabilities):
            if not np.array_equal(p1, p2):
                return False
        return self.expected_payoffs == other.expected_payoffs


@dataclass(frozen=True)
class NormalFormGameResult:
    """戦略型ゲームのナッシュ均衡解析結果.

    Attributes:
        pure_nash_equilibria: 純粋戦略ナッシュ均衡の一覧。
        mixed_nash_equilibria: 純粋戦略ではない（少なくとも1人のプレイヤーが
            複数の戦略に正の確率を置く）混合戦略ナッシュ均衡の一覧。
            純粋戦略ナッシュ均衡は ``pure_nash_equilibria`` にのみ格納される。
        is_zero_sum: ゼロサムゲームか否か。
    """

    pure_nash_equilibria: tuple[PureStrategyProfile, ...]
    mixed_nash_equilibria: tuple[MixedStrategyProfile, ...]
    is_zero_sum: bool


@dataclass(frozen=True)
class ExtensiveFormNode:
    """展開型ゲーム（ゲーム木）のノード構造.

    Attributes:
        node_id: ノードの一意な識別子。
        player_idx: 当該ノードで行動を選択するプレイヤーのインデックス。
            ``None`` の場合は終端ノード (Terminal Node)。
        actions: 当該ノードで選択可能な行動リスト（終端ノードでは空タプル）。
        children: 行動名 -> 子ノードID のマッピング（読み取り専用）。
        payoffs: 終端ノードにおける各プレイヤーの利得。
    """

    node_id: str
    player_idx: int | None
    actions: tuple[str, ...]
    children: Mapping[str, str] | None = None
    payoffs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        if self.children is not None:
            proxy = MappingProxyType(dict(self.children))
            object.__setattr__(self, "children", proxy)
        if self.payoffs is not None:
            object.__setattr__(self, "payoffs", tuple(float(v) for v in self.payoffs))

    @property
    def is_terminal(self) -> bool:
        """終端ノードか否かを返す."""
        return self.player_idx is None


@dataclass(frozen=True)
class SubgamePerfectResult:
    """部分ゲーム完全ナッシュ均衡 (SPNE) の結果.

    Attributes:
        equilibrium_strategies: プレイヤーインデックス -> (ノードID -> 選択行動)
            の二重マッピング（各階層とも読み取り専用）。均衡経路外の
            意思決定ノードを含む完全な戦略を保持する。
        payoffs: 均衡経路上で実現する各プレイヤーの利得。
    """

    equilibrium_strategies: Mapping[int, Mapping[str, str]]
    payoffs: tuple[float, ...]

    def __post_init__(self) -> None:
        read_only_strat = {
            p: MappingProxyType(dict(st)) for p, st in self.equilibrium_strategies.items()
        }
        proxy = MappingProxyType(read_only_strat)
        object.__setattr__(self, "equilibrium_strategies", proxy)


@dataclass(frozen=True)
class ContinuousGameResult:
    """連続戦略ゲームの計算結果.

    Attributes:
        strategies: 均衡における各プレイヤーの戦略 (s_1*, s_2*)。
        payoffs: 均衡における各プレイヤーの利得 (π_1*, π_2*)。
        is_corner: いずれかのプレイヤーの戦略が戦略空間の境界（角点）に
            位置する場合 ``True``。
    """

    strategies: tuple[float, ...]
    payoffs: tuple[float, ...]
    is_corner: bool
