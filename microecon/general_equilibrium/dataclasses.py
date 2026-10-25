"""一般均衡理論モジュールの解析結果を保持する不変dataclass群.

内部保持する ``dict`` は :class:`types.MappingProxyType` により読み取り専用化される。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp


@dataclass(frozen=True)
class CompetitiveEquilibriumResult:
    """競争均衡（Arrow-Debreu均衡）の計算結果.

    Attributes:
        prices: 基準財（Numeraire）の価格を 1.0 に正規化した価格ベクトル (p1, p2, ...)。
        allocations: 消費者名 -> 均衡消費ベクトルのマッピング（読み取り専用）。
        utilities: 消費者名 -> 達成効用値のマッピング（読み取り専用）。
        market_clearing_error: 全財の超過需要ベクトルの L2 ノルム。
    """

    prices: tuple[float, ...]
    allocations: Mapping[str, tuple[float, ...]]
    utilities: Mapping[str, float]
    market_clearing_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "prices", tuple(float(p) for p in self.prices))
        read_only_alloc = {
            k: tuple(float(x) for x in v) for k, v in self.allocations.items()
        }
        object.__setattr__(self, "allocations", MappingProxyType(read_only_alloc))
        read_only_util = {k: float(v) for k, v in self.utilities.items()}
        object.__setattr__(self, "utilities", MappingProxyType(read_only_util))


@dataclass(frozen=True)
class EdgeworthBoxResult:
    """2財2消費者純粋交換経済（Edgeworth Box）の解析結果.

    Attributes:
        total_endowment: 総初期保有量 (E1, E2)。
        contract_curve_expression: 契約曲線 ``x2A = f(x1A)`` の右辺を表す SymPy 代数式。
        competitive_equilibrium: 競争均衡の計算結果。
    """

    total_endowment: tuple[float, float]
    contract_curve_expression: sp.Expr
    competitive_equilibrium: CompetitiveEquilibriumResult
