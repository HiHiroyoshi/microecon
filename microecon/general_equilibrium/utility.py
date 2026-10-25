"""一般均衡分析用の N 財効用関数インターフェースと具象クラス.

既存の :class:`microecon.consumer.base.BaseUtilityFunction` は2財専用の
シグネチャ（``evaluate(x, y)`` など）を持つため継承せず、
:class:`typing.Protocol` による構造的型付けで N 財効用関数の規約を定める。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import sympy as sp

from microecon.exceptions import InvalidEconomicParameterError


class UtilityProtocol(Protocol):
    """一般均衡分析で用いる N 財効用関数のインターフェース."""

    @property
    def utility_variables(self) -> tuple[sp.Symbol, ...]:
        """各財の消費量を表す SymPy シンボルのタプル."""
        ...

    def get_expression(self) -> sp.Expr:
        """効用関数の SymPy 代数式を返す."""
        ...

    def evaluate(self, *consumption: float) -> float:
        """消費ベクトルに対する効用値を返す."""
        ...

    def get_demand(self, prices: tuple[float, ...], income: float) -> tuple[float, ...]:
        """価格ベクトルと所得に対するマーシャル需要を返す."""
        ...


class NGoodCobbDouglasUtility:
    """N 財コブ＝ダグラス型効用関数 ``U(x) = Π_k x_k^{α_k}``.

    指数ウェイトは ``sp.Rational(str(alpha))`` により有理数へ変換して式を構成し、
    浮動小数点数による ``sp.solve`` の不安定化や Float 係数の残留を防ぐ。

    Args:
        weights: 各財のウェイト ``(α_1, ..., α_N)``。全要素が正でなければならない。
        utility_variables: 各財の消費量シンボル。``None`` の場合は
            ``x1, ..., xN`` (``positive=True``) を自動生成する。

    Raises:
        InvalidEconomicParameterError: ウェイトが正でない場合、または
            ``utility_variables`` の長さが財の数と一致しない場合
            （いずれも ``ValueError`` のサブクラス）。
    """

    def __init__(
        self,
        weights: Sequence[float],
        utility_variables: Sequence[sp.Symbol] | None = None,
    ) -> None:
        weights_tuple = tuple(float(w) for w in weights)
        if not weights_tuple:
            raise InvalidEconomicParameterError("weights must contain at least one good.")
        for w in weights_tuple:
            if not (math.isfinite(w) and w > 0.0):
                raise InvalidEconomicParameterError(
                    f"All weights must be positive and finite, got {weights_tuple}."
                )

        n_goods = len(weights_tuple)
        if utility_variables is None:
            variables = tuple(
                sp.Symbol(f"x{k + 1}", positive=True) for k in range(n_goods)
            )
        else:
            variables = tuple(utility_variables)
            if len(variables) != n_goods:
                raise InvalidEconomicParameterError(
                    f"utility_variables must have length {n_goods} (the number of goods), "
                    f"got {len(variables)}."
                )
            if len(set(variables)) != n_goods:
                raise InvalidEconomicParameterError("utility_variables must be distinct.")

        self._weights = weights_tuple
        self._exponents = tuple(sp.Rational(str(w)) for w in weights_tuple)
        self._utility_variables: tuple[sp.Symbol, ...] = variables

    @property
    def weights(self) -> tuple[float, ...]:
        """各財のウェイト ``(α_1, ..., α_N)``."""
        return self._weights

    @property
    def n_goods(self) -> int:
        """財の数 ``N``."""
        return len(self._weights)

    @property
    def utility_variables(self) -> tuple[sp.Symbol, ...]:
        """各財の消費量を表す SymPy シンボルのタプル（読み取り専用）."""
        return self._utility_variables

    def get_expression(self) -> sp.Expr:
        """有理数指数による代数式 ``Π_k x_k^{α_k}`` を返す."""
        return sp.Mul(
            *(var**exp for var, exp in zip(self._utility_variables, self._exponents))
        )

    def evaluate(self, *consumption: float) -> float:
        """消費ベクトル ``consumption`` に対する効用値を返す."""
        if len(consumption) != self.n_goods:
            raise InvalidEconomicParameterError(
                f"Expected {self.n_goods} consumption values, got {len(consumption)}."
            )
        if any(x < 0.0 for x in consumption):
            raise InvalidEconomicParameterError(
                f"Consumption must be non-negative, got {consumption}."
            )
        return float(math.prod(x**w for x, w in zip(consumption, self._weights)))

    def get_demand(self, prices: tuple[float, ...], income: float) -> tuple[float, ...]:
        """解析的マーシャル需要 ``x_k* = (α_k / Σ_j α_j) * I / p_k`` を返す.

        Raises:
            InvalidEconomicParameterError: 価格ベクトルの長さが財の数と一致しない場合、
                価格が正でない場合、または所得が正でない場合。
        """
        if len(prices) != self.n_goods:
            raise InvalidEconomicParameterError(
                f"Expected {self.n_goods} prices, got {len(prices)}."
            )
        if any(not p > 0.0 for p in prices):
            raise InvalidEconomicParameterError(f"All prices must be positive, got {prices}.")
        if not income > 0.0:
            raise InvalidEconomicParameterError(f"Income must be positive, got {income}.")

        total_weight = sum(self._weights)
        return tuple(w / total_weight * income / p for w, p in zip(self._weights, prices))
