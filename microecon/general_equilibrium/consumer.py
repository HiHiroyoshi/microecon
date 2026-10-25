"""純粋交換経済における消費者（効用関数 + 初期保有）."""

from __future__ import annotations

import math
from collections.abc import Sequence

import sympy as sp

from microecon.exceptions import InvalidEconomicParameterError
from microecon.general_equilibrium.utility import UtilityProtocol


class BaseConsumer:
    """効用関数と初期保有を統合管理する消費者.

    Args:
        name: 消費者名（経済内で一意であること）。
        utility_function: :class:`UtilityProtocol` を満たす効用関数オブジェクト。
        endowment: 各財の初期保有量 ``(ω_1, ..., ω_N)``。

    Raises:
        ValueError: 効用式に効用変数以外の自由シンボルが含まれる場合、
            財の数と効用変数の数が一致しない場合、または初期保有に負の要素が
            含まれる・全要素が 0 である場合。
    """

    def __init__(
        self,
        name: str,
        utility_function: UtilityProtocol,
        endowment: Sequence[float],
    ) -> None:
        variables = tuple(utility_function.utility_variables)
        free_symbols = utility_function.get_expression().free_symbols
        if not free_symbols.issubset(set(variables)):
            extra = sorted(str(s) for s in free_symbols - set(variables))
            raise ValueError(
                f"Utility expression of '{name}' contains symbols {extra} that are not "
                "utility variables."
            )

        endowment_tuple = tuple(float(e) for e in endowment)
        if len(endowment_tuple) != len(variables):
            raise ValueError(
                f"Endowment of '{name}' has {len(endowment_tuple)} goods but the utility "
                f"function has {len(variables)} variables."
            )
        if any(not (math.isfinite(e) and e >= 0.0) for e in endowment_tuple):
            raise InvalidEconomicParameterError(
                f"Endowment of '{name}' must be finite and non-negative, got {endowment_tuple}."
            )
        if all(e == 0.0 for e in endowment_tuple):
            raise InvalidEconomicParameterError(
                f"Endowment of '{name}' must not be all zeros (income would always be 0)."
            )

        self._name = name
        self._utility_function = utility_function
        self._endowment = endowment_tuple
        self._mrs_cache: dict[tuple[int, int], sp.Expr] = {}

    @property
    def name(self) -> str:
        """消費者名."""
        return self._name

    @property
    def utility_function(self) -> UtilityProtocol:
        """効用関数オブジェクト."""
        return self._utility_function

    @property
    def endowment(self) -> tuple[float, ...]:
        """初期保有量 ``(ω_1, ..., ω_N)``."""
        return self._endowment

    @property
    def n_goods(self) -> int:
        """財の数 ``N``."""
        return len(self._endowment)

    def get_income(self, prices: Sequence[float]) -> float:
        """初期保有の市場価値 ``I = Σ_k p_k ω_k`` を返す."""
        self._validate_prices(prices)
        return float(sum(p * e for p, e in zip(prices, self._endowment)))

    def get_demand(self, prices: tuple[float, ...]) -> tuple[float, ...]:
        """価格ベクトル ``prices`` における需要（初期保有の価値を所得とする）を返す."""
        income = self.get_income(prices)
        return tuple(self._utility_function.get_demand(tuple(prices), income))

    def get_mrs_expression(self, i: int = 0, j: int = 1) -> sp.Expr:
        """限界代替率 ``MRS_{i,j} = (∂U/∂x_i) / (∂U/∂x_j)`` の SymPy 代数式を返す."""
        self._validate_good_index(i)
        self._validate_good_index(j)
        key = (i, j)
        if key not in self._mrs_cache:
            expr = self._utility_function.get_expression()
            variables = self._utility_function.utility_variables
            mrs = sp.diff(expr, variables[i]) / sp.diff(expr, variables[j])
            self._mrs_cache[key] = sp.simplify(mrs)
        return self._mrs_cache[key]

    def get_mrs(self, consumption: Sequence[float], i: int = 0, j: int = 1) -> float:
        """消費ベクトル ``consumption`` における ``MRS_{i,j}`` の数値を返す.

        消費量 0 によるゼロ割などで評価結果が SymPy の ``zoo``（複素無限大）・
        ``oo``・``nan`` になる場合は例外を送出せず ``float('inf')`` を返す。
        """
        if len(consumption) != self.n_goods:
            raise ValueError(
                f"Expected {self.n_goods} consumption values, got {len(consumption)}."
            )
        variables = self._utility_function.utility_variables
        substitution = {var: sp.Float(x) for var, x in zip(variables, consumption)}
        value = self.get_mrs_expression(i, j).subs(substitution, simultaneous=True)
        if value.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            return math.inf
        result = complex(sp.N(value))
        if not math.isfinite(result.real) or abs(result.imag) > 1e-12:
            return math.inf
        return result.real

    def _validate_good_index(self, idx: int) -> None:
        if not 0 <= idx < self.n_goods:
            raise ValueError(f"Good index must satisfy 0 <= idx < {self.n_goods}, got {idx}.")

    def _validate_prices(self, prices: Sequence[float]) -> None:
        if len(prices) != self.n_goods:
            raise InvalidEconomicParameterError(
                f"Expected {self.n_goods} prices, got {len(prices)}."
            )
        if any(not p > 0.0 for p in prices):
            raise InvalidEconomicParameterError(f"All prices must be positive, got {prices}.")
