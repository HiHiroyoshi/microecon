"""需要曲線と供給曲線から競争均衡を求めるソルバー."""

from __future__ import annotations

from collections.abc import Callable

import sympy
from scipy.optimize import root_scalar

from microecon.exceptions import NoEquilibriumError
from microecon.market.curves import BaseDemandCurve, BaseSupplyCurve
from microecon.market.dataclasses import EquilibriumResult

_INITIAL_UPPER_BOUND = 1.0
_MAX_BRACKET_EXPANSIONS = 200
_LOWER_BOUND = 1e-9


class MarketEquilibrium:
    """競争均衡 Q_d(P*) = Q_s(P*) を解くソルバー.

    解法選択ロジック: 需要・供給曲線の双方が `has_closed_form == True` の場合は
    両曲線の :meth:`~microecon.market.curves.BaseCurve.get_expression` が返す
    SymPy記号式を連立させて代数的に解く（`is_closed_form=True`）。曲線の型ごとの
    `isinstance` 分岐は持たず、記号計算に委譲することでポリモーフィズムを保つ。
    いずれか一方でも閉形式に対応しない場合のみ、`scipy.optimize.root_scalar`
    による数値解法（`is_closed_form=False`）にフォールバックする。
    """

    def __init__(
        self, demand_curve: BaseDemandCurve, supply_curve: BaseSupplyCurve
    ) -> None:
        self.demand_curve = demand_curve
        self.supply_curve = supply_curve

    def solve(self) -> EquilibriumResult:
        if self.demand_curve.has_closed_form and self.supply_curve.has_closed_form:
            return self._solve_closed_form()
        return self._solve_numeric()

    def _solve_closed_form(self) -> EquilibriumResult:
        demand_expr = self.demand_curve.get_expression()
        supply_expr = self.supply_curve.get_expression()

        try:
            solutions = sympy.solve(sympy.Eq(demand_expr, supply_expr))
        except NotImplementedError:
            return self._solve_numeric()

        positive_real_prices = [
            complex(candidate).real
            for candidate in solutions
            if sympy.im(candidate) == 0 and sympy.re(candidate) > 0
        ]
        if not positive_real_prices:
            raise NoEquilibriumError("No positive market equilibrium exists")

        price_star = min(positive_real_prices)
        quantity_star = self.demand_curve.get_quantity(price_star)
        if quantity_star <= 0:
            raise NoEquilibriumError("No positive market equilibrium exists")

        return EquilibriumResult(
            price=price_star, quantity=quantity_star, is_closed_form=True
        )

    def _solve_numeric(self) -> EquilibriumResult:
        def excess_demand(price: float) -> float:
            return self.demand_curve.get_quantity(
                price
            ) - self.supply_curve.get_quantity(price)

        lower, upper = _find_bracket(excess_demand)
        result = root_scalar(excess_demand, bracket=[lower, upper], method="brentq")
        if not result.converged:
            raise NoEquilibriumError("No positive market equilibrium exists")

        price_star = float(result.root)
        quantity_star = self.demand_curve.get_quantity(price_star)
        if quantity_star <= 0:
            raise NoEquilibriumError("No positive market equilibrium exists")

        return EquilibriumResult(
            price=price_star, quantity=quantity_star, is_closed_form=False
        )


def _find_bracket(
    excess_demand: Callable[[float], float],
    lower: float = _LOWER_BOUND,
) -> tuple[float, float]:
    """excess_demand(price) の符号が反転する [lower, upper] の探索区間を返す.

    価格 0 近傍では需要が供給を上回り（excess_demand > 0）、価格が十分高ければ
    需要が供給を下回る（excess_demand < 0）という典型的な市場の性質を用いて、
    上限を倍加させながら符号反転が起きる区間を探索する。
    """
    if excess_demand(lower) <= 0:
        raise NoEquilibriumError("No positive market equilibrium exists")

    upper = _INITIAL_UPPER_BOUND
    for _ in range(_MAX_BRACKET_EXPANSIONS):
        if excess_demand(upper) < 0:
            return lower, upper
        upper *= 2.0
    raise NoEquilibriumError("No positive market equilibrium exists")
