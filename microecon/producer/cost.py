"""費用構造の分析モジュール（長期費用指標・短期費用分解）.

長期の費用指標（限界費用・平均費用）は総費用関数 C(q, w, r) のSymPy記号微分から
導出する（:class:`CostAnalyzer`）。短期の費用分解は資本量を固定した下での
可変費用・固定費用への分解を行う（:class:`ShortRunProduction`）。
"""

from __future__ import annotations

import sympy

from microecon.producer.base import BaseProductionFunction


class CostAnalyzer:
    """総費用関数 C(q, w, r) から長期費用指標を導出するアナライザー.

    要素価格 (w, r) を固定した下で、目標産出量 q の関数としての
    限界費用 MC(q) と平均費用 AC(q) を計算する。
    """

    def __init__(
        self, production: BaseProductionFunction, wage: float, rental: float
    ) -> None:
        if wage <= 0:
            raise ValueError(f"Wage w must be strictly positive (w > 0), got {wage}")
        if rental <= 0:
            raise ValueError(
                f"Rental price r must be strictly positive (r > 0), got {rental}"
            )
        self.production = production
        self.wage = wage
        self.rental = rental

    def marginal_cost(self, target_output: float) -> float:
        """限界費用 MC(q) = dC(q, w, r) / dq を返す（総費用関数のSymPy記号微分）."""
        if target_output <= 0:
            raise ValueError(
                f"Target output q must be strictly positive (q > 0), got {target_output}"
            )
        q, w, r = sympy.symbols("q w r", positive=True)
        cost_expr = self.production.get_cost_expression()
        marginal_expr = sympy.diff(cost_expr, q)
        value = marginal_expr.subs({q: target_output, w: self.wage, r: self.rental})
        return float(value.evalf())

    def average_cost(self, target_output: float) -> float:
        """平均費用 AC(q) = C(q, w, r) / q を返す."""
        if target_output <= 0:
            raise ValueError(
                f"Target output q must be strictly positive (q > 0), got {target_output}"
            )
        total_cost = self.production.calculate_cost_function(
            target_output, self.wage, self.rental
        )
        return float(total_cost / target_output)


class ShortRunProduction:
    """資本量を `fixed_capital` に固定した短期における費用分解.

    固定費用 FC = r * K_bar と可変費用 VC(q) = w * L(q; K_bar) を
    生産関数の :meth:`~microecon.producer.base.BaseProductionFunction.get_short_run_labor`
    を用いて算出する。
    """

    def __init__(
        self,
        production: BaseProductionFunction,
        fixed_capital: float,
        wage: float,
        rental: float,
    ) -> None:
        if fixed_capital <= 0:
            raise ValueError(
                "Fixed capital K_bar must be strictly positive (K_bar > 0)"
            )
        if wage <= 0:
            raise ValueError(f"Wage w must be strictly positive (w > 0), got {wage}")
        if rental <= 0:
            raise ValueError(
                f"Rental price r must be strictly positive (r > 0), got {rental}"
            )
        self.production = production
        self.fixed_capital = fixed_capital
        self.wage = wage
        self.rental = rental

    def _validate_target_output(self, target_output: float) -> None:
        if target_output < 0:
            raise ValueError(
                f"Target output q must be non-negative (q >= 0), got {target_output}"
            )

    def fixed_cost(self) -> float:
        """固定費用 FC = r * K_bar を返す."""
        return float(self.rental * self.fixed_capital)

    def variable_cost(self, target_output: float) -> float:
        """短期可変費用 VC(q) = w * L(q; K_bar) を返す."""
        self._validate_target_output(target_output)
        labor = self.production.get_short_run_labor(target_output, self.fixed_capital)
        return float(self.wage * labor)

    def total_cost(self, target_output: float) -> float:
        """短期総費用 STC(q) = FC + VC(q) を返す."""
        return float(self.fixed_cost() + self.variable_cost(target_output))

    def average_fixed_cost(self, target_output: float) -> float:
        """平均固定費用 AFC(q) = FC / q を返す."""
        self._validate_target_output(target_output)
        return float(self.fixed_cost() / target_output)

    def average_variable_cost(self, target_output: float) -> float:
        """平均可変費用 AVC(q) = VC(q) / q を返す."""
        self._validate_target_output(target_output)
        return float(self.variable_cost(target_output) / target_output)

    def average_total_cost(self, target_output: float) -> float:
        """平均総費用 SATC(q) = STC(q) / q を返す."""
        self._validate_target_output(target_output)
        return float(self.total_cost(target_output) / target_output)
