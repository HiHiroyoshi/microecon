"""独占: 単一価格独占・第1種（完全）価格差別・第3種価格差別."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sympy as sp

from microecon.exceptions import InvalidEconomicParameterError, NoEquilibriumError
from microecon.industrial_organization._sympy_utils import (
    DOMAIN_TOL,
    check_distinct_symbols,
    check_free_symbols,
    check_nonnegative_marginal_cost_at_zero,
    fmt,
    latex,
    real_solutions,
    require_float,
    select_best,
)
from microecon.industrial_organization._welfare import (
    consumer_surplus,
    finalize_deadweight_loss,
    integrate,
)
from microecon.industrial_organization.dataclasses import MonopolyResult, ThirdDegreeResult

_CONSISTENCY_TOL = 1e-6


def _natural_monopoly_error(quantity: float, slope: float) -> NoEquilibriumError:
    return NoEquilibriumError(
        "自然独占のため対象外です: 競争均衡点 Q_c = "
        f"{fmt(quantity)} で限界費用が逓減しており (dMC/dQ = {fmt(slope)} < 0)、"
        "P = MC の競争均衡を定義できません。"
    )


class Monopoly:
    """単一市場の独占企業.

    Args:
        inverse_demand: 逆需要関数 P(Q)。
        quantity_variable: 数量シンボル Q。
        cost_function: 費用関数 C(Q)。固定費用を含んでよい。

    Raises:
        InvalidEconomicParameterError: 式に Q 以外のシンボルが含まれる場合、
            または MC(0) < 0 の場合。
    """

    def __init__(
        self,
        inverse_demand: sp.Expr,
        quantity_variable: sp.Symbol,
        cost_function: sp.Expr,
    ) -> None:
        check_distinct_symbols([quantity_variable], "quantity_variable")
        demand = sp.sympify(inverse_demand)
        cost = sp.sympify(cost_function)
        check_free_symbols(demand, [quantity_variable], "inverse_demand")
        check_free_symbols(cost, [quantity_variable], "cost_function")
        check_nonnegative_marginal_cost_at_zero(cost, quantity_variable, "the firm")

        self.inverse_demand: sp.Expr = demand
        self.quantity_variable = quantity_variable
        self.cost_function: sp.Expr = cost
        self.total_revenue: sp.Expr = sp.expand(demand * quantity_variable)
        self.marginal_revenue: sp.Expr = sp.expand(sp.diff(self.total_revenue, quantity_variable))
        self.marginal_cost: sp.Expr = sp.diff(cost, quantity_variable)

    def solve_single_price(self) -> MonopolyResult:
        """MR = MC により単一価格独占の均衡を求める.

        Raises:
            NoEquilibriumError: 独占解または競争均衡が存在しない場合、
                自然独占（競争均衡で dMC/dQ < 0）の場合。
        """
        q = self.quantity_variable
        q_m = self._solve_monopoly_quantity()
        p_m = self._eval(self.inverse_demand, q_m)
        if p_m <= 0.0:
            raise NoEquilibriumError(f"Monopoly price must be positive, got {p_m}.")
        mc_m = self._eval(self.marginal_cost, q_m)
        q_c, p_c = self._solve_competitive()

        fixed_cost = self._eval(self.cost_function, 0.0)
        variable_cost = self._eval(self.cost_function, q_m) - fixed_cost
        cs = consumer_surplus(self.inverse_demand, q, q_m, p_m)
        ps = p_m * q_m - variable_cost
        profit = ps - fixed_cost
        welfare = cs + ps
        competitive_welfare = integrate(self.inverse_demand - self.marginal_cost, q, 0.0, q_c)
        dwl_integral = integrate(self.inverse_demand - self.marginal_cost, q, q_m, q_c)
        if abs(dwl_integral - (competitive_welfare - welfare)) > _CONSISTENCY_TOL:
            raise RuntimeError(
                "DWL integral and W_c - W disagree: "
                f"{dwl_integral} vs {competitive_welfare - welfare}."
            )
        dwl = finalize_deadweight_loss(competitive_welfare - welfare)
        lerner = (p_m - mc_m) / p_m
        slope = self._eval(sp.diff(self.inverse_demand, q), q_m)
        inverse_elasticity = -slope * q_m / p_m

        steps = self._single_price_markdown(
            q_m=q_m, p_m=p_m, mc_m=mc_m, q_c=q_c, p_c=p_c, cs=cs, ps=ps, profit=profit,
            welfare=welfare, competitive_welfare=competitive_welfare, dwl=dwl,
            lerner=lerner, inverse_elasticity=inverse_elasticity,
        )
        return MonopolyResult(
            regime="single_price",
            price=p_m,
            quantity=q_m,
            competitive_price=p_c,
            competitive_quantity=q_c,
            consumer_surplus=cs,
            producer_surplus=ps,
            profit=profit,
            social_welfare=welfare,
            competitive_welfare=competitive_welfare,
            deadweight_loss=dwl,
            lerner_index=lerner,
            markdown_steps=steps,
        )

    def solve_first_degree_price_discrimination(self) -> MonopolyResult:
        """第1種（完全）価格差別: Q_c まで生産し各単位を留保価格で販売する.

        Raises:
            NoEquilibriumError: 競争均衡が存在しない場合、または自然独占の場合。
        """
        q = self.quantity_variable
        q_c, p_c = self._solve_competitive()
        fixed_cost = self._eval(self.cost_function, 0.0)
        ps = integrate(self.inverse_demand - self.marginal_cost, q, 0.0, q_c)
        profit = ps - fixed_cost

        steps = "\n".join(
            [
                "## 第1種（完全）価格差別の解法過程",
                "",
                "**留保価格での販売:**",
                "企業は各消費者の留保価格（逆需要曲線の高さ）をすべて把握しており、"
                "各単位をその留保価格 $P(Q)$ で販売する。",
                f"$$P(Q) = {latex(self.inverse_demand)}, \\quad "
                f"MC(Q) = {latex(self.marginal_cost)}$$",
                "",
                "**生産量:**",
                "追加の1単位の収入は留保価格 $P(Q)$ そのものなので、"
                "$P(Q) = MC(Q)$ となる競争均衡数量まで生産する。",
                f"$$Q_c = {fmt(q_c)}, \\quad P(Q_c) = {fmt(p_c)}$$",
                "",
                "**余剰:**",
                "すべての単位が留保価格で販売されるため消費者余剰は 0 となり、"
                "競争均衡の総余剰がすべて生産者余剰になる（死荷重は 0）。",
                f"$$PS = \\int_0^{{Q_c}} (P(Q) - MC(Q))\\,dQ = {fmt(ps)}, "
                f"\\quad CS = 0, \\quad DWL = 0$$",
                f"$$\\pi = PS - C(0) = {fmt(profit)}$$",
            ]
        )
        return MonopolyResult(
            regime="first_degree",
            price=p_c,
            quantity=q_c,
            competitive_price=p_c,
            competitive_quantity=q_c,
            consumer_surplus=0.0,
            producer_surplus=ps,
            profit=profit,
            social_welfare=ps,
            competitive_welfare=ps,
            deadweight_loss=0.0,
            lerner_index=None,
            markdown_steps=steps,
        )

    def _eval(self, expr: sp.Expr, quantity: float) -> float:
        return require_float(expr.subs(self.quantity_variable, quantity), str(expr))

    def _solve_monopoly_quantity(self) -> float:
        """MR = MC の解を §2.3 の規則（実数 → 定義域 → SOC → 利潤最大）で選ぶ."""
        q = self.quantity_variable
        mr_slope = sp.diff(self.marginal_revenue, q)
        mc_slope = sp.diff(self.marginal_cost, q)
        profit = self.total_revenue - self.cost_function

        candidates = []
        for sol in real_solutions([self.marginal_revenue - self.marginal_cost], [q]):
            q_val = sol[q]
            if q_val < -DOMAIN_TOL or self._eval(self.inverse_demand, q_val) < -DOMAIN_TOL:
                continue
            if not self._eval(mr_slope, q_val) < self._eval(mc_slope, q_val):
                continue
            candidates.append(max(q_val, 0.0))
        return select_best(
            candidates,
            lambda q_val: self._eval(profit, q_val),
            "No quantity satisfies MR = MC with the second-order condition.",
        )

    def _solve_competitive(self) -> tuple[float, float]:
        """P = MC の競争均衡 (Q_c, P_c) を求める.

        §2.3 手順1（実数）の直後、定義域・SOC の選別より前に自然独占を判定する。
        """
        q = self.quantity_variable
        mc_slope = sp.diff(self.marginal_cost, q)
        demand_slope = sp.diff(self.inverse_demand, q)

        real = [sol[q] for sol in real_solutions([self.inverse_demand - self.marginal_cost], [q])]
        for q_val in real:
            if q_val >= -DOMAIN_TOL:
                slope = self._eval(mc_slope, q_val)
                if slope < 0.0:
                    raise _natural_monopoly_error(q_val, slope)

        candidates = [
            max(q_val, 0.0)
            for q_val in real
            if q_val >= -DOMAIN_TOL
            and self._eval(self.inverse_demand, q_val) >= -DOMAIN_TOL
            and self._eval(demand_slope, q_val) < self._eval(mc_slope, q_val)
        ]
        q_c = select_best(
            candidates,
            lambda q_val: integrate(self.inverse_demand - self.marginal_cost, q, 0.0, q_val),
            "No competitive equilibrium satisfies P = MC.",
        )
        return q_c, self._eval(self.inverse_demand, q_c)

    def _single_price_markdown(
        self, *, q_m: float, p_m: float, mc_m: float, q_c: float, p_c: float, cs: float,
        ps: float, profit: float, welfare: float, competitive_welfare: float, dwl: float,
        lerner: float, inverse_elasticity: float,
    ) -> str:
        q = self.quantity_variable
        mr_slope = self._eval(sp.diff(self.marginal_revenue, q), q_m)
        mc_slope = self._eval(sp.diff(self.marginal_cost, q), q_m)
        return "\n".join(
            [
                "## 単一価格独占の解法過程",
                "",
                "**逆需要関数と費用関数:**",
                f"$$P(Q) = {latex(self.inverse_demand)}, \\quad "
                f"C(Q) = {latex(self.cost_function)}$$",
                "",
                "**総収入:**",
                f"$$TR(Q) = P(Q) \\cdot Q = {latex(self.total_revenue)}$$",
                "",
                "**限界収入 (MR):**",
                f"$$MR(Q) = \\frac{{dTR}}{{dQ}} = {latex(self.marginal_revenue)}$$",
                "",
                "**限界費用 (MC):**",
                f"$$MC(Q) = \\frac{{dC}}{{dQ}} = {latex(self.marginal_cost)}$$",
                "",
                "**利潤最大化条件（限界収入 = 限界費用）:**",
                f"$${latex(self.marginal_revenue)} = {latex(self.marginal_cost)} "
                f"\\implies Q_m = {fmt(q_m)}$$",
                "",
                "**2階条件 (SOC):**",
                f"$$MR'(Q_m) = {fmt(mr_slope)} < MC'(Q_m) = {fmt(mc_slope)}$$",
                "",
                "**独占価格:**",
                f"$$P_m = P(Q_m) = {fmt(p_m)}$$",
                "",
                "**競争均衡（P = MC）:**",
                f"$${latex(self.inverse_demand)} = {latex(self.marginal_cost)} "
                f"\\implies Q_c = {fmt(q_c)}, \\quad P_c = {fmt(p_c)}$$",
                "",
                "**消費者余剰:**",
                f"$$CS = \\int_0^{{Q_m}} (P(Q) - P_m)\\,dQ = {fmt(cs)}$$",
                "",
                "**生産者余剰と利潤:**",
                f"$$PS = P_m Q_m - (C(Q_m) - C(0)) = {fmt(ps)}, \\quad "
                f"\\pi = P_m Q_m - C(Q_m) = {fmt(profit)}$$",
                "",
                "**死荷重:**",
                f"$$DWL = \\int_{{Q_m}}^{{Q_c}} (P(Q) - MC(Q))\\,dQ = {fmt(dwl)}$$",
                f"$$W_c - W = {fmt(competitive_welfare)} - {fmt(welfare)} = {fmt(dwl)}$$",
                "",
                "**ラーナー指数:**",
                f"$$L = \\frac{{P_m - MC(Q_m)}}{{P_m}} = \\frac{{{fmt(p_m)} - {fmt(mc_m)}}}"
                f"{{{fmt(p_m)}}} = {fmt(lerner)}$$",
                "需要の価格弾力性 $\\varepsilon$ を $P_m$ で評価すると、"
                "独占の価格設定ルールによりラーナー指数は $1/\\varepsilon$ と一致する。",
                f"$$\\frac{{1}}{{\\varepsilon}} = -P'(Q_m) \\frac{{Q_m}}{{P_m}} "
                f"= {fmt(inverse_elasticity)}$$",
            ]
        )


class ThirdDegreePriceDiscrimination:
    """第3種価格差別（市場ごとに異なる価格。費用は総生産量の関数で市場間は非分離）.

    Args:
        submarket_inverse_demands: 各市場の逆需要関数 P_i(q_i)。
        submarket_quantity_variables: 各市場の数量シンボル q_i。
        cost_function: 費用関数 C(Q)（総生産量 Q の関数）。
        total_quantity_variable: 総生産量シンボル Q。

    Raises:
        InvalidEconomicParameterError: 市場の数が1以下、タプル長の不一致、
            未宣言シンボル、MC(0) < 0 の場合。
    """

    def __init__(
        self,
        submarket_inverse_demands: Sequence[sp.Expr],
        submarket_quantity_variables: Sequence[sp.Symbol],
        cost_function: sp.Expr,
        total_quantity_variable: sp.Symbol,
    ) -> None:
        demands = tuple(sp.sympify(d) for d in submarket_inverse_demands)
        variables = tuple(submarket_quantity_variables)
        if len(demands) <= 1:
            raise InvalidEconomicParameterError("Third-degree discrimination needs >= 2 markets.")
        if len(demands) != len(variables):
            raise InvalidEconomicParameterError(
                "submarket_inverse_demands and submarket_quantity_variables must have the "
                "same length."
            )
        check_distinct_symbols([*variables, total_quantity_variable], "quantity variables")
        for i, (demand, var) in enumerate(zip(demands, variables)):
            check_free_symbols(demand, [var], f"inverse demand of market {i + 1}")
        cost = sp.sympify(cost_function)
        check_free_symbols(cost, [total_quantity_variable], "cost_function")
        check_nonnegative_marginal_cost_at_zero(cost, total_quantity_variable, "the firm")

        self.submarket_inverse_demands = demands
        self.submarket_quantity_variables = variables
        self.cost_function: sp.Expr = cost
        self.total_quantity_variable = total_quantity_variable

    def solve(self) -> ThirdDegreeResult:
        """MR_i(q_i) = MC(Q) の連立により第3種価格差別の均衡を求める.

        Raises:
            NoEquilibriumError: 独占解または競争解でいずれかの市場の数量が負になる
                （市場閉鎖の角解・対象外）場合、または解が存在しない場合。
        """
        qs = self.submarket_quantity_variables
        big_q = self.total_quantity_variable
        q_sum = sp.Add(*qs)
        cost_in_q = self.cost_function.subs(big_q, q_sum)
        marginal_cost = sp.diff(self.cost_function, big_q)
        revenues = [d * q for d, q in zip(self.submarket_inverse_demands, qs)]
        profit = sp.Add(*revenues) - cost_in_q
        marginal_revenues = [sp.expand(sp.diff(r, q)) for r, q in zip(revenues, qs)]
        focs = [sp.diff(profit, q) for q in qs]
        hessian = sp.hessian(profit, qs)

        # --- 独占解 ---
        monopoly_real = real_solutions(focs, qs)
        self._check_market_closure(monopoly_real, "monopoly")
        monopoly_candidates = []
        for sol in monopoly_real:
            if not self._in_domain(sol):
                continue
            eigenvalues = np.linalg.eigvalsh(self._numeric_matrix(hessian.subs(sol)))
            if np.all(eigenvalues < 0.0):
                monopoly_candidates.append(sol)
        best = select_best(
            monopoly_candidates,
            lambda sol: require_float(profit.subs(sol), "profit"),
            "No solution of MR_i = MC(Q) satisfies the domain and the negative-definite "
            "Hessian condition.",
        )
        quantities = tuple(max(best[q], 0.0) for q in qs)
        subs_m = dict(zip(qs, quantities))
        prices = tuple(
            require_float(d.subs(q, v), "price")
            for d, q, v in zip(self.submarket_inverse_demands, qs, quantities)
        )
        total = sum(quantities)
        mc_value = require_float(marginal_cost.subs(big_q, total), "marginal cost")
        hessian_value = self._numeric_matrix(hessian.subs(subs_m))
        eigenvalues = np.linalg.eigvalsh(hessian_value)

        # --- 競争解: P_i(q_i) = MC(Q) ---
        competitive_eqs = [d - marginal_cost.subs(big_q, q_sum) for d in self.submarket_inverse_demands]
        competitive_real = real_solutions(competitive_eqs, qs)
        self._check_market_closure(competitive_real, "competitive")
        mc_slope = sp.diff(marginal_cost, big_q)
        for sol in competitive_real:
            total_c = sum(sol[q] for q in qs)
            if all(sol[q] >= -DOMAIN_TOL for q in qs):
                slope = require_float(mc_slope.subs(big_q, total_c), "MC slope")
                if slope < 0.0:
                    raise _natural_monopoly_error(total_c, slope)
        welfare_expr = sp.Add(
            *(
                sp.integrate(d, (q, 0, q))
                for d, q in zip(self.submarket_inverse_demands, qs)
            )
        ) - (cost_in_q - self.cost_function.subs(big_q, 0))
        competitive_candidates = [sol for sol in competitive_real if self._in_domain(sol)]
        best_c = select_best(
            competitive_candidates,
            lambda sol: require_float(welfare_expr.subs(sol), "welfare"),
            "No competitive equilibrium satisfies P_i = MC(Q) for all markets.",
        )
        competitive_quantities = tuple(max(best_c[q], 0.0) for q in qs)

        # --- 余剰 ---
        consumer_surpluses = tuple(
            consumer_surplus(d, q, v, p)
            for d, q, v, p in zip(self.submarket_inverse_demands, qs, quantities, prices)
        )
        fixed_cost = require_float(self.cost_function.subs(big_q, 0), "fixed cost")
        revenue = sum(p * v for p, v in zip(prices, quantities))
        total_cost = require_float(self.cost_function.subs(big_q, total), "total cost")
        ps = revenue - (total_cost - fixed_cost)
        profit_value = revenue - total_cost
        cs_total = sum(consumer_surpluses)
        welfare = cs_total + ps
        competitive_welfare = require_float(
            welfare_expr.subs(dict(zip(qs, competitive_quantities))), "competitive welfare"
        )
        dwl = finalize_deadweight_loss(competitive_welfare - welfare)
        lerner_indices = tuple((p - mc_value) / p for p in prices)
        inverse_elasticities = tuple(
            -require_float(sp.diff(d, q).subs(q, v), "demand slope") * v / p
            for d, q, v, p in zip(self.submarket_inverse_demands, qs, quantities, prices)
        )

        steps = self._markdown(
            marginal_revenues=marginal_revenues,
            marginal_cost=marginal_cost,
            quantities=quantities,
            prices=prices,
            mc_value=mc_value,
            hessian=hessian_value,
            eigenvalues=tuple(float(e) for e in eigenvalues),
            inverse_elasticities=inverse_elasticities,
            consumer_surpluses=consumer_surpluses,
            ps=ps,
            profit=profit_value,
            competitive_quantities=competitive_quantities,
            competitive_welfare=competitive_welfare,
            welfare=welfare,
            dwl=dwl,
        )
        return ThirdDegreeResult(
            prices=prices,
            quantities=quantities,
            consumer_surpluses=consumer_surpluses,
            lerner_indices=lerner_indices,
            marginal_cost=mc_value,
            total_quantity=total,
            consumer_surplus=cs_total,
            producer_surplus=ps,
            profit=profit_value,
            social_welfare=welfare,
            competitive_welfare=competitive_welfare,
            deadweight_loss=dwl,
            markdown_steps=steps,
        )

    def _in_domain(self, sol: dict[sp.Symbol, float]) -> bool:
        for demand, q in zip(self.submarket_inverse_demands, self.submarket_quantity_variables):
            if sol[q] < -DOMAIN_TOL:
                return False
            if require_float(demand.subs(q, sol[q]), "price") < -DOMAIN_TOL:
                return False
        return True

    def _check_market_closure(self, solutions: list[dict[sp.Symbol, float]], label: str) -> None:
        """§2.3 手順1の直後・定義域の選別より前に、市場閉鎖（負の数量）を判定する.

        非負の数量を持つ実数解が1つもなく、負の数量を含む解がある場合に送出する。
        """
        qs = self.submarket_quantity_variables
        if any(all(sol[q] >= -DOMAIN_TOL for q in qs) for sol in solutions):
            return
        closed = sorted(
            {i + 1 for sol in solutions for i, q in enumerate(qs) if sol[q] < -DOMAIN_TOL}
        )
        if closed:
            raise NoEquilibriumError(
                f"市場閉鎖の角解は対象外です: {label} 解で市場 {closed} の数量が負になります。"
            )

    @staticmethod
    def _numeric_matrix(matrix: sp.Matrix) -> np.ndarray:
        return np.array(
            [[require_float(matrix[i, j], "Hessian") for j in range(matrix.cols)]
             for i in range(matrix.rows)],
            dtype=float,
        )

    def _markdown(
        self, *, marginal_revenues: list[sp.Expr], marginal_cost: sp.Expr,
        quantities: tuple[float, ...], prices: tuple[float, ...], mc_value: float,
        hessian: np.ndarray, eigenvalues: tuple[float, ...],
        inverse_elasticities: tuple[float, ...], consumer_surpluses: tuple[float, ...],
        ps: float, profit: float, competitive_quantities: tuple[float, ...],
        competitive_welfare: float, welfare: float, dwl: float,
    ) -> str:
        lines = [
            "## 第3種価格差別の解法過程",
            "",
            "**費用関数（総生産量の関数・市場間で非分離）:**",
            f"$$C(Q) = {latex(self.cost_function)}, \\quad Q = "
            f"{' + '.join(latex(q) for q in self.submarket_quantity_variables)}, \\quad "
            f"MC(Q) = {latex(marginal_cost)}$$",
            "",
            "**各市場の限界収入:**",
        ]
        for i, (demand, mr) in enumerate(zip(self.submarket_inverse_demands, marginal_revenues)):
            lines.append(f"$$P_{{{i + 1}}} = {latex(demand)}, \\quad MR_{{{i + 1}}} = {latex(mr)}$$")
        lines += [
            "",
            "**利潤最大化条件（MR_i = MC(Q) の連立）:**",
            *(
                f"$${latex(mr)} = {latex(marginal_cost.subs(self.total_quantity_variable, sp.Add(*self.submarket_quantity_variables)))}$$"
                for mr in marginal_revenues
            ),
            "$$\\implies "
            + ", \\quad ".join(
                f"q_{{{i + 1}}}^* = {fmt(v)}" for i, v in enumerate(quantities)
            )
            + f", \\quad MC = {fmt(mc_value)}$$",
            "",
            "**2階条件（ヘッセ行列の負定値性）:**",
            "$$H = \\begin{pmatrix} "
            + " \\\\ ".join(" & ".join(fmt(float(v)) for v in row) for row in hessian)
            + " \\end{pmatrix}, \\quad \\text{固有値} = "
            + ", ".join(fmt(e) for e in eigenvalues)
            + " < 0$$",
            "",
            "**価格・弾力性・ラーナー指数:**",
        ]
        for i, (p, inv_e) in enumerate(zip(prices, inverse_elasticities)):
            lines.append(
                f"$$P_{{{i + 1}}}^* = {fmt(p)}, \\quad \\varepsilon_{{{i + 1}}} = {fmt(1 / inv_e)}, "
                f"\\quad L_{{{i + 1}}} = \\frac{{P_{{{i + 1}}} - MC}}{{P_{{{i + 1}}}}} = "
                f"{fmt((p - mc_value) / p)} = \\frac{{1}}{{\\varepsilon_{{{i + 1}}}}}$$"
            )
        order = sorted(range(len(prices)), key=lambda i: 1 / inverse_elasticities[i])
        lines += [
            "弾力性が低い市場ほど高価格になる（逆弾力性ルール）: 弾力性の低い順に "
            + " → ".join(
                f"市場{i + 1}（$\\varepsilon$ = {fmt(1 / inverse_elasticities[i])}, "
                f"$P$ = {fmt(prices[i])}）"
                for i in order
            )
            + "。",
            "",
            "**余剰:**",
            "$$"
            + ", \\quad ".join(f"CS_{{{i + 1}}} = {fmt(v)}" for i, v in enumerate(consumer_surpluses))
            + f", \\quad PS = {fmt(ps)}, \\quad \\pi = {fmt(profit)}$$",
            "",
            "**競争均衡（P_i = MC(Q)）と死荷重:**",
            "$$"
            + ", \\quad ".join(
                f"q_{{{i + 1}}}^c = {fmt(v)}" for i, v in enumerate(competitive_quantities)
            )
            + f", \\quad DWL = W_c - W = {fmt(competitive_welfare)} - {fmt(welfare)} "
            f"= {fmt(dwl)}$$",
        ]
        return "\n".join(lines)
