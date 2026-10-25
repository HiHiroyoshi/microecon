"""シュタッケルベルク複占（同質財・逐次数量競争、企業1が先導者）."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from microecon.industrial_organization._sympy_utils import (
    DOMAIN_TOL,
    fmt,
    interior_best_response,
    latex,
    real_solutions,
    require_float,
    select_best,
    validate_quantity_duopoly,
)
from microecon.industrial_organization._welfare import homogeneous_duopoly_welfare
from microecon.industrial_organization.dataclasses import OligopolyResult

_TIE_TOL = 1e-9


class StackelbergDuopoly:
    """シュタッケルベルク複占（企業1が先導者、企業2が追従者）.

    後方帰納法で解く。追従者の反応関数は ``R2(q1) = max(0, R2^I(q1))`` とし、
    先導者は区分ごとに利潤を最大化する。区分を無視して内点解を求めた後に負の数量を
    0 に丸める処理は行わない（先導者の最適解を誤るため）。

    Args:
        inverse_demand: 逆需要関数 P(q1 + q2)（総生産量のみに依存）。
        quantity_variables: ``(q1, q2)``（q1 が先導者）。
        cost_functions: ``(C1(q1), C2(q2))``。

    Raises:
        InvalidEconomicParameterError: 未宣言シンボル、MC_i(0) < 0、
            逆需要が総生産量のみに依存しない場合。
    """

    def __init__(
        self,
        inverse_demand: sp.Expr,
        quantity_variables: Sequence[sp.Symbol],
        cost_functions: Sequence[sp.Expr],
    ) -> None:
        (
            self.inverse_demand,
            self.quantity_variables,
            self.cost_functions,
            self._total_demand,
            self._total_var,
        ) = validate_quantity_duopoly(inverse_demand, quantity_variables, cost_functions)
        q1, q2 = self.quantity_variables
        self.profit_functions: tuple[sp.Expr, sp.Expr] = (
            sp.expand(self.inverse_demand * q1 - self.cost_functions[0]),
            sp.expand(self.inverse_demand * q2 - self.cost_functions[1]),
        )

    def follower_reaction_function(self) -> sp.Expr:
        """追従者の反応関数 ``R2(q1) = max(0, R2^I(q1))`` を ``Piecewise`` で返す."""
        interior = self._follower_interior()
        return sp.Piecewise((interior, interior > 0), (0, True))

    def reaction_functions(self) -> tuple[sp.Expr, sp.Expr]:
        """図示用に各社の内点反応関数 ``(R1(q2), R2(q1))``（同時手番の形）を返す."""
        q1, q2 = self.quantity_variables
        return (
            interior_best_response(self.profit_functions[0], q1),
            interior_best_response(self.profit_functions[1], q2),
        )

    def solve(self) -> OligopolyResult:
        """後方帰納法によりシュタッケルベルク均衡を求める.

        Raises:
            NoEquilibriumError: 追従者の反応関数が一意に定まらない、または
                先導者の候補が存在しない場合。
        """
        q1, q2 = self.quantity_variables
        r2_interior = self._follower_interior()
        leader_profit_interior = sp.expand(self.profit_functions[0].subs(q2, r2_interior))
        leader_profit_exit = sp.expand(self.profit_functions[0].subs(q2, 0))

        def follower_quantity(q1_val: float) -> float:
            return max(0.0, require_float(r2_interior.subs(q1, q1_val), "R2"))

        def leader_profit(q1_val: float) -> float:
            q2_val = follower_quantity(q1_val)
            return require_float(
                self.profit_functions[0].subs({q1: q1_val, q2: q2_val}), "leader profit"
            )

        def r2_value(q1_val: float) -> float:
            return require_float(r2_interior.subs(q1, q1_val), "R2")

        def price_ok(q1_val: float) -> bool:
            q2_val = follower_quantity(q1_val)
            price = require_float(self.inverse_demand.subs({q1: q1_val, q2: q2_val}), "price")
            return price >= -DOMAIN_TOL

        candidates: list[tuple[float, str]] = []
        for profit_expr, follower_produces, label in (
            (leader_profit_interior, True, "追従者が生産する領域の内点解"),
            (leader_profit_exit, False, "追従者が退出する領域の内点解"),
        ):
            foc = sp.diff(profit_expr, q1)
            soc = sp.diff(foc, q1)
            for sol in real_solutions([foc], [q1]):
                v = sol[q1]
                in_region = (r2_value(v) > DOMAIN_TOL) == follower_produces
                if v >= -DOMAIN_TOL and in_region and require_float(soc.subs(q1, v), "SOC") < 0:
                    candidates.append((max(v, 0.0), label))
        for sol in real_solutions([r2_interior], [q1]):
            if sol[q1] >= -DOMAIN_TOL:
                candidates.append((max(sol[q1], 0.0), "区分の境界点"))
        candidates.append((0.0, "q1 = 0"))
        candidates = [(v, label) for v, label in candidates if price_ok(v)]

        best_value = select_best(
            candidates,
            lambda c: leader_profit(c[0]),
            "No candidate for the leader's optimum.",
        )
        top_profit = leader_profit(best_value[0])
        # 同値なら q1 が小さい方を選ぶ
        q1_star, chosen_label = min(
            (c for c in candidates if leader_profit(c[0]) >= top_profit - _TIE_TOL),
            key=lambda c: c[0],
        )
        q2_star = follower_quantity(q1_star)
        quantities = (q1_star, q2_star)
        subs = {q1: q1_star, q2: q2_star}
        price = require_float(self.inverse_demand.subs(subs), "price")
        profits = (
            require_float(self.profit_functions[0].subs(subs), "profit"),
            require_float(self.profit_functions[1].subs(subs), "profit"),
        )
        cs, ps, welfare, dwl, note = homogeneous_duopoly_welfare(
            self._total_demand, self._total_var, quantities, price,
            self.cost_functions, self.quantity_variables,
        )
        steps = self._markdown(
            r2_interior, leader_profit_interior, leader_profit_exit,
            [(v, label, leader_profit(v)) for v, label in candidates],
            chosen_label, quantities, price, profits, cs, ps, dwl, note,
        )
        return OligopolyResult(
            model_type="stackelberg",
            prices=(price, price),
            market_price=price,
            quantities=quantities,
            profits=profits,
            consumer_surplus=cs,
            producer_surplus=ps,
            social_welfare=welfare,
            deadweight_loss=dwl,
            is_corner=q2_star <= DOMAIN_TOL,
            welfare_note=note,
            markdown_steps=steps,
        )

    def _follower_interior(self) -> sp.Expr:
        return interior_best_response(self.profit_functions[1], self.quantity_variables[1])

    def _markdown(
        self, r2_interior: sp.Expr, leader_profit_interior: sp.Expr,
        leader_profit_exit: sp.Expr, candidates: list[tuple[float, str, float]],
        chosen_label: str, quantities: tuple[float, float], price: float,
        profits: tuple[float, float], cs: float, ps: float, dwl: float | None, note: str,
    ) -> str:
        q1, q2 = self.quantity_variables
        lines = [
            "## シュタッケルベルク複占の解法過程",
            "",
            f"**逆需要関数:** $P = {latex(self.inverse_demand)}$（企業1が先導者、企業2が追従者）",
            "",
            "### ステップ1: 追従者の反応関数",
            "",
            "**追従者の利潤と FOC:**",
            f"$$\\pi_2 = {latex(self.profit_functions[1])}$$",
            f"$$\\frac{{\\partial \\pi_2}}{{\\partial {latex(q2)}}} = "
            f"{latex(sp.diff(self.profit_functions[1], q2))} = 0$$",
            "**反応関数（非負制約付き）:**",
            f"$$R_2({latex(q1)}) = \\max\\left(0, {latex(r2_interior)}\\right)$$",
            "",
            "### ステップ2: 先導者の最適化",
            "",
            "先導者は追従者の反応を織り込み、区分ごとに利潤を最大化する。",
            f"$$R_2 > 0 \\text{{ の領域: }} \\pi_1 = {latex(leader_profit_interior)}$$",
            f"$$R_2 = 0 \\text{{ の領域: }} \\pi_1 = {latex(leader_profit_exit)}$$",
            "**候補と先導者の利潤:**",
            "",
            "| 候補 | $q_1$ | $\\pi_1$ |",
            "|---|---|---|",
            *(f"| {label} | {fmt(v)} | {fmt(p)} |" for v, label, p in candidates),
            "",
            f"利潤が最大となる候補（{chosen_label}）を選ぶ。",
            "",
            "### 均衡",
            "",
            f"$${latex(q1)}^* = {fmt(quantities[0])}, \\quad {latex(q2)}^* = "
            f"{fmt(quantities[1])}, \\quad P^* = {fmt(price)}$$",
            f"$$\\pi_1^* = {fmt(profits[0])}, \\quad \\pi_2^* = {fmt(profits[1])}$$",
            f"$$CS = {fmt(cs)}, \\quad PS = {fmt(ps)}$$",
            f"$$DWL = {fmt(dwl)}$$" if dwl is not None else f"死荷重: 計算しない（{note}）",
        ]
        return "\n".join(lines)
