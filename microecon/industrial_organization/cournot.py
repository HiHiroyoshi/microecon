"""クールノー複占（同質財・同時数量競争）."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from microecon.exceptions import NoEquilibriumError
from microecon.game_theory.continuous import ContinuousGame
from microecon.industrial_organization._sympy_utils import (
    best_response_expression,
    continuous_game_errors,
    fmt,
    latex,
    require_float,
    unique_pure_nash,
    validate_quantity_duopoly,
)
from microecon.industrial_organization._welfare import homogeneous_duopoly_welfare
from microecon.industrial_organization.dataclasses import OligopolyResult

_INF = float("inf")


class CournotDuopoly:
    """クールノー複占.

    均衡計算は :class:`~microecon.game_theory.continuous.ContinuousGame` を内部で
    生成して委譲する（コンポジション）。

    Args:
        inverse_demand: 逆需要関数 P(q1 + q2)（``q1``, ``q2`` の式で、総生産量のみに依存）。
        quantity_variables: 各社の数量シンボル ``(q1, q2)``。
        cost_functions: 各社の費用関数 ``(C1(q1), C2(q2))``（各社自身の数量の式）。

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

    def _game(self) -> ContinuousGame:
        with continuous_game_errors():
            return ContinuousGame(
                self.profit_functions,
                self.quantity_variables,
                ((0.0, _INF), (0.0, _INF)),
            )

    def reaction_functions(self) -> tuple[sp.Expr, sp.Expr]:
        """各社の内点反応関数 ``(R1(q2), R2(q1))``（非負制約を課さない）を返す."""
        game = self._game()
        return (best_response_expression(game, 0), best_response_expression(game, 1))

    def solve(self) -> OligopolyResult:
        """クールノー均衡を求める.

        Raises:
            NoEquilibriumError: 純粋戦略均衡が存在しない、または一意に選べない場合。
        """
        q1, q2 = self.quantity_variables
        equilibrium = unique_pure_nash(self._game())
        quantities = (equilibrium.strategies[0], equilibrium.strategies[1])
        price = require_float(
            self.inverse_demand.subs({q1: quantities[0], q2: quantities[1]}), "price"
        )
        profits = (equilibrium.payoffs[0], equilibrium.payoffs[1])
        cs, ps, welfare, dwl, note = homogeneous_duopoly_welfare(
            self._total_demand, self._total_var, quantities, price,
            self.cost_functions, self.quantity_variables,
        )
        return OligopolyResult(
            model_type="cournot",
            prices=(price, price),
            market_price=price,
            quantities=quantities,
            profits=profits,
            consumer_surplus=cs,
            producer_surplus=ps,
            social_welfare=welfare,
            deadweight_loss=dwl,
            is_corner=equilibrium.is_corner,
            welfare_note=note,
            markdown_steps=self._markdown(quantities, price, profits, cs, ps, dwl, note),
        )

    def _markdown(
        self, quantities: tuple[float, float], price: float, profits: tuple[float, float],
        cs: float, ps: float, dwl: float | None, note: str,
    ) -> str:
        q1, q2 = self.quantity_variables
        lines = [
            "## クールノー複占の解法過程",
            "",
            "**逆需要関数と利潤:**",
            f"$$P = {latex(self.inverse_demand)}$$",
            f"$$\\pi_1 = {latex(self.profit_functions[0])}, \\quad "
            f"\\pi_2 = {latex(self.profit_functions[1])}$$",
            "",
            "**各社の一階の条件 (FOC):**",
            f"$$\\frac{{\\partial \\pi_1}}{{\\partial {latex(q1)}}} = "
            f"{latex(sp.diff(self.profit_functions[0], q1))} = 0$$",
            f"$$\\frac{{\\partial \\pi_2}}{{\\partial {latex(q2)}}} = "
            f"{latex(sp.diff(self.profit_functions[1], q2))} = 0$$",
            "",
            "**反応関数:**",
        ]
        try:
            br1, br2 = self.reaction_functions()
            lines += [
                f"$$R_1({latex(q2)}) = \\max\\left(0, {latex(br1)}\\right), \\quad "
                f"R_2({latex(q1)}) = \\max\\left(0, {latex(br2)}\\right)$$",
                "反応関数の傾きが負であり、数量は戦略的代替である。",
            ]
        except NoEquilibriumError:
            lines.append("内点反応関数を閉形式で一意に得られないため、数値的に均衡を探索した。")
        lines += [
            "",
            "**連立による均衡:**",
            "両社の反応関数を連立して解く（境界 $q_i = 0$ の角点も候補に含める）。",
            f"$${latex(q1)}^* = {fmt(quantities[0])}, \\quad {latex(q2)}^* = "
            f"{fmt(quantities[1])}, \\quad P^* = {fmt(price)}$$",
            f"$$\\pi_1^* = {fmt(profits[0])}, \\quad \\pi_2^* = {fmt(profits[1])}$$",
            "",
            "**厚生:**",
            f"$$CS = \\int_0^{{Q^*}} (P(Q) - P^*)\\,dQ = {fmt(cs)}, \\quad PS = {fmt(ps)}$$",
            f"$$DWL = {fmt(dwl)}$$" if dwl is not None else f"死荷重: 計算しない（{note}）",
        ]
        return "\n".join(lines)
