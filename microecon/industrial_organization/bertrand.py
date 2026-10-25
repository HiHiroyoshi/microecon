"""ベルトラン複占（価格競争）: 差別化財・同質財."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import sympy as sp

from microecon.exceptions import InvalidEconomicParameterError, NoEquilibriumError
from microecon.game_theory.continuous import ContinuousGame
from microecon.industrial_organization._sympy_utils import (
    DOMAIN_TOL,
    best_response_expression,
    check_distinct_symbols,
    check_free_symbols,
    check_nonnegative_marginal_cost_at_zero,
    check_strategy_bounds,
    constant_marginal_cost,
    continuous_game_errors,
    fmt,
    latex,
    require_float,
    to_float,
    unique_pure_nash,
)
from microecon.industrial_organization._welfare import (
    competitive_benchmark_welfare,
    consumer_surplus,
    finalize_deadweight_loss,
    quantity_at_price,
)
from microecon.industrial_organization.dataclasses import OligopolyResult
from microecon.industrial_organization.monopoly import Monopoly

_BOUND_TOL = 1e-6
_SYMMETRY_TOL = 1e-12


class DifferentiatedBertrand:
    """差別化財ベルトラン複占.

    利潤は ``π_i = p_i·D_i(p) − C_i(D_i(p))`` とし、均衡計算は
    :class:`~microecon.game_theory.continuous.ContinuousGame` へ委譲する。

    消費者余剰は、需要体系が価格のアフィン関数でヤコビ行列が対称かつ負定値の
    場合に限り、逆需要から構成した代表的消費者の二次効用 U(q) により
    ``CS = U(q*) − p*·q*`` と定義する。それ以外では CS・W・DWL を ``None`` とする。

    Args:
        demand_expressions: 各社の需要関数 ``(D1(p1, p2), D2(p1, p2))``。
        price_variables: 価格シンボル ``(p1, p2)``。
        cost_functions: 各社の費用関数 ``(C1(q), C2(q))``（``quantity_symbol`` の式）。
        quantity_symbol: 費用関数の数量シンボル。
        price_bounds: 各社の価格の範囲（有限値のみ）。

    Raises:
        InvalidEconomicParameterError: 未宣言シンボル、非有限・大小関係が不正な境界、
            MC_i(0) < 0 の場合。
    """

    def __init__(
        self,
        demand_expressions: Sequence[sp.Expr],
        price_variables: Sequence[sp.Symbol],
        cost_functions: Sequence[sp.Expr],
        quantity_symbol: sp.Symbol,
        price_bounds: Sequence[Sequence[float]],
    ) -> None:
        if len(demand_expressions) != 2 or len(price_variables) != 2 or len(cost_functions) != 2:
            raise InvalidEconomicParameterError("A duopoly needs exactly 2 firms.")
        p1, p2 = price_variables
        check_distinct_symbols([p1, p2, quantity_symbol], "price/quantity symbols")
        demands = (sp.sympify(demand_expressions[0]), sp.sympify(demand_expressions[1]))
        costs = (sp.sympify(cost_functions[0]), sp.sympify(cost_functions[1]))
        for i in range(2):
            check_free_symbols(demands[i], [p1, p2], f"demand of firm {i + 1}")
            check_free_symbols(costs[i], [quantity_symbol], f"cost function of firm {i + 1}")
            check_nonnegative_marginal_cost_at_zero(costs[i], quantity_symbol, f"firm {i + 1}")

        self.demand_expressions = demands
        self.price_variables: tuple[sp.Symbol, sp.Symbol] = (p1, p2)
        self.cost_functions = costs
        self.quantity_symbol = quantity_symbol
        self.price_bounds = check_strategy_bounds(price_bounds, require_finite=True)
        self.profit_functions: tuple[sp.Expr, sp.Expr] = (
            sp.expand(p1 * demands[0] - costs[0].subs(quantity_symbol, demands[0])),
            sp.expand(p2 * demands[1] - costs[1].subs(quantity_symbol, demands[1])),
        )

    def _game(self) -> ContinuousGame:
        with continuous_game_errors():
            return ContinuousGame(self.profit_functions, self.price_variables, self.price_bounds)

    def reaction_functions(self) -> tuple[sp.Expr, sp.Expr]:
        """各社の内点反応関数 ``(R1(p2), R2(p1))`` を返す."""
        game = self._game()
        return (best_response_expression(game, 0), best_response_expression(game, 1))

    def find_pure_nash(self) -> OligopolyResult:
        """差別化財ベルトラン均衡を求める.

        Raises:
            NoEquilibriumError: 均衡が存在しない・一意に選べない場合、
                または均衡で需要が負になる場合。
        """
        p1, p2 = self.price_variables
        equilibrium = unique_pure_nash(self._game())
        prices = (equilibrium.strategies[0], equilibrium.strategies[1])
        subs = {p1: prices[0], p2: prices[1]}
        quantities_raw = tuple(require_float(d.subs(subs), "demand") for d in self.demand_expressions)
        if any(q < -DOMAIN_TOL for q in quantities_raw):
            raise NoEquilibriumError(
                f"Demand is negative at the equilibrium prices {prices}: {quantities_raw}."
            )
        quantities = (max(quantities_raw[0], 0.0), max(quantities_raw[1], 0.0))
        profits = (equilibrium.payoffs[0], equilibrium.payoffs[1])
        ps = sum(
            p * q - self._variable_cost(i, q)
            for i, (p, q) in enumerate(zip(prices, quantities))
        )
        is_corner = any(
            abs(p - bound) <= _BOUND_TOL
            for p, bounds in zip(prices, self.price_bounds)
            for bound in bounds
        )
        cs, welfare, dwl, note, utility = self._welfare(prices, quantities)
        steps = self._markdown(prices, quantities, profits, cs, ps, dwl, note, utility)
        return OligopolyResult(
            model_type="bertrand_differentiated",
            prices=prices,
            market_price=None,
            quantities=quantities,
            profits=profits,
            consumer_surplus=cs,
            producer_surplus=ps,
            social_welfare=welfare,
            deadweight_loss=dwl,
            is_corner=is_corner,
            welfare_note=note,
            markdown_steps=steps,
        )

    def _variable_cost(self, i: int, quantity: float) -> float:
        cost = self.cost_functions[i]
        q = self.quantity_symbol
        return require_float(cost.subs(q, quantity) - cost.subs(q, 0), "variable cost")

    def _welfare(
        self, prices: tuple[float, float], quantities: tuple[float, float]
    ) -> tuple[float | None, float | None, float | None, str, sp.Expr | None]:
        """CS・W・DWL・welfare_note・効用関数 U(q) を返す."""
        p1, p2 = self.price_variables
        jacobian = sp.Matrix(
            [[sp.diff(d, p) for p in (p1, p2)] for d in self.demand_expressions]
        )
        if any(entry.free_symbols for entry in jacobian):
            return None, None, None, (
                "需要体系が価格のアフィン関数ではないため、消費者余剰を一意に定義できない（対象外）。"
            ), None
        j_num = np.array([[require_float(jacobian[i, k], "Jacobian") for k in range(2)]
                          for i in range(2)])
        if abs(j_num[0, 1] - j_num[1, 0]) > _SYMMETRY_TOL:
            return None, None, None, (
                "需要体系が可積分条件（ヤコビ行列の対称性）を満たさないため、"
                "消費者余剰を一意に定義できない。"
            ), None
        if not np.all(np.linalg.eigvalsh(j_num) < 0.0):
            return None, None, None, (
                "需要のヤコビ行列が負定値でないため、代表的消費者の効用を構成できない。"
            ), None

        # D(p) = a + J p  =>  p(q) = K (q - a),  K = J^{-1}
        # U(q) = ½ qᵀ K q − aᵀ K q  （∂U/∂q = p(q)、U(0) = 0）
        a = sp.Matrix([d.subs({p1: 0, p2: 0}) for d in self.demand_expressions])
        k_mat = jacobian.inv()
        qv = sp.Matrix(sp.symbols("q_1 q_2"))
        utility = sp.expand(
            (qv.T * k_mat * qv / 2)[0, 0] - (a.T * k_mat * qv)[0, 0]
        )

        def welfare_at(q_vals: tuple[float, float]) -> float:
            u_val = require_float(utility.subs({qv[0]: q_vals[0], qv[1]: q_vals[1]}), "utility")
            return u_val - sum(self._variable_cost(i, q) for i, q in enumerate(q_vals))

        u_star = require_float(utility.subs({qv[0]: quantities[0], qv[1]: quantities[1]}), "U")
        cs = u_star - sum(p * q for p, q in zip(prices, quantities))
        welfare = welfare_at(quantities)

        marginal_costs = [constant_marginal_cost(c, self.quantity_symbol) for c in self.cost_functions]
        if any(mc is None for mc in marginal_costs):
            return cs, welfare, None, (
                "限界費用が一定でないため競争ベンチマークを定義しない（対象外）。死荷重は計算しない。"
            ), utility
        competitive_prices = {p1: marginal_costs[0], p2: marginal_costs[1]}
        competitive_q = tuple(
            require_float(d.subs(competitive_prices), "demand") for d in self.demand_expressions
        )
        if any(q < -DOMAIN_TOL for q in competitive_q):
            return cs, welfare, None, (
                "競争ベンチマーク（p_i = MC_i）で需要が負になるため死荷重を定義しない。"
            ), utility
        competitive_welfare = welfare_at((max(competitive_q[0], 0.0), max(competitive_q[1], 0.0)))
        return cs, welfare, finalize_deadweight_loss(competitive_welfare - welfare), "", utility

    def _markdown(
        self, prices: tuple[float, float], quantities: tuple[float, float],
        profits: tuple[float, float], cs: float | None, ps: float, dwl: float | None,
        note: str, utility: sp.Expr | None,
    ) -> str:
        p1, p2 = self.price_variables
        lines = [
            "## 差別化財ベルトラン複占の解法過程",
            "",
            "**需要関数:**",
            f"$$D_1 = {latex(self.demand_expressions[0])}, \\quad "
            f"D_2 = {latex(self.demand_expressions[1])}$$",
            "",
            "**利潤式（π_i = p_i D_i − C_i(D_i)）:**",
            f"$$\\pi_1 = {latex(self.profit_functions[0])}$$",
            f"$$\\pi_2 = {latex(self.profit_functions[1])}$$",
            "",
            "**一階の条件 (FOC):**",
            f"$$\\frac{{\\partial \\pi_1}}{{\\partial {latex(p1)}}} = "
            f"{latex(sp.diff(self.profit_functions[0], p1))} = 0$$",
            f"$$\\frac{{\\partial \\pi_2}}{{\\partial {latex(p2)}}} = "
            f"{latex(sp.diff(self.profit_functions[1], p2))} = 0$$",
            "",
            "**反応関数:**",
        ]
        try:
            br1, br2 = self.reaction_functions()
            slopes = [to_float(sp.diff(br1, p2)), to_float(sp.diff(br2, p1))]
            lines.append(
                f"$$R_1({latex(p2)}) = {latex(br1)}, \\quad R_2({latex(p1)}) = {latex(br2)}$$"
            )
            if all(s is not None and s > 0 for s in slopes):
                lines.append("反応関数の傾きが正であり、価格は戦略的補完である。")
            elif all(s is not None and s < 0 for s in slopes):
                lines.append("反応関数の傾きが負であり、価格は戦略的代替である。")
        except NoEquilibriumError:
            lines.append("内点反応関数を閉形式で一意に得られないため、数値的に均衡を探索した。")
        lines += [
            "",
            "**連立による均衡:**",
            f"$${latex(p1)}^* = {fmt(prices[0])}, \\quad {latex(p2)}^* = {fmt(prices[1])}$$",
            f"$$q_1^* = {fmt(quantities[0])}, \\quad q_2^* = {fmt(quantities[1])}, \\quad "
            f"\\pi_1^* = {fmt(profits[0])}, \\quad \\pi_2^* = {fmt(profits[1])}$$",
            f"$$PS = {fmt(ps)}$$",
            "",
            "**消費者余剰:**",
        ]
        if cs is not None and utility is not None:
            lines += [
                "需要体系がアフィンでヤコビ行列が対称かつ負定値なので、逆需要を積分して"
                "代表的消費者の二次効用を構成する。",
                f"$$U(q) = {latex(utility)}$$",
                f"$$CS = U(q^*) - p^* \\cdot q^* = {fmt(cs)}$$",
                f"$$DWL = {fmt(dwl)}$$" if dwl is not None else f"死荷重: 計算しない（{note}）",
            ]
        else:
            lines.append(f"消費者余剰は None とする: {note}")
        return "\n".join(lines)


class HomogeneousBertrand:
    """同質財ベルトラン複占（限界費用一定）の解析解.

    同価格の場合は **低費用企業が全需要を獲得する** という慣行に従う
    （純粋戦略均衡を存在させるための標準的な仮定）。高費用企業は ``p_j = c_j`` を
    設定するものとして ``prices`` に格納する。

    Args:
        inverse_demand: 逆需要関数 P(Q)。
        quantity_variable: 数量シンボル Q。
        marginal_costs: 各社の一定限界費用 ``(c1, c2)``（非負・有限）。

    Raises:
        InvalidEconomicParameterError: 未宣言シンボル、負・非有限の限界費用の場合。
    """

    def __init__(
        self,
        inverse_demand: sp.Expr,
        quantity_variable: sp.Symbol,
        marginal_costs: Sequence[float],
    ) -> None:
        check_distinct_symbols([quantity_variable], "quantity_variable")
        demand = sp.sympify(inverse_demand)
        check_free_symbols(demand, [quantity_variable], "inverse_demand")
        if len(marginal_costs) != 2:
            raise InvalidEconomicParameterError("A duopoly needs exactly 2 marginal costs.")
        costs = (float(marginal_costs[0]), float(marginal_costs[1]))
        if any(not (math.isfinite(c) and c >= 0.0) for c in costs):
            raise InvalidEconomicParameterError(
                f"Marginal costs must be finite and non-negative, got {costs}."
            )
        self.inverse_demand: sp.Expr = demand
        self.quantity_variable = quantity_variable
        self.marginal_costs = costs

    def solve(self) -> OligopolyResult:
        """同質財ベルトラン均衡を解析的に求める.

        Raises:
            NoEquilibriumError: 低い方の限界費用でも需要が正にならない
              （min(c1, c2) ≥ P(0)）場合。
        """
        q = self.quantity_variable
        c1, c2 = self.marginal_costs
        choke_price = require_float(self.inverse_demand.subs(q, 0), "P(0)")
        if min(c1, c2) >= choke_price:
            raise NoEquilibriumError(
                f"No positive demand: min marginal cost {min(c1, c2)} >= P(0) = {choke_price}."
            )

        if c1 == c2:
            price = c1
            total = quantity_at_price(self.inverse_demand, q, price)
            quantities = (total / 2, total / 2)
            prices = (c1, c2)
            case = (
                "対称ケース（ベルトランのパラドックス）: 両社の限界費用が等しく、"
                "相手より少しでも高い価格をつけると需要をすべて失うため、"
                "価格は限界費用まで下がり利潤は 0 になる。"
            )
            is_corner = False
            monopoly_price = None
        else:
            low = 0 if c1 < c2 else 1
            high = 1 - low
            c_low, c_high = self.marginal_costs[low], self.marginal_costs[high]
            monopoly_price = Monopoly(self.inverse_demand, q, c_low * q).solve_single_price().price
            if monopoly_price >= c_high:
                price = c_high
                case = (
                    "費用優位ケース: 低費用企業の独占価格 "
                    f"{fmt(monopoly_price)} が高費用企業の限界費用 {fmt(c_high)} 以上なので、"
                    f"低費用企業は価格を {fmt(c_high)} に設定して相手を排除する"
                    "（同価格では低費用企業が全需要を得る慣行）。"
                )
            else:
                price = monopoly_price
                case = (
                    "抜本的費用優位ケース: 低費用企業の独占価格 "
                    f"{fmt(monopoly_price)} が高費用企業の限界費用 {fmt(c_high)} 未満なので、"
                    "独占価格を設定しても相手は参入できず、全需要を獲得する。"
                )
            total = quantity_at_price(self.inverse_demand, q, price)
            quantity_list = [0.0, 0.0]
            quantity_list[low] = total
            quantities = (quantity_list[0], quantity_list[1])
            price_list = [0.0, 0.0]
            price_list[low] = price
            price_list[high] = c_high
            prices = (price_list[0], price_list[1])
            is_corner = True

        profits = (
            (prices[0] - c1) * quantities[0] if quantities[0] > 0 else 0.0,
            (prices[1] - c2) * quantities[1] if quantities[1] > 0 else 0.0,
        )
        cs = consumer_surplus(self.inverse_demand, q, total, price)
        ps = profits[0] + profits[1]
        welfare = cs + ps
        competitive_price = min(c1, c2)
        q_c, competitive_welfare = competitive_benchmark_welfare(
            self.inverse_demand, q, competitive_price
        )
        dwl = finalize_deadweight_loss(competitive_welfare - welfare)

        steps = "\n".join(
            [
                "## 同質財ベルトラン複占の解法過程",
                "",
                "**前提:**",
                "財が同質なので、消費者は価格の低い企業からのみ購入する。需要が価格について"
                "不連続になるため FOC は使わず、ケース分けにより解析的に解く。",
                f"$$P(Q) = {latex(self.inverse_demand)}, \\quad c_1 = {fmt(c1)}, "
                f"\\quad c_2 = {fmt(c2)}$$",
                "",
                "**適用されたケース:**",
                case,
                "",
                "**均衡:**",
                f"$$p_1 = {fmt(prices[0])}, \\quad p_2 = {fmt(prices[1])}, \\quad "
                f"P^* = {fmt(price)}, \\quad Q^* = {fmt(total)}$$",
                f"$$q_1 = {fmt(quantities[0])}, \\quad q_2 = {fmt(quantities[1])}, \\quad "
                f"\\pi_1 = {fmt(profits[0])}, \\quad \\pi_2 = {fmt(profits[1])}$$",
                "",
                "**厚生（競争ベンチマーク P = min(c1, c2)）:**",
                f"$$CS = {fmt(cs)}, \\quad PS = {fmt(ps)}, \\quad Q_c = {fmt(q_c)}, "
                f"\\quad DWL = {fmt(dwl)}$$",
            ]
        )
        return OligopolyResult(
            model_type="bertrand_homogeneous",
            prices=prices,
            market_price=price,
            quantities=quantities,
            profits=profits,
            consumer_surplus=cs,
            producer_surplus=ps,
            social_welfare=welfare,
            deadweight_loss=dwl,
            is_corner=is_corner,
            welfare_note="",
            markdown_steps=steps,
        )
