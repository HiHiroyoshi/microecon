"""市場構造の比較（完全競争・同質財ベルトラン・シュタッケルベルク・クールノー・独占）."""

from __future__ import annotations

import math

import sympy as sp

from microecon.exceptions import InvalidEconomicParameterError
from microecon.industrial_organization._sympy_utils import check_free_symbols
from microecon.industrial_organization._welfare import consumer_surplus
from microecon.industrial_organization.bertrand import HomogeneousBertrand
from microecon.industrial_organization.cournot import CournotDuopoly
from microecon.industrial_organization.dataclasses import (
    MarketStructure,
    MarketStructureComparison,
    MarketStructureRow,
    OligopolyResult,
)
from microecon.industrial_organization.monopoly import Monopoly
from microecon.industrial_organization.stackelberg import StackelbergDuopoly


def compare_market_structures(
    inverse_demand: sp.Expr,
    quantity_variable: sp.Symbol,
    marginal_cost: float,
) -> MarketStructureComparison:
    """同一の需要・費用条件の下で5つの市場構造を比較する.

    計算は各モデルのクラスに委ね、ここでは再実装しない。完全競争の行は
    ``Monopoly.solve_single_price()`` の競争均衡 (P_c, Q_c) から余剰を計算する。

    Args:
        inverse_demand: 逆需要関数 P(Q)。
        quantity_variable: 数量シンボル Q。
        marginal_cost: 両社共通の一定限界費用（固定費用なし）。

    Raises:
        InvalidEconomicParameterError: 限界費用が負・非有限、または未宣言シンボルの場合。
    """
    mc = float(marginal_cost)
    if not (math.isfinite(mc) and mc >= 0.0):
        raise InvalidEconomicParameterError(
            f"marginal_cost must be finite and non-negative, got {marginal_cost}."
        )
    demand = sp.sympify(inverse_demand)
    check_free_symbols(demand, [quantity_variable], "inverse_demand")
    q = quantity_variable

    monopoly = Monopoly(demand, q, mc * q).solve_single_price()
    p_c, q_c = monopoly.competitive_price, monopoly.competitive_quantity
    competitive = MarketStructureRow(
        structure="perfect_competition",
        quantity=q_c,
        price=p_c,
        consumer_surplus=consumer_surplus(demand, q, q_c, p_c),
        producer_surplus=p_c * q_c - mc * q_c,
        deadweight_loss=0.0,
    )

    q1, q2 = sp.Dummy("q_1"), sp.Dummy("q_2")
    duopoly_demand = demand.subs(q, q1 + q2)
    costs = (mc * q1, mc * q2)
    bertrand = HomogeneousBertrand(demand, q, (mc, mc)).solve()
    stackelberg = StackelbergDuopoly(duopoly_demand, (q1, q2), costs).solve()
    cournot = CournotDuopoly(duopoly_demand, (q1, q2), costs).solve()

    rows = [
        competitive,
        _oligopoly_row("bertrand_homogeneous", bertrand),
        _oligopoly_row("stackelberg", stackelberg),
        _oligopoly_row("cournot", cournot),
        MarketStructureRow(
            structure="monopoly",
            quantity=monopoly.quantity,
            price=monopoly.price,
            consumer_surplus=monopoly.consumer_surplus,
            producer_surplus=monopoly.producer_surplus,
            deadweight_loss=monopoly.deadweight_loss,
        ),
    ]
    rows.sort(key=lambda row: -row.quantity)
    return MarketStructureComparison(rows=tuple(rows))


def _oligopoly_row(structure: MarketStructure, result: OligopolyResult) -> MarketStructureRow:
    if (
        result.market_price is None
        or result.consumer_surplus is None
        or result.deadweight_loss is None
    ):
        raise RuntimeError(f"Welfare of {structure} must be defined for constant marginal cost.")
    return MarketStructureRow(
        structure=structure,
        quantity=sum(result.quantities),
        price=result.market_price,
        consumer_surplus=result.consumer_surplus,
        producer_surplus=result.producer_surplus,
        deadweight_loss=result.deadweight_loss,
    )
