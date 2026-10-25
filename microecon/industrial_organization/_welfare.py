"""SymPy 記号式ベースの余剰・死荷重計算ヘルパー（内部用）.

符号規約は :mod:`microecon.market.welfare` に合わせる（CS・PS は積分値、DWL は非負）。
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from scipy.integrate import quad

from microecon.industrial_organization._sympy_utils import (
    constant_marginal_cost,
    real_solutions,
    require_float,
    select_best,
)

DWL_TOL = 1e-6


def integrate(expr: sp.Expr, var: sp.Symbol, lower: float, upper: float) -> float:
    """定積分を計算する。``sympy.integrate`` で閉じない場合のみ ``scipy.integrate.quad``."""
    integral = sp.integrate(expr, (var, lower, upper))
    if not integral.has(sp.Integral):
        return require_float(integral, "integral")
    func: Any = sp.lambdify(var, expr, "math")
    value, _ = quad(func, lower, upper)
    return float(value)


def consumer_surplus(
    inverse_demand: sp.Expr, var: sp.Symbol, quantity: float, price: float
) -> float:
    """CS = ∫₀^Q (P(q) − P*) dq."""
    return integrate(inverse_demand - price, var, 0.0, quantity)


def finalize_deadweight_loss(value: float) -> float:
    """DWL を検証する: −1e-6 以上 0 未満は 0.0 に切り上げ、−1e-6 未満は ``RuntimeError``."""
    if value < -DWL_TOL:
        raise RuntimeError(
            f"Deadweight loss became negative ({value}); internal welfare computation "
            "is inconsistent."
        )
    return max(value, 0.0)


def quantity_at_price(inverse_demand: sp.Expr, var: sp.Symbol, price: float) -> float:
    """P(Q) = price を満たす非負の数量 Q を返す（複数ある場合は最大のもの）."""
    candidates = [
        sol[var]
        for sol in real_solutions([inverse_demand - price], [var])
        if sol[var] >= -1e-9
    ]
    quantity = select_best(
        candidates,
        lambda q: q,
        f"No non-negative quantity satisfies P(Q) = {price}.",
    )
    return max(quantity, 0.0)


def competitive_benchmark_welfare(
    inverse_demand: sp.Expr, var: sp.Symbol, competitive_price: float
) -> tuple[float, float]:
    """限界費用一定の同質財市場の競争ベンチマーク ``(Q_c, W_c)`` を返す.

    W_c = ∫₀^{Q_c} (P(Q) − P_c) dQ（PS_c = 0 なので W_c = CS_c）。
    """
    q_c = quantity_at_price(inverse_demand, var, competitive_price)
    return q_c, consumer_surplus(inverse_demand, var, q_c, competitive_price)


def homogeneous_duopoly_welfare(
    total_demand: sp.Expr,
    total_var: sp.Symbol,
    quantities: tuple[float, float],
    price: float,
    costs: tuple[sp.Expr, sp.Expr],
    quantity_variables: tuple[sp.Symbol, sp.Symbol],
) -> tuple[float, float, float, float | None, str]:
    """同質財複占（クールノー・シュタッケルベルク）の厚生を計算する.

    Returns:
        ``(CS, PS, W, DWL, welfare_note)``。両社の限界費用が一定でない場合は
        DWL を ``None`` とし、理由を ``welfare_note`` に記載する。
    """
    total_quantity = sum(quantities)
    cs = consumer_surplus(total_demand, total_var, total_quantity, price)
    ps = 0.0
    for cost, q, q_val in zip(costs, quantity_variables, quantities):
        variable_cost = require_float(cost.subs(q, q_val) - cost.subs(q, 0), "variable cost")
        ps += price * q_val - variable_cost
    welfare = cs + ps

    marginal_costs = [constant_marginal_cost(c, q) for c, q in zip(costs, quantity_variables)]
    if any(mc is None for mc in marginal_costs):
        return (
            cs,
            ps,
            welfare,
            None,
            "限界費用が一定でないため競争ベンチマークを定義しない（対象外）。死荷重は計算しない。",
        )
    competitive_price = min(mc for mc in marginal_costs if mc is not None)
    _, competitive_welfare = competitive_benchmark_welfare(
        total_demand, total_var, competitive_price
    )
    return cs, ps, welfare, finalize_deadweight_loss(competitive_welfare - welfare), ""
