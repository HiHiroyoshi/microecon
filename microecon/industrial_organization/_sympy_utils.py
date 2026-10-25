"""産業組織論モジュール共通の SymPy ヘルパー（内部用）.

``sympy.solve`` の解の選別（§2.3 の規則）、入力検証、``ContinuousGame`` への
委譲時の例外変換、解説文用の数値整形を提供する。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

import sympy as sp

from microecon.exceptions import InvalidEconomicParameterError, NoEquilibriumError
from microecon.game_theory.continuous import ContinuousGame
from microecon.game_theory.dataclasses import ContinuousGameResult

IMAG_TOL = 1e-9
DOMAIN_TOL = 1e-9
STRATEGY_DEDUP_TOL = 1e-6

T = TypeVar("T")


def to_float(expr: object) -> float | None:
    """数値式を ``float`` に変換する。実数かつ有限でなければ ``None`` を返す."""
    try:
        value = complex(sp.N(sp.sympify(expr)))
    except (TypeError, ValueError):
        return None
    if abs(value.imag) >= IMAG_TOL or not math.isfinite(value.real):
        return None
    return value.real


def require_float(expr: object, label: str) -> float:
    """数値式を ``float`` に変換する。変換できなければ ``RuntimeError``."""
    value = to_float(expr)
    if value is None:
        raise RuntimeError(f"Could not evaluate {label} to a finite real number: {expr}.")
    return value


def real_solutions(
    equations: Sequence[sp.Expr], unknowns: Sequence[sp.Symbol]
) -> list[dict[sp.Symbol, float]]:
    """§2.3 手順1: 連立方程式の解のうち、全未知数が実数かつ有限に定まるものを返す."""
    try:
        solutions = sp.solve(list(equations), list(unknowns), dict=True)
    except NotImplementedError:
        return []
    result: list[dict[sp.Symbol, float]] = []
    for solution in solutions:
        if set(solution) != set(unknowns):
            continue
        values: dict[sp.Symbol, float] = {}
        for var in unknowns:
            value = to_float(solution[var]) if not solution[var].free_symbols else None
            if value is None:
                break
            values[var] = value
        else:
            result.append(values)
    return result


def select_best(
    candidates: Iterable[T],
    objective: Callable[[T], float],
    no_candidate_message: str,
) -> T:
    """§2.3 手順4・5: 目的関数が最大の候補を返す。候補が0件なら ``NoEquilibriumError``."""
    candidate_list = list(candidates)
    if not candidate_list:
        raise NoEquilibriumError(no_candidate_message)
    return max(candidate_list, key=objective)


def check_free_symbols(expr: sp.Expr, allowed: Iterable[sp.Symbol], label: str) -> None:
    """``expr`` の自由シンボルが ``allowed`` に含まれなければ ``InvalidEconomicParameterError``."""
    allowed_set = set(allowed)
    extra = expr.free_symbols - allowed_set
    if extra:
        raise InvalidEconomicParameterError(
            f"{label} contains undeclared symbols {sorted(str(s) for s in extra)}; "
            f"allowed symbols are {sorted(str(s) for s in allowed_set)}."
        )


def check_distinct_symbols(symbols: Sequence[sp.Symbol], label: str) -> None:
    """シンボルがすべて ``sp.Symbol`` で、互いに異なることを検証する."""
    if any(not isinstance(s, sp.Symbol) for s in symbols):
        raise InvalidEconomicParameterError(f"{label} must be SymPy symbols.")
    if len(set(symbols)) != len(symbols):
        raise InvalidEconomicParameterError(f"{label} must be distinct symbols.")


def check_nonnegative_marginal_cost_at_zero(
    cost: sp.Expr, quantity: sp.Symbol, label: str
) -> None:
    """``MC(0) >= 0`` を検証する（負の限界費用は不正な入力）."""
    mc_at_zero = to_float(sp.diff(cost, quantity).subs(quantity, 0))
    if mc_at_zero is None:
        raise InvalidEconomicParameterError(f"Marginal cost of {label} at zero is not finite.")
    if mc_at_zero < 0.0:
        raise InvalidEconomicParameterError(
            f"Marginal cost of {label} must be non-negative, got MC(0) = {mc_at_zero}."
        )


def constant_marginal_cost(cost: sp.Expr, quantity: sp.Symbol) -> float | None:
    """限界費用が定数なら その値を、数量に依存するなら ``None`` を返す."""
    marginal_cost = sp.simplify(sp.diff(cost, quantity))
    if marginal_cost.free_symbols:
        return None
    return to_float(marginal_cost)


def check_strategy_bounds(
    bounds: Sequence[Sequence[float]], require_finite: bool
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``ContinuousGame`` 生成前に戦略の境界を検証する."""
    if len(bounds) != 2 or any(len(b) != 2 for b in bounds):
        raise InvalidEconomicParameterError(
            "Strategy bounds must be ((lower_1, upper_1), (lower_2, upper_2))."
        )
    checked: list[tuple[float, float]] = []
    for lower, upper in bounds:
        lo, hi = float(lower), float(upper)
        if math.isnan(lo) or math.isnan(hi):
            raise InvalidEconomicParameterError(f"Strategy bounds must not be NaN: {bounds}.")
        if require_finite and not (math.isfinite(lo) and math.isfinite(hi)):
            raise InvalidEconomicParameterError(f"Strategy bounds must be finite: {bounds}.")
        if not lo < hi:
            raise InvalidEconomicParameterError(
                f"Each strategy bound must satisfy lower < upper, got ({lo}, {hi})."
            )
        checked.append((lo, hi))
    return (checked[0], checked[1])


@contextmanager
def continuous_game_errors() -> Iterator[None]:
    """``ContinuousGame`` が送出する素の ``ValueError`` を ``NoEquilibriumError`` に変換する."""
    try:
        yield
    except ValueError as exc:
        if type(exc) is ValueError:
            raise NoEquilibriumError(f"No equilibrium found: {exc}") from exc
        raise


def unique_pure_nash(game: ContinuousGame) -> ContinuousGameResult:
    """``find_pure_nash()`` の結果を重複除去し、ちょうど1件でなければ ``NoEquilibriumError``."""
    with continuous_game_errors():
        results = game.find_pure_nash()

    unique: list[ContinuousGameResult] = []
    for res in results:
        if not any(
            all(abs(a - b) <= STRATEGY_DEDUP_TOL for a, b in zip(res.strategies, u.strategies))
            for u in unique
        ):
            unique.append(res)

    if not unique:
        raise NoEquilibriumError("No pure-strategy equilibrium exists.")
    if len(unique) > 1:
        found = [tuple(round(s, 6) for s in res.strategies) for res in unique]
        raise NoEquilibriumError(
            f"Multiple equilibria were found and none can be selected uniquely: {found}."
        )
    return unique[0]


def best_response_expression(game: ContinuousGame, player_idx: int) -> sp.Expr:
    """``ContinuousGame`` の内点反応関数を、例外変換付きで取得する."""
    with continuous_game_errors():
        return game.get_best_response_expression(player_idx)


def fmt(value: float) -> str:
    """解説文用に数値を整形する（整数値は整数表記、それ以外は小数第4位まで）."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def latex(expr: object) -> str:
    """SymPy 式の LaTeX 表現を返す."""
    return str(sp.latex(sp.sympify(expr)))


def validate_quantity_duopoly(
    inverse_demand: sp.Expr,
    quantity_variables: Sequence[sp.Symbol],
    cost_functions: Sequence[sp.Expr],
) -> tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol], tuple[sp.Expr, sp.Expr], sp.Expr, sp.Symbol]:
    """数量競争複占の入力を検証し、総生産量の逆需要 P(Q) を返す.

    Returns:
        ``(P(q1, q2), (q1, q2), (C1(q1), C2(q2)), P(Q), Q)``。``Q`` は内部用の Dummy。

    Raises:
        InvalidEconomicParameterError: シンボルの重複・未宣言、逆需要が総生産量
            q1 + q2 のみに依存しない場合、MC_i(0) < 0 の場合。
    """
    if len(quantity_variables) != 2 or len(cost_functions) != 2:
        raise InvalidEconomicParameterError("A duopoly needs exactly 2 firms.")
    q1, q2 = quantity_variables
    check_distinct_symbols([q1, q2], "quantity_variables")
    demand = sp.sympify(inverse_demand)
    check_free_symbols(demand, [q1, q2], "inverse_demand")
    costs = (sp.sympify(cost_functions[0]), sp.sympify(cost_functions[1]))
    for i, (cost, q) in enumerate(zip(costs, (q1, q2))):
        check_free_symbols(cost, [q], f"cost function of firm {i + 1}")
        check_nonnegative_marginal_cost_at_zero(cost, q, f"firm {i + 1}")

    total = sp.Dummy("Q")
    total_demand = demand.subs({q1: total, q2: 0}, simultaneous=True)
    if sp.simplify(total_demand.subs(total, q1 + q2) - demand) != 0:
        raise InvalidEconomicParameterError(
            "inverse_demand must depend only on total output q1 + q2 (homogeneous goods)."
        )
    return demand, (q1, q2), costs, total_demand, total


def interior_best_response(profit: sp.Expr, own: sp.Symbol) -> sp.Expr:
    """FOC ∂π/∂own = 0 を解き、SOC < 0 を満たす内点反応関数を一意に返す."""
    foc = sp.diff(profit, own)
    soc = sp.diff(foc, own)
    try:
        solutions = sp.solve(sp.Eq(foc, 0), own)
    except NotImplementedError:
        solutions = []
    valid = []
    for sol in solutions:
        soc_value = sp.simplify(soc.subs(own, sol))
        if soc_value.is_negative or (not soc_value.free_symbols and (to_float(soc_value) or 0) < 0):
            valid.append(sp.simplify(sol))
    if len(valid) != 1:
        raise NoEquilibriumError(
            "The best response could not be determined uniquely from the FOC/SOC "
            f"({len(valid)} candidates)."
        )
    return valid[0]
