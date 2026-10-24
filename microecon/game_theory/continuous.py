"""連続戦略空間を持つ2人ゲームの純粋戦略ナッシュ均衡探索.

利得関数を SymPy 記号式として受け取り、1階条件 (FOC)・2階条件 (SOC) による
最適反応関数の導出と、内点解・角点解（境界解）を網羅した均衡探索を行う。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Literal

import sympy as sp

from microecon.game_theory.dataclasses import ContinuousGameResult

_FOC_TOL = 1e-6
_BOUND_TOL = 1e-9
_IMAG_TOL = 1e-9
_DEDUP_TOL = 1e-8

_Regime = Literal["interior", "lower", "upper"]


class ContinuousGame:
    """連続戦略空間を持つ2人ゲーム.

    Args:
        payoff_expressions: 各プレイヤーの利得関数 ``(π_1, π_2)`` (SymPy記号式)。
            :meth:`find_pure_nash` を用いる場合、自由記号は ``strategy_variables``
            のみでなければならない（:meth:`get_best_response_expression` は
            記号パラメータを含む式にも対応する）。
        strategy_variables: 各プレイヤーの戦略変数 ``(s_1, s_2)``。
        strategy_bounds: 各プレイヤーの戦略空間 ``((lo_1, hi_1), (lo_2, hi_2))``。
            無限大 (``float('inf')``) を指定してもよい。
    """

    def __init__(
        self,
        payoff_expressions: tuple[sp.Expr, sp.Expr],
        strategy_variables: tuple[sp.Symbol, sp.Symbol],
        strategy_bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        if len(payoff_expressions) != 2 or len(strategy_variables) != 2:
            raise ValueError("ContinuousGame supports exactly 2 players.")
        if len(strategy_bounds) != 2:
            raise ValueError("strategy_bounds must contain one (lower, upper) pair per player.")
        if strategy_variables[0] == strategy_variables[1]:
            raise ValueError("Strategy variables of both players must be distinct symbols.")

        bounds: list[tuple[float, float]] = []
        for lower, upper in strategy_bounds:
            lo, hi = float(lower), float(upper)
            if math.isnan(lo) or math.isnan(hi) or not lo < hi:
                raise ValueError(
                    f"Each strategy bound must satisfy lower < upper, got ({lower}, {upper})."
                )
            bounds.append((lo, hi))

        expressions = tuple(sp.sympify(expr) for expr in payoff_expressions)

        self.payoff_expressions: tuple[sp.Expr, sp.Expr] = (expressions[0], expressions[1])
        self.strategy_variables = strategy_variables
        self.strategy_bounds: tuple[tuple[float, float], tuple[float, float]] = (
            bounds[0],
            bounds[1],
        )

        variables = list(strategy_variables)
        self._focs = tuple(
            sp.diff(expr, var) for expr, var in zip(expressions, strategy_variables)
        )
        self._socs = tuple(
            sp.diff(expr, var, 2) for expr, var in zip(expressions, strategy_variables)
        )
        self._payoff_funcs = tuple(sp.lambdify(variables, expr, "math") for expr in expressions)
        self._foc_funcs = tuple(sp.lambdify(variables, expr, "math") for expr in self._focs)
        self._soc_funcs = tuple(sp.lambdify(variables, expr, "math") for expr in self._socs)

    def get_best_response_expression(self, player_idx: int) -> sp.Expr:
        """FOC ``∂π_i/∂s_i = 0`` を解き、SOC ``∂²π_i/∂s_i² < 0`` を満たす解を返す.

        返される式は相手戦略の関数としての（戦略空間の制約を課さない）内点最適反応で
        ある。SOC を満たす解が複数ある場合は最初の解を返す。

        Raises:
            ValueError: ``player_idx`` が不正な場合、または SOC を満たす
                FOC 解が存在しない（符号が確定しない場合を含む）場合。
        """
        self._validate_player_idx(player_idx)
        own_var = self.strategy_variables[player_idx]
        for solution in sp.solve(sp.Eq(self._focs[player_idx], 0), own_var):
            soc_value = sp.simplify(self._socs[player_idx].subs(own_var, solution))
            if soc_value.is_negative:
                return sp.simplify(solution)
        raise ValueError(
            f"No first-order condition solution satisfying the second-order condition "
            f"was found for player {player_idx}."
        )

    def find_pure_nash(self) -> tuple[ContinuousGameResult, ...]:
        """内点解・有限境界を網羅した候補点から純粋戦略ナッシュ均衡を列挙する.

        各プレイヤーについて「内点 (FOC)」「有限な下限」「有限な上限」の各レジームを
        考え（無限大の境界は候補に含めない）、全組み合わせについて候補点を算出する。
        内点レジームのプレイヤーの戦略は、他プレイヤーの戦略を固定した FOC
        （両者が内点なら連立 FOC）の解として求める。各候補点 ``(s_1, s_2)`` が
        戦略空間内にあり、かつ相互最適反応（局所的最適性 + SOC + 候補点間の利得比較）
        を満たす場合に均衡として採用する。

        Raises:
            ValueError: 利得関数が戦略変数以外の記号を含む場合。
        """
        extra = set().union(*(e.free_symbols for e in self.payoff_expressions)) - set(
            self.strategy_variables
        )
        if extra:
            raise ValueError(
                "find_pure_nash requires numeric payoff expressions depending only on "
                f"the strategy variables, found extra symbols {sorted(map(str, extra))}."
            )

        regimes_per_player = [self._regimes(i) for i in range(2)]
        candidates: list[tuple[float, float]] = []
        for regimes in itertools.product(*regimes_per_player):
            candidates.extend(self._candidates_for_regimes(regimes))

        results: list[ContinuousGameResult] = []
        accepted: list[tuple[float, float]] = []
        for point in candidates:
            if any(_is_close_point(point, other) for other in accepted):
                continue
            if not all(self._within_bounds(i, point[i]) for i in range(2)):
                continue
            if not all(self._is_best_response(i, point) for i in range(2)):
                continue
            accepted.append(point)
            results.append(
                ContinuousGameResult(
                    strategies=point,
                    payoffs=(self._payoff(0, point), self._payoff(1, point)),
                    is_corner=any(self._at_bound(i, point[i]) is not None for i in range(2)),
                )
            )
        return tuple(results)

    def _validate_player_idx(self, player_idx: int) -> None:
        if player_idx not in (0, 1):
            raise ValueError(f"player_idx must be 0 or 1, got {player_idx}.")

    def _regimes(self, player_idx: int) -> list[_Regime]:
        lower, upper = self.strategy_bounds[player_idx]
        regimes: list[_Regime] = ["interior"]
        if not math.isinf(lower):
            regimes.append("lower")
        if not math.isinf(upper):
            regimes.append("upper")
        return regimes

    def _candidates_for_regimes(
        self, regimes: Sequence[_Regime]
    ) -> list[tuple[float, float]]:
        fixed: dict[sp.Symbol, float] = {}
        interior_players: list[int] = []
        for i, regime in enumerate(regimes):
            lower, upper = self.strategy_bounds[i]
            if regime == "lower":
                fixed[self.strategy_variables[i]] = lower
            elif regime == "upper":
                fixed[self.strategy_variables[i]] = upper
            else:
                interior_players.append(i)

        if not interior_players:
            return [(fixed[self.strategy_variables[0]], fixed[self.strategy_variables[1]])]

        unknowns = [self.strategy_variables[i] for i in interior_players]
        equations = [self._focs[i].subs(fixed) for i in interior_players]
        points: list[tuple[float, float]] = []
        for solution in _solve_real(equations, unknowns):
            values = {**fixed, **solution}
            points.append(
                (values[self.strategy_variables[0]], values[self.strategy_variables[1]])
            )
        return points

    def _is_best_response(self, player_idx: int, point: tuple[float, float]) -> bool:
        """``point`` において、相手戦略を所与としたプレイヤー ``player_idx`` の
        戦略が（局所的・候補点間比較の意味で）最適反応か否かを判定する."""
        derivative = self._eval(self._foc_funcs[player_idx], point)
        curvature = self._eval(self._soc_funcs[player_idx], point)
        bound = self._at_bound(player_idx, point[player_idx])

        if bound is None:
            if abs(derivative) > _FOC_TOL or not curvature < 0.0:
                return False
        elif bound == "lower":
            if derivative > _FOC_TOL:
                return False
            if abs(derivative) <= _FOC_TOL and curvature > 0.0:
                return False
        else:
            if derivative < -_FOC_TOL:
                return False
            if abs(derivative) <= _FOC_TOL and curvature > 0.0:
                return False

        own_payoff = self._payoff(player_idx, point)
        for alternative in self._own_candidates(player_idx, point):
            if self._payoff(player_idx, alternative) > own_payoff + _FOC_TOL:
                return False
        return True

    def _own_candidates(
        self, player_idx: int, point: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """相手戦略を固定したときのプレイヤー ``player_idx`` の候補点（内点 FOC 解・有限境界）."""
        other_idx = 1 - player_idx
        own_var = self.strategy_variables[player_idx]
        other_var = self.strategy_variables[other_idx]
        lower, upper = self.strategy_bounds[player_idx]

        own_values = [v for v in (lower, upper) if not math.isinf(v)]
        equation = self._focs[player_idx].subs(other_var, point[other_idx])
        for solution in _solve_real([equation], [own_var]):
            if self._within_bounds(player_idx, solution[own_var]):
                own_values.append(solution[own_var])

        alternatives: list[tuple[float, float]] = []
        for value in own_values:
            alt = [0.0, 0.0]
            alt[player_idx] = value
            alt[other_idx] = point[other_idx]
            alternatives.append((alt[0], alt[1]))
        return alternatives

    def _within_bounds(self, player_idx: int, value: float) -> bool:
        lower, upper = self.strategy_bounds[player_idx]
        return lower - _BOUND_TOL <= value <= upper + _BOUND_TOL

    def _at_bound(self, player_idx: int, value: float) -> _Regime | None:
        lower, upper = self.strategy_bounds[player_idx]
        if not math.isinf(lower) and abs(value - lower) <= _BOUND_TOL * max(1.0, abs(lower)):
            return "lower"
        if not math.isinf(upper) and abs(value - upper) <= _BOUND_TOL * max(1.0, abs(upper)):
            return "upper"
        return None

    def _payoff(self, player_idx: int, point: tuple[float, float]) -> float:
        # ``+ 0.0`` で負のゼロ (-0.0) を 0.0 に正規化する
        return self._eval(self._payoff_funcs[player_idx], point) + 0.0

    @staticmethod
    def _eval(func: Callable[..., object], point: tuple[float, float]) -> float:
        return float(func(*point))  # type: ignore[arg-type]


def _solve_real(
    equations: list[sp.Expr], unknowns: list[sp.Symbol]
) -> list[dict[sp.Symbol, float]]:
    """連立方程式を解き、全未知数が実数値に確定する解のみを ``float`` で返す."""
    try:
        solutions = sp.solve(equations, unknowns, dict=True)
    except NotImplementedError:
        return []

    real_solutions: list[dict[sp.Symbol, float]] = []
    for solution in solutions:
        if set(solution) != set(unknowns):
            continue  # 解が一意に定まらない（連続体）場合は候補としない
        values: dict[sp.Symbol, float] = {}
        for var in unknowns:
            expr = solution[var]
            if expr.free_symbols:
                break
            numeric = complex(sp.N(expr))
            if abs(numeric.imag) > _IMAG_TOL:
                break
            values[var] = numeric.real
        else:
            real_solutions.append(values)
    return real_solutions


def _is_close_point(lhs: tuple[float, float], rhs: tuple[float, float]) -> bool:
    return all(abs(a - b) <= _DEDUP_TOL * max(1.0, abs(a), abs(b)) for a, b in zip(lhs, rhs))
