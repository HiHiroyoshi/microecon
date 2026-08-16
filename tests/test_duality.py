"""理論の自己無矛盾性（双対性・同次性・記号/数値一致）を検証するテストスイート.

各効用関数クラスについて、以下の4点をパラメトライズテストとして検証する:

1. 主問題（UMP）と双対問題（EMP）の解の一致（双対性）。
2. Roy の恒等式・Shephard の補題（SymPyによる記号微分からのマーシャル需要・
   ヒックス需要の導出）。
3. 記号版（SymPy）と数値版（Python）の間接効用関数・支出関数の値の一致
   （クロスチェック）。
4. 支出関数の価格に関する1次同次性、マーシャル需要の価格・所得に関する0次同次性。

比較誤差範囲は全項目共通で pytest.approx(..., abs=1e-4) を用いる。
"""

from __future__ import annotations

import sympy
import pytest

from microecon.consumer.base import BaseUtilityFunction
from microecon.consumer.models import (
    BudgetConstraint,
    CESUtility,
    CobbDouglasUtility,
    LeontiefUtility,
    LinearUtility,
    QuasiLinearUtility,
)
from microecon.consumer.solver import ConsumerProblem

_PRICE_X, _PRICE_Y, _INCOME = sympy.symbols("P_x P_y M", positive=True)
_TARGET_UTILITY = sympy.symbols("U", positive=True)

_ABS_TOL = 1e-4


def _cases() -> list[tuple[str, BaseUtilityFunction, BudgetConstraint]]:
    """各効用関数型を代表する (名称, 効用関数, 予算制約) の組を返す.

    それぞれ tests/test_consumer.py の数値アンカーテストと同一のパラメータを用い、
    内点解・角点解・キンク点解の各解の種類を最低1つずつ網羅する。
    """
    return [
        (
            "cobb_douglas",
            CobbDouglasUtility(A=1.0, alpha=1.0, beta=1.0),
            BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0),
        ),
        (
            "linear",
            LinearUtility(a=1.0, b=2.0),
            BudgetConstraint(price_x=2.0, price_y=3.0, income=100.0),
        ),
        (
            "quasi_linear_interior",
            QuasiLinearUtility(alpha=10.0),
            BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0),
        ),
        (
            "quasi_linear_corner",
            QuasiLinearUtility(alpha=10.0),
            BudgetConstraint(price_x=2.0, price_y=4.0, income=15.0),
        ),
        (
            "ces",
            CESUtility(a=1.0, b=1.0, rho=-1.0),
            BudgetConstraint(price_x=2.0, price_y=2.0, income=100.0),
        ),
        (
            "leontief",
            LeontiefUtility(a=1.0, b=1.0),
            BudgetConstraint(price_x=2.0, price_y=3.0, income=100.0),
        ),
    ]


_CASES = _cases()
_CASE_IDS = [case[0] for case in _CASES]


def _symbolic_subs(
    expr: sympy.Expr,
    *,
    price_x: float,
    price_y: float,
    income: float | None = None,
    target_utility: float | None = None,
) -> float:
    """SymPy式にP_x, P_y, (M または U) の具体値を代入し、Python floatへ評価する."""
    subs: dict[sympy.Symbol, float] = {_PRICE_X: price_x, _PRICE_Y: price_y}
    if income is not None:
        subs[_INCOME] = income
    if target_utility is not None:
        subs[_TARGET_UTILITY] = target_utility
    return float(expr.subs(subs).evalf())


@pytest.mark.parametrize("name,utility,budget", _CASES, ids=_CASE_IDS)
class TestPrimalDualConsistency:
    """UMPの解(x*, y*, U*)に対し、U*を目標としたEMPの解が一致することを検証する."""

    def test_hicksian_demand_matches_marshallian_demand(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        emp = utility.solve_emp(result.optimal_utility, budget.price_x, budget.price_y)

        assert emp.hicksian_x == pytest.approx(result.optimal_x, abs=_ABS_TOL)
        assert emp.hicksian_y == pytest.approx(result.optimal_y, abs=_ABS_TOL)

    def test_minimum_expenditure_matches_income(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        emp = utility.solve_emp(result.optimal_utility, budget.price_x, budget.price_y)

        assert emp.expenditure == pytest.approx(budget.income, abs=_ABS_TOL)

    def test_optimization_result_embeds_matching_dual_solution(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        """OptimizationResultに統合されたEMP解自体も主問題の解と一致すること."""
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.hicksian_x == pytest.approx(result.optimal_x, abs=_ABS_TOL)
        assert result.hicksian_y == pytest.approx(result.optimal_y, abs=_ABS_TOL)
        assert result.expenditure == pytest.approx(budget.income, abs=_ABS_TOL)


@pytest.mark.parametrize("name,utility,budget", _CASES, ids=_CASE_IDS)
class TestRoyIdentityAndShephardLemma:
    """SymPyによる記号微分を用いてRoyの恒等式・Shephardの補題を検証する."""

    def test_roys_identity_derives_marshallian_x(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        v_expr = utility.get_indirect_utility_expression()

        dv_dpx = sympy.diff(v_expr, _PRICE_X)
        dv_dm = sympy.diff(v_expr, _INCOME)
        roy_x = -dv_dpx / dv_dm

        derived_x = _symbolic_subs(
            roy_x, price_x=budget.price_x, price_y=budget.price_y, income=budget.income
        )
        assert derived_x == pytest.approx(result.optimal_x, abs=_ABS_TOL)

    def test_roys_identity_derives_marshallian_y(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        v_expr = utility.get_indirect_utility_expression()

        dv_dpy = sympy.diff(v_expr, _PRICE_Y)
        dv_dm = sympy.diff(v_expr, _INCOME)
        roy_y = -dv_dpy / dv_dm

        derived_y = _symbolic_subs(
            roy_y, price_x=budget.price_x, price_y=budget.price_y, income=budget.income
        )
        assert derived_y == pytest.approx(result.optimal_y, abs=_ABS_TOL)

    def test_shephards_lemma_derives_hicksian_x(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        e_expr = utility.get_expenditure_expression()

        de_dpx = sympy.diff(e_expr, _PRICE_X)
        derived_hx = _symbolic_subs(
            de_dpx,
            price_x=budget.price_x,
            price_y=budget.price_y,
            target_utility=result.optimal_utility,
        )
        assert derived_hx == pytest.approx(result.optimal_x, abs=_ABS_TOL)

    def test_shephards_lemma_derives_hicksian_y(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        e_expr = utility.get_expenditure_expression()

        de_dpy = sympy.diff(e_expr, _PRICE_Y)
        derived_hy = _symbolic_subs(
            de_dpy,
            price_x=budget.price_x,
            price_y=budget.price_y,
            target_utility=result.optimal_utility,
        )
        assert derived_hy == pytest.approx(result.optimal_y, abs=_ABS_TOL)


@pytest.mark.parametrize("name,utility,budget", _CASES, ids=_CASE_IDS)
class TestSymbolicNumericCrossCheck:
    """記号版（SymPy）と数値版（Python）の間接効用関数・支出関数の一致を検証する."""

    def test_indirect_utility_symbolic_matches_numeric(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        v_expr = utility.get_indirect_utility_expression()
        symbolic_value = _symbolic_subs(
            v_expr, price_x=budget.price_x, price_y=budget.price_y, income=budget.income
        )
        numeric_value = utility.calculate_indirect_utility(
            budget.price_x, budget.price_y, budget.income
        )
        assert symbolic_value == pytest.approx(numeric_value, abs=_ABS_TOL)

    def test_expenditure_symbolic_matches_numeric(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        e_expr = utility.get_expenditure_expression()

        symbolic_value = _symbolic_subs(
            e_expr,
            price_x=budget.price_x,
            price_y=budget.price_y,
            target_utility=result.optimal_utility,
        )
        numeric_value = utility.calculate_expenditure(
            budget.price_x, budget.price_y, result.optimal_utility
        )
        assert symbolic_value == pytest.approx(numeric_value, abs=_ABS_TOL)


@pytest.mark.parametrize("name,utility,budget", _CASES, ids=_CASE_IDS)
class TestHomogeneity:
    """支出関数の価格1次同次性、マーシャル需要の価格・所得0次同次性を検証する."""

    _SCALE_FACTOR = 3.0

    def test_expenditure_is_homogeneous_degree_one_in_prices(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        result = ConsumerProblem(utility=utility, budget=budget).solve()
        target_utility = result.optimal_utility
        k = self._SCALE_FACTOR

        base_expenditure = utility.calculate_expenditure(
            budget.price_x, budget.price_y, target_utility
        )
        scaled_expenditure = utility.calculate_expenditure(
            k * budget.price_x, k * budget.price_y, target_utility
        )

        assert scaled_expenditure == pytest.approx(k * base_expenditure, abs=_ABS_TOL)

    def test_marshallian_demand_is_homogeneous_degree_zero(
        self, name: str, utility: BaseUtilityFunction, budget: BudgetConstraint
    ) -> None:
        k = self._SCALE_FACTOR
        scaled_budget = BudgetConstraint(
            price_x=k * budget.price_x,
            price_y=k * budget.price_y,
            income=k * budget.income,
        )

        base_result = ConsumerProblem(utility=utility, budget=budget).solve()
        scaled_result = ConsumerProblem(utility=utility, budget=scaled_budget).solve()

        assert scaled_result.optimal_x == pytest.approx(base_result.optimal_x, abs=_ABS_TOL)
        assert scaled_result.optimal_y == pytest.approx(base_result.optimal_y, abs=_ABS_TOL)
