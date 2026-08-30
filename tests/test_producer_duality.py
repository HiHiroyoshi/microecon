"""生産者理論の理論整合性検証テストスイート.

以下の観点を検証する:

1. 確定的な数値アンカーテスト（手計算で検証済みのケース）。
2. 理論の自己無矛盾性:
   - Hotelling の補題（利潤関数の記号微分 → 供給関数・要素需要関数）。
   - Shephard の補題（生産者版、総費用関数の記号微分 → 条件付要素需要関数）。
   - PMP と CMP の整合性（DRSケース限定）。
   - 記号版（SymPy）と数値版（Python）の一致（クロスチェック）。
3. 短期費用分解（ShortRunProduction）とガード節。
4. 長期費用指標（CostAnalyzer）。

比較誤差範囲には全項目共通で pytest.approx(..., abs=1e-4) を適用する。
"""

from __future__ import annotations

import sympy
import pytest

from microecon.exceptions import InvalidEconomicParameterError
from microecon.producer.base import BaseProductionFunction
from microecon.producer.cost import CostAnalyzer, ShortRunProduction
from microecon.producer.models import (
    CESProduction,
    CobbDouglasProduction,
    CostMinResult,
    LeontiefProduction,
    LinearProduction,
    ProfitMaxResult,
)
from microecon.producer.solver import ProducerProblem

_ABS_TOL = 1e-4

_PRICE, _WAGE, _RENTAL, _OUTPUT = sympy.symbols("p w r q", positive=True)


def _symbolic_subs(
    expr: sympy.Expr,
    *,
    wage: float,
    rental: float,
    price: float | None = None,
    target_output: float | None = None,
) -> float:
    """SymPy式にp, w, r, (またはq) の具体値を代入し、Python floatへ評価する."""
    subs: dict[sympy.Symbol, float] = {_WAGE: wage, _RENTAL: rental}
    if price is not None:
        subs[_PRICE] = price
    if target_output is not None:
        subs[_OUTPUT] = target_output
    return float(expr.subs(subs).evalf())


# --------------------------------------------------------------------------
# 1. 確定的な数値アンカーテスト
# --------------------------------------------------------------------------


class TestCobbDouglasPmpAnchor:
    """コブ＝ダグラス型 PMP / DRS: A=1, alpha=beta=0.25, w=r=2, p=16."""

    @pytest.fixture
    def result(self) -> ProfitMaxResult:
        production = CobbDouglasProduction(A=1.0, alpha=0.25, beta=0.25)
        return production.solve_pmp(price=16.0, wage=2.0, rental=2.0)

    def test_optimal_labor(self, result: ProfitMaxResult) -> None:
        assert result.optimal_labor == pytest.approx(4.0, abs=_ABS_TOL)

    def test_optimal_capital(self, result: ProfitMaxResult) -> None:
        assert result.optimal_capital == pytest.approx(4.0, abs=_ABS_TOL)

    def test_optimal_output(self, result: ProfitMaxResult) -> None:
        assert result.optimal_output == pytest.approx(2.0, abs=_ABS_TOL)

    def test_optimal_profit(self, result: ProfitMaxResult) -> None:
        assert result.optimal_profit == pytest.approx(16.0, abs=_ABS_TOL)

    def test_solution_type_is_interior(self, result: ProfitMaxResult) -> None:
        assert result.solution_type == "interior"


class TestCobbDouglasCmpAnchor:
    """コブ＝ダグラス型 CMP: A=1, alpha=beta=0.5, w=4, r=9, q=12."""

    @pytest.fixture
    def result(self) -> CostMinResult:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        return production.solve_cmp(target_output=12.0, wage=4.0, rental=9.0)

    def test_optimal_labor(self, result: CostMinResult) -> None:
        assert result.optimal_labor == pytest.approx(18.0, abs=_ABS_TOL)

    def test_optimal_capital(self, result: CostMinResult) -> None:
        assert result.optimal_capital == pytest.approx(8.0, abs=_ABS_TOL)

    def test_minimum_cost(self, result: CostMinResult) -> None:
        assert result.minimum_cost == pytest.approx(144.0, abs=_ABS_TOL)


class TestLinearCmpAnchor:
    """完全代替型 CMP: a=2, b=1, w=3, r=2, q=10 (a/w > b/r につき労働のみ)."""

    @pytest.fixture
    def result(self) -> CostMinResult:
        production = LinearProduction(a=2.0, b=1.0)
        return production.solve_cmp(target_output=10.0, wage=3.0, rental=2.0)

    def test_optimal_labor(self, result: CostMinResult) -> None:
        assert result.optimal_labor == pytest.approx(5.0, abs=_ABS_TOL)

    def test_optimal_capital(self, result: CostMinResult) -> None:
        assert result.optimal_capital == pytest.approx(0.0, abs=_ABS_TOL)

    def test_minimum_cost(self, result: CostMinResult) -> None:
        assert result.minimum_cost == pytest.approx(15.0, abs=_ABS_TOL)

    def test_solution_type_is_corner(self, result: CostMinResult) -> None:
        assert result.solution_type == "corner"


class TestLeontiefCmpAnchor:
    """レオンチェフ型 CMP: a=2, b=3, w=10, r=15, q=60."""

    @pytest.fixture
    def result(self) -> CostMinResult:
        production = LeontiefProduction(a=2.0, b=3.0)
        return production.solve_cmp(target_output=60.0, wage=10.0, rental=15.0)

    def test_optimal_labor(self, result: CostMinResult) -> None:
        assert result.optimal_labor == pytest.approx(30.0, abs=_ABS_TOL)

    def test_optimal_capital(self, result: CostMinResult) -> None:
        assert result.optimal_capital == pytest.approx(20.0, abs=_ABS_TOL)

    def test_minimum_cost(self, result: CostMinResult) -> None:
        assert result.minimum_cost == pytest.approx(600.0, abs=_ABS_TOL)

    def test_solution_type_is_kink(self, result: CostMinResult) -> None:
        assert result.solution_type == "kink"


class TestCesCmpAnchor:
    """CES型 CMP: A=1, a=b=1, rho=-1 (sigma=0.5), gamma=1, w=r=2, q=10."""

    @pytest.fixture
    def result(self) -> CostMinResult:
        production = CESProduction(A=1.0, a=1.0, b=1.0, rho=-1.0, gamma=1.0)
        return production.solve_cmp(target_output=10.0, wage=2.0, rental=2.0)

    def test_optimal_labor(self, result: CostMinResult) -> None:
        assert result.optimal_labor == pytest.approx(20.0, abs=_ABS_TOL)

    def test_optimal_capital(self, result: CostMinResult) -> None:
        assert result.optimal_capital == pytest.approx(20.0, abs=_ABS_TOL)

    def test_minimum_cost(self, result: CostMinResult) -> None:
        assert result.minimum_cost == pytest.approx(80.0, abs=_ABS_TOL)


class TestCesPmpAnchor:
    """CES型 PMP / DRS: A=1, a=b=1, rho=0.5 (sigma=2), gamma=0.5, w=r=2, p=8."""

    @pytest.fixture
    def result(self) -> ProfitMaxResult:
        production = CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=0.5)
        return production.solve_pmp(price=8.0, wage=2.0, rental=2.0)

    def test_optimal_labor(self, result: ProfitMaxResult) -> None:
        assert result.optimal_labor == pytest.approx(4.0, abs=_ABS_TOL)

    def test_optimal_capital(self, result: ProfitMaxResult) -> None:
        assert result.optimal_capital == pytest.approx(4.0, abs=_ABS_TOL)

    def test_optimal_output(self, result: ProfitMaxResult) -> None:
        assert result.optimal_output == pytest.approx(4.0, abs=_ABS_TOL)

    def test_optimal_profit(self, result: ProfitMaxResult) -> None:
        assert result.optimal_profit == pytest.approx(16.0, abs=_ABS_TOL)


# --------------------------------------------------------------------------
# 2. 理論の自己無矛盾性検証
# --------------------------------------------------------------------------

# CMP整合性検証（Shephardの補題）のための (名称, 生産関数, target_output, wage, rental)。
# 全4関数型を、それぞれの数値アンカーテストと同一パラメータで網羅する。
_CMP_CASES: list[tuple[str, BaseProductionFunction, float, float, float]] = [
    (
        "cobb_douglas",
        CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5),
        12.0,
        4.0,
        9.0,
    ),
    ("linear", LinearProduction(a=2.0, b=1.0), 10.0, 3.0, 2.0),
    ("leontief", LeontiefProduction(a=2.0, b=3.0), 60.0, 10.0, 15.0),
    (
        "ces",
        CESProduction(A=1.0, a=1.0, b=1.0, rho=-1.0, gamma=1.0),
        10.0,
        2.0,
        2.0,
    ),
]
_CMP_CASE_IDS = [case[0] for case in _CMP_CASES]

# PMP整合性検証（Hotellingの補題・PMP-CMP整合性）のための
# (名称, 生産関数, price, wage, rental)。DRSケースのみ（内点解が存在する）。
_PMP_DRS_CASES: list[tuple[str, BaseProductionFunction, float, float, float]] = [
    (
        "cobb_douglas_drs",
        CobbDouglasProduction(A=1.0, alpha=0.25, beta=0.25),
        16.0,
        2.0,
        2.0,
    ),
    (
        "ces_drs",
        CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=0.5),
        8.0,
        2.0,
        2.0,
    ),
]
_PMP_DRS_CASE_IDS = [case[0] for case in _PMP_DRS_CASES]


@pytest.mark.parametrize(
    "name,production,target_output,wage,rental", _CMP_CASES, ids=_CMP_CASE_IDS
)
class TestShephardsLemmaProducer:
    """全関数型の総費用関数C(q, w, r)のSymPy記号微分がL^c, K^cと一致することを検証する."""

    def test_shephards_lemma_derives_labor_demand(
        self,
        name: str,
        production: BaseProductionFunction,
        target_output: float,
        wage: float,
        rental: float,
    ) -> None:
        result = production.solve_cmp(target_output, wage, rental)
        cost_expr = production.get_cost_expression()

        dc_dw = sympy.diff(cost_expr, _WAGE)
        derived_labor = _symbolic_subs(
            dc_dw, wage=wage, rental=rental, target_output=target_output
        )
        assert derived_labor == pytest.approx(result.optimal_labor, abs=_ABS_TOL)

    def test_shephards_lemma_derives_capital_demand(
        self,
        name: str,
        production: BaseProductionFunction,
        target_output: float,
        wage: float,
        rental: float,
    ) -> None:
        result = production.solve_cmp(target_output, wage, rental)
        cost_expr = production.get_cost_expression()

        dc_dr = sympy.diff(cost_expr, _RENTAL)
        derived_capital = _symbolic_subs(
            dc_dr, wage=wage, rental=rental, target_output=target_output
        )
        assert derived_capital == pytest.approx(result.optimal_capital, abs=_ABS_TOL)

    def test_cost_expression_symbolic_matches_numeric(
        self,
        name: str,
        production: BaseProductionFunction,
        target_output: float,
        wage: float,
        rental: float,
    ) -> None:
        cost_expr = production.get_cost_expression()
        symbolic_value = _symbolic_subs(
            cost_expr, wage=wage, rental=rental, target_output=target_output
        )
        numeric_value = production.calculate_cost_function(
            target_output, wage, rental
        )
        assert symbolic_value == pytest.approx(numeric_value, abs=_ABS_TOL)


@pytest.mark.parametrize(
    "name,production,price,wage,rental", _PMP_DRS_CASES, ids=_PMP_DRS_CASE_IDS
)
class TestHotellingsLemma:
    """DRSモデルの利潤関数Pi(p, w, r)のSymPy記号微分が供給関数・負の要素需要と一致することを検証する."""

    def test_hotelling_derives_supply_function(
        self,
        name: str,
        production: BaseProductionFunction,
        price: float,
        wage: float,
        rental: float,
    ) -> None:
        result = production.solve_pmp(price, wage, rental)
        profit_expr = production.get_profit_expression()

        dpi_dp = sympy.diff(profit_expr, _PRICE)
        derived_output = _symbolic_subs(dpi_dp, price=price, wage=wage, rental=rental)
        assert derived_output == pytest.approx(result.optimal_output, abs=_ABS_TOL)

    def test_hotelling_derives_negative_labor_demand(
        self,
        name: str,
        production: BaseProductionFunction,
        price: float,
        wage: float,
        rental: float,
    ) -> None:
        result = production.solve_pmp(price, wage, rental)
        profit_expr = production.get_profit_expression()

        dpi_dw = sympy.diff(profit_expr, _WAGE)
        derived_negative_labor = _symbolic_subs(
            dpi_dw, price=price, wage=wage, rental=rental
        )
        assert derived_negative_labor == pytest.approx(
            -result.optimal_labor, abs=_ABS_TOL
        )

    def test_hotelling_derives_negative_capital_demand(
        self,
        name: str,
        production: BaseProductionFunction,
        price: float,
        wage: float,
        rental: float,
    ) -> None:
        result = production.solve_pmp(price, wage, rental)
        profit_expr = production.get_profit_expression()

        dpi_dr = sympy.diff(profit_expr, _RENTAL)
        derived_negative_capital = _symbolic_subs(
            dpi_dr, price=price, wage=wage, rental=rental
        )
        assert derived_negative_capital == pytest.approx(
            -result.optimal_capital, abs=_ABS_TOL
        )

    def test_profit_expression_symbolic_matches_numeric(
        self,
        name: str,
        production: BaseProductionFunction,
        price: float,
        wage: float,
        rental: float,
    ) -> None:
        profit_expr = production.get_profit_expression()
        symbolic_value = _symbolic_subs(profit_expr, price=price, wage=wage, rental=rental)
        numeric_value = production.calculate_profit_function(price, wage, rental)
        assert symbolic_value == pytest.approx(numeric_value, abs=_ABS_TOL)


@pytest.mark.parametrize(
    "name,production,price,wage,rental", _PMP_DRS_CASES, ids=_PMP_DRS_CASE_IDS
)
class TestPmpCmpConsistency:
    """DRSケース限定: PMPの最適産出量q*をCMPの目標産出量とした際、
    CMPの解 (L^c, K^c) と最小費用がPMPの最適解 (L*, K*) と w L* + r K* に一致することを検証する。
    """

    def test_cmp_factor_demands_match_pmp_optimal_inputs(
        self,
        name: str,
        production: BaseProductionFunction,
        price: float,
        wage: float,
        rental: float,
    ) -> None:
        pmp_result = production.solve_pmp(price, wage, rental)
        cmp_result = production.solve_cmp(pmp_result.optimal_output, wage, rental)

        assert cmp_result.optimal_labor == pytest.approx(
            pmp_result.optimal_labor, abs=_ABS_TOL
        )
        assert cmp_result.optimal_capital == pytest.approx(
            pmp_result.optimal_capital, abs=_ABS_TOL
        )

    def test_cmp_minimum_cost_matches_pmp_factor_expenditure(
        self,
        name: str,
        production: BaseProductionFunction,
        price: float,
        wage: float,
        rental: float,
    ) -> None:
        pmp_result = production.solve_pmp(price, wage, rental)
        cmp_result = production.solve_cmp(pmp_result.optimal_output, wage, rental)

        factor_expenditure = (
            wage * pmp_result.optimal_labor + rental * pmp_result.optimal_capital
        )
        assert cmp_result.minimum_cost == pytest.approx(factor_expenditure, abs=_ABS_TOL)


class TestReturnsToScale:
    """コブ＝ダグラス型・CES型の規模に関する収穫の自動判定."""

    def test_cobb_douglas_drs(self) -> None:
        assert CobbDouglasProduction(A=1.0, alpha=0.25, beta=0.25).returns_to_scale == "DRS"

    def test_cobb_douglas_crs(self) -> None:
        assert CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5).returns_to_scale == "CRS"

    def test_cobb_douglas_irs(self) -> None:
        assert CobbDouglasProduction(A=1.0, alpha=0.7, beta=0.7).returns_to_scale == "IRS"

    def test_ces_drs(self) -> None:
        assert CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=0.5).returns_to_scale == "DRS"

    def test_ces_crs(self) -> None:
        assert CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=1.0).returns_to_scale == "CRS"

    def test_ces_irs(self) -> None:
        assert CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=1.5).returns_to_scale == "IRS"


class TestUnboundedProfitMaximization:
    """CRS/IRSの下ではPMPが非有界のためValueErrorを送出することを検証する."""

    def test_cobb_douglas_crs_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        with pytest.raises(ValueError):
            production.solve_pmp(price=10.0, wage=1.0, rental=1.0)

    def test_cobb_douglas_irs_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.7, beta=0.7)
        with pytest.raises(ValueError):
            production.solve_pmp(price=10.0, wage=1.0, rental=1.0)

    def test_linear_raises(self) -> None:
        production = LinearProduction(a=2.0, b=1.0)
        with pytest.raises(ValueError):
            production.solve_pmp(price=10.0, wage=1.0, rental=1.0)

    def test_leontief_raises(self) -> None:
        production = LeontiefProduction(a=2.0, b=1.0)
        with pytest.raises(ValueError):
            production.solve_pmp(price=10.0, wage=1.0, rental=1.0)

    def test_ces_crs_raises(self) -> None:
        production = CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=1.0)
        with pytest.raises(ValueError):
            production.solve_pmp(price=10.0, wage=1.0, rental=1.0)


# --------------------------------------------------------------------------
# 3. 短期費用分解（ShortRunProduction）
# --------------------------------------------------------------------------


class TestShortRunProductionGuardClauses:
    """コンストラクタのガード節（K_bar, w, rの正値制約）の検証."""

    def test_non_positive_fixed_capital_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        with pytest.raises(ValueError, match="Fixed capital K_bar"):
            ShortRunProduction(production, fixed_capital=0.0, wage=1.0, rental=1.0)

    def test_negative_fixed_capital_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        with pytest.raises(ValueError, match="Fixed capital K_bar"):
            ShortRunProduction(production, fixed_capital=-4.0, wage=1.0, rental=1.0)

    def test_non_positive_wage_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        with pytest.raises(ValueError):
            ShortRunProduction(production, fixed_capital=4.0, wage=0.0, rental=1.0)

    def test_non_positive_rental_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        with pytest.raises(ValueError):
            ShortRunProduction(production, fixed_capital=4.0, wage=1.0, rental=0.0)

    def test_negative_target_output_raises(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        short_run = ShortRunProduction(production, fixed_capital=4.0, wage=1.0, rental=1.0)
        with pytest.raises(ValueError):
            short_run.variable_cost(-1.0)


class TestShortRunCobbDouglas:
    """コブ＝ダグラス型の短期費用分解: A=1, alpha=beta=0.5, K_bar=4, w=2, r=3."""

    @pytest.fixture
    def short_run(self) -> ShortRunProduction:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        return ShortRunProduction(production, fixed_capital=4.0, wage=2.0, rental=3.0)

    def test_fixed_cost(self, short_run: ShortRunProduction) -> None:
        assert short_run.fixed_cost() == pytest.approx(12.0, abs=_ABS_TOL)

    def test_variable_cost(self, short_run: ShortRunProduction) -> None:
        # L(q=8; K_bar=4) = (8 / (1 * 4^0.5))^(1/0.5) = (8/2)^2 = 16 -> VC = 2 * 16 = 32
        assert short_run.variable_cost(8.0) == pytest.approx(32.0, abs=_ABS_TOL)

    def test_total_cost(self, short_run: ShortRunProduction) -> None:
        assert short_run.total_cost(8.0) == pytest.approx(44.0, abs=_ABS_TOL)

    def test_average_variable_cost(self, short_run: ShortRunProduction) -> None:
        assert short_run.average_variable_cost(8.0) == pytest.approx(4.0, abs=_ABS_TOL)

    def test_average_fixed_cost(self, short_run: ShortRunProduction) -> None:
        assert short_run.average_fixed_cost(8.0) == pytest.approx(1.5, abs=_ABS_TOL)


class TestShortRunLinear:
    """完全代替型の短期費用分解: a=2, b=3, K_bar=5, w=1, r=1."""

    @pytest.fixture
    def short_run(self) -> ShortRunProduction:
        production = LinearProduction(a=2.0, b=3.0)
        return ShortRunProduction(production, fixed_capital=5.0, wage=1.0, rental=1.0)

    def test_variable_cost_below_capacity_is_zero(
        self, short_run: ShortRunProduction
    ) -> None:
        # b * K_bar = 15 なので q=10 <= 15 の下では L=0
        assert short_run.variable_cost(10.0) == pytest.approx(0.0, abs=_ABS_TOL)

    def test_variable_cost_above_capacity(self, short_run: ShortRunProduction) -> None:
        # q=20 > 15 なので L = (20 - 15) / 2 = 2.5 -> VC = 1 * 2.5 = 2.5
        assert short_run.variable_cost(20.0) == pytest.approx(2.5, abs=_ABS_TOL)


class TestShortRunLeontief:
    """レオンチェフ型の短期費用分解: a=2, b=3, K_bar=5, w=1, r=1."""

    @pytest.fixture
    def short_run(self) -> ShortRunProduction:
        production = LeontiefProduction(a=2.0, b=3.0)
        return ShortRunProduction(production, fixed_capital=5.0, wage=1.0, rental=1.0)

    def test_variable_cost_within_capacity(self, short_run: ShortRunProduction) -> None:
        # L = q / a = 10 / 2 = 5 -> VC = 1 * 5 = 5
        assert short_run.variable_cost(10.0) == pytest.approx(5.0, abs=_ABS_TOL)

    def test_exceeding_capacity_raises(self, short_run: ShortRunProduction) -> None:
        # b * K_bar = 15 が容量上限
        with pytest.raises(ValueError, match="exceeds fixed capital capacity"):
            short_run.variable_cost(100.0)


class TestShortRunCes:
    """CES型の短期費用分解におけるrhoの符号による分岐."""

    def test_rho_positive_below_capacity_is_zero(self) -> None:
        production = CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=1.0)
        short_run = ShortRunProduction(production, fixed_capital=100.0, wage=1.0, rental=1.0)
        assert short_run.variable_cost(1.0) == pytest.approx(0.0, abs=_ABS_TOL)

    def test_rho_negative_exceeding_asymptotic_capacity_raises(self) -> None:
        production = CESProduction(A=1.0, a=1.0, b=1.0, rho=-1.0, gamma=1.0)
        short_run = ShortRunProduction(production, fixed_capital=4.0, wage=1.0, rental=1.0)
        with pytest.raises(ValueError, match="asymptotic capacity"):
            short_run.variable_cost(1000.0)

    def test_rho_negative_within_capacity(self) -> None:
        production = CESProduction(A=1.0, a=1.0, b=1.0, rho=-1.0, gamma=1.0)
        short_run = ShortRunProduction(production, fixed_capital=4.0, wage=2.0, rental=2.0)
        # q_max = A * (b * K_bar^rho)^(gamma/rho) = (1 * 4^-1)^(1/-1) = 4
        # 長期CMPアンカー(w=r=2, q=10)ではK^c=20だが、短期ではK_barが4に固定されるため
        # 労働の需要は長期解と一致しない。ここでは正の可変費用が計算できることのみ検証する。
        assert short_run.variable_cost(0.5) > 0.0


# --------------------------------------------------------------------------
# 4. 長期費用指標（CostAnalyzer）
# --------------------------------------------------------------------------


class TestCostAnalyzer:
    """コブ＝ダグラス型(CRS)における限界費用・平均費用の一致（MC = AC）を検証する."""

    @pytest.fixture
    def analyzer(self) -> CostAnalyzer:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        return CostAnalyzer(production, wage=4.0, rental=9.0)

    def test_marginal_cost_matches_average_cost_under_crs(
        self, analyzer: CostAnalyzer
    ) -> None:
        # alpha + beta = 1 (CRS) の下では総費用関数はqに関して線形なため MC = AC。
        assert analyzer.marginal_cost(12.0) == pytest.approx(
            analyzer.average_cost(12.0), abs=_ABS_TOL
        )

    def test_average_cost_matches_cmp_anchor(self, analyzer: CostAnalyzer) -> None:
        # C(12, 4, 9) = 144 (数値アンカーテストと同一) -> AC = 144/12 = 12
        assert analyzer.average_cost(12.0) == pytest.approx(12.0, abs=_ABS_TOL)

    def test_non_positive_target_output_raises(self, analyzer: CostAnalyzer) -> None:
        with pytest.raises(ValueError):
            analyzer.marginal_cost(0.0)

    def test_analyzer_rejects_non_positive_wage(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        with pytest.raises(ValueError):
            CostAnalyzer(production, wage=0.0, rental=9.0)


# --------------------------------------------------------------------------
# 5. ファサードクラス ProducerProblem の委譲
# --------------------------------------------------------------------------


class TestProducerProblemDelegation:
    """ProducerProblemが具象生産関数クラスへ正しく委譲することを検証する."""

    def test_solve_profit_maximization_delegates(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.25, beta=0.25)
        problem = ProducerProblem(production)
        result = problem.solve_profit_maximization(price=16.0, wage=2.0, rental=2.0)
        assert result.optimal_profit == pytest.approx(16.0, abs=_ABS_TOL)

    def test_solve_cost_minimization_delegates(self) -> None:
        production = CobbDouglasProduction(A=1.0, alpha=0.5, beta=0.5)
        problem = ProducerProblem(production)
        result = problem.solve_cost_minimization(target_output=12.0, wage=4.0, rental=9.0)
        assert result.minimum_cost == pytest.approx(144.0, abs=_ABS_TOL)


# --------------------------------------------------------------------------
# 6. 経済パラメータのガード節
# --------------------------------------------------------------------------


class TestInvalidParameters:
    """異常な経済パラメータに対するガード節の検証."""

    def test_cobb_douglas_non_positive_a_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CobbDouglasProduction(A=0.0, alpha=0.5, beta=0.5)

    def test_cobb_douglas_non_positive_alpha_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CobbDouglasProduction(A=1.0, alpha=0.0, beta=0.5)

    def test_linear_non_positive_a_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            LinearProduction(a=0.0, b=1.0)

    def test_leontief_non_positive_b_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            LeontiefProduction(a=1.0, b=0.0)

    def test_ces_rho_zero_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CESProduction(A=1.0, a=1.0, b=1.0, rho=0.0, gamma=1.0)

    def test_ces_rho_greater_than_one_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CESProduction(A=1.0, a=1.0, b=1.0, rho=1.5, gamma=1.0)

    def test_ces_non_positive_gamma_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CESProduction(A=1.0, a=1.0, b=1.0, rho=0.5, gamma=0.0)
