"""microecon.market の単体テスト（市場均衡・厚生分析・課税影響分析）."""

from __future__ import annotations

import math

import pytest
import sympy

from microecon.exceptions import InvalidEconomicParameterError, NoEquilibriumError
from microecon.market.curves import BaseDemandCurve
from microecon.market.dataclasses import EquilibriumResult, TaxImpactResult, WelfareResult
from microecon.market.equilibrium import MarketEquilibrium
from microecon.market.taxation import TaxImpactAnalyzer
from microecon.market.welfare import WelfareAnalyzer
from microecon.market import (
    ConstantElasticityDemandCurve,
    ConstantElasticitySupplyCurve,
    LinearDemandCurve,
    LinearSupplyCurve,
)

APPROX_ABS = 1e-4


# --------------------------------------------------------------------------
# 1. 線形市場モデル（課税なし）
# --------------------------------------------------------------------------


@pytest.fixture
def linear_demand() -> LinearDemandCurve:
    return LinearDemandCurve(a=100.0, b=2.0)


@pytest.fixture
def linear_supply() -> LinearSupplyCurve:
    return LinearSupplyCurve(c=-20.0, d=2.0)


class TestLinearMarketEquilibrium:
    """P* = 30, Q* = 40 が閉形式解で求まることの検証."""

    def test_equilibrium_price_and_quantity(
        self, linear_demand: LinearDemandCurve, linear_supply: LinearSupplyCurve
    ) -> None:
        result = MarketEquilibrium(linear_demand, linear_supply).solve()
        assert result.price == pytest.approx(30.0, abs=APPROX_ABS)
        assert result.quantity == pytest.approx(40.0, abs=APPROX_ABS)
        assert result.is_closed_form is True


class TestLinearMarketWelfareNoTax:
    """calculate_welfare(30, 30, 40) が CS=400, PS=400, GR=0, TS=800, DWL=0 となることの検証."""

    @pytest.fixture
    def welfare(
        self, linear_demand: LinearDemandCurve, linear_supply: LinearSupplyCurve
    ) -> WelfareResult:
        analyzer = WelfareAnalyzer(linear_demand, linear_supply)
        return analyzer.calculate_welfare(buyer_price=30.0, seller_price=30.0, quantity=40.0)

    def test_consumer_surplus(self, welfare: WelfareResult) -> None:
        assert welfare.consumer_surplus == pytest.approx(400.0, abs=APPROX_ABS)

    def test_producer_surplus(self, welfare: WelfareResult) -> None:
        assert welfare.producer_surplus == pytest.approx(400.0, abs=APPROX_ABS)

    def test_government_revenue(self, welfare: WelfareResult) -> None:
        assert welfare.government_revenue == pytest.approx(0.0, abs=APPROX_ABS)

    def test_total_surplus(self, welfare: WelfareResult) -> None:
        assert welfare.total_surplus == pytest.approx(800.0, abs=APPROX_ABS)

    def test_deadweight_loss(self, welfare: WelfareResult) -> None:
        assert welfare.deadweight_loss == pytest.approx(0.0, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 2. 線形市場モデル（従量税 t = 8 課税時）
# --------------------------------------------------------------------------


class TestLinearMarketWithTax:
    """t = 8 の従量税で Q_t=32, P_b=34, P_s=26 となり、
    厚生が CS=256, PS=256, GR=256, TS=768, DWL=32 となることの検証."""

    @pytest.fixture
    def tax_result(
        self, linear_demand: LinearDemandCurve, linear_supply: LinearSupplyCurve
    ) -> TaxImpactResult:
        analyzer = TaxImpactAnalyzer(linear_demand, linear_supply)
        return analyzer.analyze_specific_tax(8.0)

    def test_taxed_quantity(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.taxed_quantity == pytest.approx(32.0, abs=APPROX_ABS)

    def test_buyer_price(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.buyer_price == pytest.approx(34.0, abs=APPROX_ABS)

    def test_seller_price(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.seller_price == pytest.approx(26.0, abs=APPROX_ABS)

    def test_tax_shares_are_symmetric(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.buyer_tax_share == pytest.approx(0.5, abs=APPROX_ABS)
        assert tax_result.seller_tax_share == pytest.approx(0.5, abs=APPROX_ABS)

    def test_elasticity_predicted_share_matches_actual(
        self, tax_result: TaxImpactResult
    ) -> None:
        assert tax_result.elasticity_predicted_buyer_share == pytest.approx(
            0.5, abs=APPROX_ABS
        )

    def test_welfare_consumer_surplus(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.welfare.consumer_surplus == pytest.approx(
            256.0, abs=APPROX_ABS
        )

    def test_welfare_producer_surplus(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.welfare.producer_surplus == pytest.approx(
            256.0, abs=APPROX_ABS
        )

    def test_welfare_government_revenue(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.welfare.government_revenue == pytest.approx(
            256.0, abs=APPROX_ABS
        )

    def test_welfare_total_surplus(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.welfare.total_surplus == pytest.approx(
            768.0, abs=APPROX_ABS
        )

    def test_welfare_deadweight_loss(self, tax_result: TaxImpactResult) -> None:
        assert tax_result.welfare.deadweight_loss == pytest.approx(
            32.0, abs=APPROX_ABS
        )


# --------------------------------------------------------------------------
# 3. 弾力性一定型市場モデル（収束ケース: epsilon = 2 > 1, eta = 1）
# --------------------------------------------------------------------------


class TestConstantElasticityMarketConvergentCase:
    """CS・PS の数値積分結果が解析解 (20 sqrt(Q*) - P*Q*, P*Q* - 0.05 (Q*)^2) と一致すること."""

    @pytest.fixture
    def ce_demand(self) -> ConstantElasticityDemandCurve:
        return ConstantElasticityDemandCurve(A=100.0, epsilon=2.0)

    @pytest.fixture
    def ce_supply(self) -> ConstantElasticitySupplyCurve:
        return ConstantElasticitySupplyCurve(B=10.0, eta=1.0)

    @pytest.fixture
    def equilibrium(
        self,
        ce_demand: ConstantElasticityDemandCurve,
        ce_supply: ConstantElasticitySupplyCurve,
    ) -> EquilibriumResult:
        return MarketEquilibrium(ce_demand, ce_supply).solve()

    def test_equilibrium_price_and_quantity(
        self, equilibrium: EquilibriumResult
    ) -> None:
        expected_price = 10.0 ** (1.0 / 3.0)
        assert equilibrium.price == pytest.approx(expected_price, abs=APPROX_ABS)
        assert equilibrium.quantity == pytest.approx(
            10.0 * expected_price, abs=APPROX_ABS
        )
        assert equilibrium.is_closed_form is True

    def test_consumer_and_producer_surplus_match_analytic_solution(
        self,
        ce_demand: ConstantElasticityDemandCurve,
        ce_supply: ConstantElasticitySupplyCurve,
        equilibrium: EquilibriumResult,
    ) -> None:
        p_star, q_star = equilibrium.price, equilibrium.quantity
        analyzer = WelfareAnalyzer(ce_demand, ce_supply)
        welfare = analyzer.calculate_welfare(
            buyer_price=p_star, seller_price=p_star, quantity=q_star
        )

        analytic_cs = 20.0 * q_star**0.5 - p_star * q_star
        analytic_ps = p_star * q_star - 0.05 * q_star**2

        assert welfare.consumer_surplus == pytest.approx(analytic_cs, abs=APPROX_ABS)
        assert welfare.producer_surplus == pytest.approx(analytic_ps, abs=APPROX_ABS)
        assert welfare.deadweight_loss == pytest.approx(0.0, abs=APPROX_ABS)


class TestConstantElasticityDemandDivergence:
    """epsilon <= 1 の需要曲線では消費者余剰が発散すること."""

    def test_is_surplus_divergent_flag(self) -> None:
        demand = ConstantElasticityDemandCurve(A=100.0, epsilon=1.0)
        assert demand.is_surplus_divergent is True

    def test_consumer_surplus_is_infinite(self) -> None:
        demand = ConstantElasticityDemandCurve(A=100.0, epsilon=1.0)
        supply = ConstantElasticitySupplyCurve(B=10.0, eta=1.0)
        analyzer = WelfareAnalyzer(demand, supply)
        equilibrium = MarketEquilibrium(demand, supply).solve()

        welfare = analyzer.calculate_welfare(
            buyer_price=equilibrium.price,
            seller_price=equilibrium.price,
            quantity=equilibrium.quantity,
        )
        assert math.isinf(welfare.consumer_surplus)


# --------------------------------------------------------------------------
# 4. 厚生経済学の基本定理・弾力性と税負担の帰属法則
# --------------------------------------------------------------------------


class TestFundamentalTheoremOfWelfareEconomics:
    """t = 0 のとき総余剰が極大化（DWL = 0）となることの検証."""

    def test_zero_tax_maximizes_total_surplus(
        self, linear_demand: LinearDemandCurve, linear_supply: LinearSupplyCurve
    ) -> None:
        analyzer = TaxImpactAnalyzer(linear_demand, linear_supply)

        no_tax = analyzer.analyze_specific_tax(0.0)
        with_tax = analyzer.analyze_specific_tax(8.0)

        assert no_tax.welfare.deadweight_loss == pytest.approx(0.0, abs=APPROX_ABS)
        assert no_tax.welfare.total_surplus > with_tax.welfare.total_surplus


class TestInelasticDemandTaxIncidence:
    """完全非弾力的な需要への近似（極めて小さい b）では、税の実質全額が
    買い手に転嫁され (P_b ~= P* + t)、Q_t ~= Q* かつ DWL ~= 0 となること."""

    def test_buyer_bears_almost_all_tax(self) -> None:
        demand = LinearDemandCurve(a=100_000.0, b=1e-6)
        supply = LinearSupplyCurve(c=-20.0, d=2.0)

        equilibrium = MarketEquilibrium(demand, supply).solve()
        analyzer = TaxImpactAnalyzer(demand, supply)
        result = analyzer.analyze_specific_tax(5.0)

        assert result.buyer_price == pytest.approx(
            equilibrium.price + 5.0, abs=APPROX_ABS
        )
        assert result.taxed_quantity == pytest.approx(
            equilibrium.quantity, abs=APPROX_ABS
        )
        assert result.buyer_tax_share == pytest.approx(1.0, abs=APPROX_ABS)
        assert result.elasticity_predicted_buyer_share == pytest.approx(
            1.0, abs=APPROX_ABS
        )
        assert result.welfare.deadweight_loss == pytest.approx(0.0, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 5. 無効なパラメータの即座例外処理（ガード節）
# --------------------------------------------------------------------------


class TestInvalidParameterGuards:
    def test_negative_tax_raises(
        self, linear_demand: LinearDemandCurve, linear_supply: LinearSupplyCurve
    ) -> None:
        analyzer = TaxImpactAnalyzer(linear_demand, linear_supply)
        with pytest.raises(InvalidEconomicParameterError):
            analyzer.analyze_specific_tax(-1.0)

    def test_non_intersecting_curves_raise_no_equilibrium_error(self) -> None:
        demand = LinearDemandCurve(a=10.0, b=1.0)
        supply = LinearSupplyCurve(c=50.0, d=1.0)
        with pytest.raises(NoEquilibriumError, match="No positive market equilibrium exists"):
            MarketEquilibrium(demand, supply).solve()

    def test_non_positive_demand_intercept_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            LinearDemandCurve(a=0.0, b=1.0)

    def test_non_positive_supply_slope_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            LinearSupplyCurve(c=-20.0, d=0.0)

    def test_non_positive_elasticity_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            ConstantElasticityDemandCurve(A=100.0, epsilon=0.0)


# --------------------------------------------------------------------------
# 6. MarketEquilibrium の数値解法フォールバック（has_closed_form == False）
# --------------------------------------------------------------------------


class _NumericOnlyDemandCurve(BaseDemandCurve):
    """has_closed_form=False の需要曲線を模した、数値解法フォールバック検証用のテストダブル."""

    def __init__(self, delegate: LinearDemandCurve) -> None:
        self._delegate = delegate

    def get_quantity(self, price: float) -> float:
        return self._delegate.get_quantity(price)

    def get_inverse_price(self, quantity: float) -> float:
        return self._delegate.get_inverse_price(quantity)

    def get_price_elasticity(self, price: float) -> float:
        return self._delegate.get_price_elasticity(price)

    def get_expression(self) -> sympy.Expr:
        return self._delegate.get_expression()

    @property
    def has_closed_form(self) -> bool:
        return False

    @property
    def is_surplus_divergent(self) -> bool:
        return self._delegate.is_surplus_divergent


class TestNumericEquilibriumFallback:
    def test_falls_back_to_scipy_root_scalar(
        self, linear_demand: LinearDemandCurve, linear_supply: LinearSupplyCurve
    ) -> None:
        numeric_demand = _NumericOnlyDemandCurve(linear_demand)
        result = MarketEquilibrium(numeric_demand, linear_supply).solve()

        assert result.price == pytest.approx(30.0, abs=APPROX_ABS)
        assert result.quantity == pytest.approx(40.0, abs=APPROX_ABS)
        assert result.is_closed_form is False
