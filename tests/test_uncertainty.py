"""microecon.uncertainty の単体テスト（不確実性・リスク選好・動的選好分析）."""

from __future__ import annotations

import pytest

from microecon.uncertainty.dataclasses import Lottery, RiskAnalysisResult
from microecon.uncertainty.expected_utility import ExpectedUtilityAnalyzer
from microecon.uncertainty.intertemporal import IntertemporalChoice
from microecon.uncertainty.time_preference import (
    ExponentialDiscounting,
    QuasiHyperbolicDiscounting,
)
from microecon.uncertainty.utility import CARAUtility, CRRAUtility

APPROX_ABS = 1e-4


# --------------------------------------------------------------------------
# 1. CRRA（対数効用 gamma = 1）におけるリスク・プレミアム試算
# --------------------------------------------------------------------------


class TestCRRALogUtilityRiskPremium:
    """gamma = 1 の対数効用における E[w]=250, CE=200, RP=50 の検証."""

    @pytest.fixture
    def lottery(self) -> Lottery:
        return Lottery(outcomes=((0.5, 100.0), (0.5, 400.0)))

    @pytest.fixture
    def result(self, lottery: Lottery) -> RiskAnalysisResult:
        analyzer = ExpectedUtilityAnalyzer(CRRAUtility(gamma=1.0))
        return analyzer.analyze(lottery)

    def test_expected_wealth(self, result: RiskAnalysisResult) -> None:
        assert result.expected_wealth == pytest.approx(250.0, abs=APPROX_ABS)

    def test_expected_utility(self, result: RiskAnalysisResult) -> None:
        assert result.expected_utility == pytest.approx(5.298317, abs=APPROX_ABS)

    def test_certainty_equivalent(self, result: RiskAnalysisResult) -> None:
        assert result.certainty_equivalent == pytest.approx(200.0, abs=APPROX_ABS)

    def test_risk_premium(self, result: RiskAnalysisResult) -> None:
        assert result.risk_premium == pytest.approx(50.0, abs=APPROX_ABS)

    def test_absolute_risk_aversion(self, result: RiskAnalysisResult) -> None:
        assert result.absolute_risk_aversion == pytest.approx(0.004, abs=APPROX_ABS)

    def test_relative_risk_aversion(self, result: RiskAnalysisResult) -> None:
        assert result.relative_risk_aversion == pytest.approx(1.0, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 2. CARA効用関数におけるリスク・プレミアム試算
# --------------------------------------------------------------------------


class TestCARAUtilityRiskPremium:
    """alpha = 0.01 のCARA効用における E[u(w)] ~= -20.8833, CE ~= 156.6219,
    RP ~= 43.3781 の検証."""

    @pytest.fixture
    def lottery(self) -> Lottery:
        return Lottery(outcomes=((0.5, 100.0), (0.5, 300.0)))

    @pytest.fixture
    def analyzer(self) -> ExpectedUtilityAnalyzer:
        return ExpectedUtilityAnalyzer(CARAUtility(alpha=0.01))

    def test_expected_utility(
        self, analyzer: ExpectedUtilityAnalyzer, lottery: Lottery
    ) -> None:
        assert analyzer.calculate_expected_utility(lottery) == pytest.approx(
            -20.8833, abs=APPROX_ABS
        )

    def test_certainty_equivalent(
        self, analyzer: ExpectedUtilityAnalyzer, lottery: Lottery
    ) -> None:
        assert analyzer.calculate_certainty_equivalent(lottery) == pytest.approx(
            156.6219, abs=APPROX_ABS
        )

    def test_risk_premium(
        self, analyzer: ExpectedUtilityAnalyzer, lottery: Lottery
    ) -> None:
        result = analyzer.analyze(lottery)
        assert result.risk_premium == pytest.approx(43.3781, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# 3. beta-delta 準双曲割引における現時点バイアス評価
# --------------------------------------------------------------------------


class TestQuasiHyperbolicPresentBias:
    """beta=0.8, delta=0.95, u(c)=c-1 の消費ストリーム(100,100,100)で
    PV=245.718 かつ時間非整合となることの検証."""

    @pytest.fixture
    def result(self) -> object:
        discounting = QuasiHyperbolicDiscounting(beta=0.8, delta=0.95)
        period_utility = CRRAUtility(gamma=0.0)
        choice = IntertemporalChoice(discounting, period_utility)
        return choice.evaluate_stream((100.0, 100.0, 100.0))

    def test_discount_factors(self, result: object) -> None:
        assert result.discount_factors[0] == pytest.approx(1.0, abs=APPROX_ABS)
        assert result.discount_factors[1] == pytest.approx(0.76, abs=APPROX_ABS)
        assert result.discount_factors[2] == pytest.approx(0.722, abs=APPROX_ABS)

    def test_present_value(self, result: object) -> None:
        assert result.present_value == pytest.approx(245.718, abs=APPROX_ABS)

    def test_is_time_inconsistent(self, result: object) -> None:
        assert result.is_time_consistent is False


# --------------------------------------------------------------------------
# 4. 理論の自己無矛盾性・エッジケース検証
# --------------------------------------------------------------------------


class TestRiskNeutrality:
    """gamma = 0 (u(w) = w - 1) のリスク中立ケースでは CE = E[w], RP = 0 となること."""

    def test_certainty_equivalent_equals_expected_wealth(self) -> None:
        lottery = Lottery(outcomes=((0.5, 100.0), (0.5, 400.0)))
        analyzer = ExpectedUtilityAnalyzer(CRRAUtility(gamma=0.0))
        result = analyzer.analyze(lottery)

        assert result.certainty_equivalent == pytest.approx(250.0, abs=APPROX_ABS)
        assert result.risk_premium == pytest.approx(0.0, abs=APPROX_ABS)


class TestTimeConsistency:
    def test_beta_equal_one_is_time_consistent(self) -> None:
        discounting = QuasiHyperbolicDiscounting(beta=1.0, delta=0.95)
        assert discounting.is_time_consistent is True

    def test_beta_less_than_one_is_time_inconsistent(self) -> None:
        discounting = QuasiHyperbolicDiscounting(beta=0.8, delta=0.95)
        assert discounting.is_time_consistent is False

    def test_exponential_discounting_is_time_consistent(self) -> None:
        discounting = ExponentialDiscounting(delta=0.95)
        assert discounting.is_time_consistent is True


class TestInvalidParameterGuards:
    def test_lottery_probabilities_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError):
            Lottery(outcomes=((0.5, 100.0), (0.4, 400.0)))

    def test_lottery_negative_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            Lottery(outcomes=((-0.5, 100.0), (1.5, 400.0)))

    def test_crra_negative_gamma_raises(self) -> None:
        with pytest.raises(ValueError):
            CRRAUtility(gamma=-1.0)

    def test_crra_non_positive_wealth_raises(self) -> None:
        utility = CRRAUtility(gamma=1.0)
        with pytest.raises(ValueError):
            utility.evaluate(-10.0)

    def test_cara_non_positive_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            CARAUtility(alpha=0.0)

    def test_exponential_discounting_requires_exactly_one_parameter(self) -> None:
        with pytest.raises(ValueError):
            ExponentialDiscounting()
        with pytest.raises(ValueError):
            ExponentialDiscounting(delta=0.9, discount_rate=0.1)


# --------------------------------------------------------------------------
# 5. 補助的な整合性検証
# --------------------------------------------------------------------------


class TestExponentialDiscountingFromRate:
    def test_discount_rate_equivalent_to_delta(self) -> None:
        discounting = ExponentialDiscounting(discount_rate=0.05)
        assert discounting.evaluate(1) == pytest.approx(1.0 / 1.05, abs=APPROX_ABS)

    def test_discount_factor_alias_matches_evaluate(self) -> None:
        discounting = ExponentialDiscounting(delta=0.9)
        assert discounting.get_discount_factor(3) == pytest.approx(
            discounting.evaluate(3), abs=APPROX_ABS
        )


class TestBernoulliUtilityDerivedRiskAversion:
    def test_cara_constant_absolute_risk_aversion(self) -> None:
        utility = CARAUtility(alpha=0.02)
        assert utility.get_ara(50.0) == pytest.approx(0.02, abs=APPROX_ABS)
        assert utility.get_ara(500.0) == pytest.approx(0.02, abs=APPROX_ABS)

    def test_cara_relative_risk_aversion_scales_with_wealth(self) -> None:
        utility = CARAUtility(alpha=0.02)
        assert utility.get_rra(100.0) == pytest.approx(2.0, abs=APPROX_ABS)

    def test_crra_constant_relative_risk_aversion(self) -> None:
        utility = CRRAUtility(gamma=2.0)
        assert utility.get_rra(50.0) == pytest.approx(2.0, abs=APPROX_ABS)
        assert utility.get_rra(500.0) == pytest.approx(2.0, abs=APPROX_ABS)
