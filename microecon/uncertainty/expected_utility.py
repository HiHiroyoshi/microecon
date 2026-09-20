"""期待効用理論に基づくリスク指標の分析器."""

from __future__ import annotations

from scipy.optimize import root_scalar

from microecon.uncertainty.dataclasses import Lottery, RiskAnalysisResult
from microecon.uncertainty.utility import BaseBernoulliUtility


class ExpectedUtilityAnalyzer:
    """くじ :class:`~microecon.uncertainty.dataclasses.Lottery` を
    ベルヌーイ効用関数 :class:`~microecon.uncertainty.utility.BaseBernoulliUtility`
    で評価し、期待効用・確実性等価・リスク・プレミアム等を算出するアナライザー.

    確実性等価の導出は、効用関数が `has_closed_form == True` の場合は
    :meth:`~microecon.uncertainty.utility.BaseBernoulliUtility.get_inverse_utility`
    による解析解を用い、それ以外の場合のみ `scipy.optimize.root_scalar` による
    数値解法にフォールバックする。
    """

    def __init__(self, utility_function: BaseBernoulliUtility) -> None:
        self.utility_function = utility_function

    def calculate_expected_utility(self, lottery: Lottery) -> float:
        """期待効用 E[u(w)] = sum_i p_i * u(w_i) を返す."""
        return sum(
            p * self.utility_function.evaluate(w) for p, w in lottery.outcomes
        )

    def calculate_certainty_equivalent(self, lottery: Lottery) -> float:
        """確実性等価 CE = u^{-1}(E[u(w)]) を返す."""
        expected_utility = self.calculate_expected_utility(lottery)
        if self.utility_function.has_closed_form:
            return self.utility_function.get_inverse_utility(expected_utility)
        return self._solve_certainty_equivalent_numerically(expected_utility, lottery)

    def _solve_certainty_equivalent_numerically(
        self, expected_utility: float, lottery: Lottery
    ) -> float:
        outcome_wealth = [w for _, w in lottery.outcomes]
        lower, upper = min(outcome_wealth), max(outcome_wealth)

        def excess_utility(wealth: float) -> float:
            return self.utility_function.evaluate(wealth) - expected_utility

        result = root_scalar(
            excess_utility, bracket=[lower, upper], method="brentq"
        )
        return float(result.root)

    def analyze(self, lottery: Lottery) -> RiskAnalysisResult:
        """期待所得・期待効用・確実性等価・リスク・プレミアム・
        リスク回避度 (ARA/RRA、E[w] における評価値) を網羅的に算出する."""
        expected_wealth = sum(p * w for p, w in lottery.outcomes)
        expected_utility = self.calculate_expected_utility(lottery)
        certainty_equivalent = self.calculate_certainty_equivalent(lottery)
        risk_premium = expected_wealth - certainty_equivalent

        return RiskAnalysisResult(
            expected_wealth=expected_wealth,
            expected_utility=expected_utility,
            certainty_equivalent=certainty_equivalent,
            risk_premium=risk_premium,
            absolute_risk_aversion=self.utility_function.get_ara(expected_wealth),
            relative_risk_aversion=self.utility_function.get_rra(expected_wealth),
        )
