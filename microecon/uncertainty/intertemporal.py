"""割引関数と期間内効用関数を組み合わせた異時点間の消費選択評価."""

from __future__ import annotations

from microecon.uncertainty.dataclasses import DiscountedUtilityResult
from microecon.uncertainty.time_preference import BaseDiscountFunction
from microecon.uncertainty.utility import BaseBernoulliUtility


class IntertemporalChoice:
    """消費ストリーム C = (c_0, c_1, ..., c_T) の通期割引効用
    PV = sum_{t=0}^{T} D(t) * u(c_t) を評価するアナライザー.
    """

    def __init__(
        self,
        discount_function: BaseDiscountFunction,
        period_utility: BaseBernoulliUtility,
    ) -> None:
        self.discount_function = discount_function
        self.period_utility = period_utility

    def evaluate_stream(
        self, consumption_stream: tuple[float, ...]
    ) -> DiscountedUtilityResult:
        """消費ストリームの割引現在価値・各期割引因子・時間整合性を評価する."""
        discount_factors = tuple(
            self.discount_function.evaluate(t)
            for t in range(len(consumption_stream))
        )
        present_value = sum(
            factor * self.period_utility.evaluate(consumption)
            for factor, consumption in zip(discount_factors, consumption_stream)
        )

        return DiscountedUtilityResult(
            present_value=present_value,
            discount_factors=discount_factors,
            is_time_consistent=self.discount_function.is_time_consistent,
        )
