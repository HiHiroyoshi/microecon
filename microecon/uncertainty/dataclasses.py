"""期待効用分析・動的選好分析の結果を保持する不可変dataclass群."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lottery:
    """くじ（確率変数）を表す不変データクラス.

    Attributes:
        outcomes: (確率 p_i, 状態別所得/財貨 w_i) のタプル
            ((p_1, w_1), (p_2, w_2), ...)。確率の合計は1.0でなければならない。
    """

    outcomes: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        total_p = sum(p for p, _ in self.outcomes)
        if not (0.999999 <= total_p <= 1.000001):
            raise ValueError(f"Sum of probabilities must equal 1.0, got {total_p}")
        for p, _ in self.outcomes:
            if p < 0:
                raise ValueError(f"Probability must be non-negative, got {p}")


@dataclass(frozen=True)
class RiskAnalysisResult:
    """期待効用理論に基づくリスク指標の評価結果.

    Attributes:
        expected_wealth: 期待所得 E[w]。
        expected_utility: 期待効用 E[u(w)]。
        certainty_equivalent: 確実性等価 CE（u(CE) = E[u(w)] を満たす）。
        risk_premium: リスク・プレミアム RP = E[w] - CE。
        absolute_risk_aversion: 絶対的リスク回避度 ARA = -u''(w)/u'(w)
            （E[w] における評価値）。
        relative_risk_aversion: 相対的リスク回避度 RRA = -w * u''(w)/u'(w)
            （E[w] における評価値）。
    """

    expected_wealth: float
    expected_utility: float
    certainty_equivalent: float
    risk_premium: float
    absolute_risk_aversion: float
    relative_risk_aversion: float


@dataclass(frozen=True)
class DiscountedUtilityResult:
    """異時点間の消費計画の割引効用評価結果.

    Attributes:
        present_value: 割引現在価値 PV = sum_t D(t) u(c_t)。
        discount_factors: 各期の割引因子 (D(0), D(1), ..., D(T))。
        is_time_consistent: 時間整合性を満たすか否か。
    """

    present_value: float
    discount_factors: tuple[float, ...]
    is_time_consistent: bool
