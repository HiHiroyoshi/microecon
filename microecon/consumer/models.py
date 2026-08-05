"""消費者理論のドメインモデル（不可変dataclass群）."""

from __future__ import annotations

from dataclasses import dataclass

from microecon.consumer.base import BaseUtilityFunction
from microecon.exceptions import InvalidEconomicParameterError


@dataclass(frozen=True)
class BudgetConstraint:
    """予算制約 P_x * x + P_y * y = M を表す不可変モデル.

    Attributes:
        price_x: 財Xの価格 P_x。正の実数でなければならない。
        price_y: 財Yの価格 P_y。正の実数でなければならない。
        income: 所得 M。正の実数でなければならない。
    """

    price_x: float
    price_y: float
    income: float

    def __post_init__(self) -> None:
        if self.price_x <= 0:
            raise InvalidEconomicParameterError(
                f"price_x must be positive, got {self.price_x}"
            )
        if self.price_y <= 0:
            raise InvalidEconomicParameterError(
                f"price_y must be positive, got {self.price_y}"
            )
        if self.income <= 0:
            raise InvalidEconomicParameterError(
                f"income must be positive, got {self.income}"
            )


@dataclass(frozen=True)
class CobbDouglasUtility(BaseUtilityFunction):
    """コブ＝ダグラス型効用関数 U(x, y) = x^alpha * y^beta.

    Attributes:
        alpha: 財Xに対する選好の強さを表す指数。正の実数でなければならない。
        beta: 財Yに対する選好の強さを表す指数。正の実数でなければならない。
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise InvalidEconomicParameterError(
                f"alpha must be positive, got {self.alpha}"
            )
        if self.beta <= 0:
            raise InvalidEconomicParameterError(
                f"beta must be positive, got {self.beta}"
            )

    def evaluate(self, x: float, y: float) -> float:
        return float((x**self.alpha) * (y**self.beta))

    def calculate_mrs(self, x: float, y: float) -> float:
        """MRS_xy = (alpha * y) / (beta * x) を返す."""
        return float((self.alpha * y) / (self.beta * x))

    def get_symbolic_expression(self) -> str:
        """SymPyで解析可能な文字列 "x**alpha * y**beta" を返す（alpha, betaは小数第4位で丸め）."""
        rounded_alpha = round(self.alpha, 4)
        rounded_beta = round(self.beta, 4)
        return f"x**{rounded_alpha} * y**{rounded_beta}"


@dataclass(frozen=True)
class OptimizationResult:
    """消費者の効用最大化問題の解.

    Attributes:
        optimal_x: 最適消費量 x*。
        optimal_y: 最適消費量 y*。
        optimal_utility: 最適効用水準 U*。
        lambda_: ラグランジュ乗数（所得の限界効用・影の価格）。
        mrs: 最適点における限界代替率 MRS_xy。
        markdown_steps: 解法過程を示すMarkdown/LaTeX形式の解説文字列。
    """

    optimal_x: float
    optimal_y: float
    optimal_utility: float
    lambda_: float
    mrs: float
    markdown_steps: str
