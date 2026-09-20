"""ベルヌーイ効用関数（リスク選好の表現）の抽象基底クラスと具象実装."""

from __future__ import annotations

import math
from abc import abstractmethod

import sympy

from microecon.uncertainty.base import BaseEconomicFunction

_WEALTH_SYMBOL = sympy.Symbol("w", positive=True)


class BaseBernoulliUtility(BaseEconomicFunction):
    """フォン・ノイマン=モルゲンシュテルン型ベルヌーイ効用関数のインターフェース.

    絶対的・相対的リスク回避度 (:meth:`get_ara`, :meth:`get_rra`) は
    Arrow-Prattの定義 A(w) = -u''(w)/u'(w), R(w) = w * A(w) に従い、
    一階・二階導関数 (:meth:`get_first_derivative`, :meth:`get_second_derivative`)
    への委譲のみで一般に算出できるため、具象クラスごとの再実装を要しない。
    """

    @abstractmethod
    def evaluate(self, wealth: float) -> float:  # type: ignore[override]
        """所得（富）`wealth` に対する効用水準 u(w) を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_first_derivative(self, wealth: float) -> float:
        """限界効用 u'(w) を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_second_derivative(self, wealth: float) -> float:
        """限界効用の変化率 u''(w) を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_inverse_utility(self, utility_value: float) -> float:
        """効用の逆関数 u^{-1}(U) を返す（確実性等価の導出に用いる）."""
        raise NotImplementedError

    def get_ara(self, wealth: float) -> float:
        """絶対的リスク回避度 A(w) = -u''(w) / u'(w) を返す."""
        return -self.get_second_derivative(wealth) / self.get_first_derivative(wealth)

    def get_rra(self, wealth: float) -> float:
        """相対的リスク回避度 R(w) = -w * u''(w) / u'(w) = w * A(w) を返す."""
        return wealth * self.get_ara(wealth)


class CRRAUtility(BaseBernoulliUtility):
    """相対的リスク回避度一定 (CRRA) 効用関数.

    u(w) = (w^(1-gamma) - 1) / (1 - gamma)  (gamma >= 0, gamma != 1)
    u(w) = ln(w)                            (gamma == 1)

    gamma = 0 のリスク中立ケース (u(w) = w - 1) も上式の特殊ケースとして
    自然に許容される。RRA = gamma（一定）、ARA = gamma / w。

    Attributes:
        gamma: 相対的リスク回避度 RRA。0以上の実数でなければならない。
    """

    def __init__(self, gamma: float) -> None:
        if gamma < 0:
            raise ValueError("Gamma (RRA) must be non-negative")
        self.gamma = gamma

    def _validate_wealth(self, wealth: float) -> None:
        if wealth <= 0:
            raise ValueError("Wealth must be strictly positive for CRRA utility")

    def evaluate(self, wealth: float) -> float:  # type: ignore[override]
        self._validate_wealth(wealth)
        if self.gamma == 1.0:
            return math.log(wealth)
        return float((wealth ** (1.0 - self.gamma) - 1.0) / (1.0 - self.gamma))

    def get_first_derivative(self, wealth: float) -> float:
        self._validate_wealth(wealth)
        return float(wealth ** (-self.gamma))

    def get_second_derivative(self, wealth: float) -> float:
        self._validate_wealth(wealth)
        return float(-self.gamma * wealth ** (-self.gamma - 1.0))

    def get_inverse_utility(self, utility_value: float) -> float:
        if self.gamma == 1.0:
            return math.exp(utility_value)
        return float(
            (utility_value * (1.0 - self.gamma) + 1.0) ** (1.0 / (1.0 - self.gamma))
        )

    def get_expression(self) -> sympy.Expr:
        if self.gamma == 1.0:
            return sympy.log(_WEALTH_SYMBOL)
        gamma = sympy.Float(self.gamma)
        return (_WEALTH_SYMBOL ** (1 - gamma) - 1) / (1 - gamma)

    @property
    def has_closed_form(self) -> bool:
        return True


class CARAUtility(BaseBernoulliUtility):
    """絶対的リスク回避度一定 (CARA) 効用関数.

    u(w) = -exp(-alpha * w) / alpha  (alpha > 0)

    ARA = alpha（一定）、RRA = alpha * w。

    Attributes:
        alpha: 絶対的リスク回避度 ARA。正の実数でなければならない。
    """

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("Alpha (ARA) must be strictly positive")
        self.alpha = alpha

    def evaluate(self, wealth: float) -> float:  # type: ignore[override]
        return -math.exp(-self.alpha * wealth) / self.alpha

    def get_first_derivative(self, wealth: float) -> float:
        return math.exp(-self.alpha * wealth)

    def get_second_derivative(self, wealth: float) -> float:
        return -self.alpha * math.exp(-self.alpha * wealth)

    def get_inverse_utility(self, utility_value: float) -> float:
        return -math.log(-self.alpha * utility_value) / self.alpha

    def get_expression(self) -> sympy.Expr:
        alpha = sympy.Float(self.alpha)
        w = sympy.Symbol("w", real=True)
        return -sympy.exp(-alpha * w) / alpha

    @property
    def has_closed_form(self) -> bool:
        return True
