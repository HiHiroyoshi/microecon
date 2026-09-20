"""時間的選好（動学的選好）を表す割引関数の抽象基底クラスと具象実装."""

from __future__ import annotations

from abc import abstractmethod

import sympy

from microecon.uncertainty.base import BaseEconomicFunction

_PERIOD_SYMBOL = sympy.Symbol("t", nonnegative=True, integer=True)


class BaseDiscountFunction(BaseEconomicFunction):
    """割引関数（時間的選好）が実装すべきインターフェース."""

    @abstractmethod
    def evaluate(self, period: int) -> float:  # type: ignore[override]
        """第t期の割引因子 D(t) を返す."""
        raise NotImplementedError

    def get_discount_factor(self, period: int) -> float:
        """:meth:`evaluate` のエイリアス（後方互換性・可読性のため）."""
        return self.evaluate(period)

    @property
    def has_closed_form(self) -> bool:
        return True

    @property
    @abstractmethod
    def is_time_consistent(self) -> bool:
        """時間整合的（動学的選好の逆転が生じない）か否か."""
        raise NotImplementedError


class ExponentialDiscounting(BaseDiscountFunction):
    """指数割引モデル D(t) = delta^t.

    `delta`（割引因子）または `discount_rate`（割引率 r、delta = 1/(1+r)）の
    いずれか一方のみを指定する。指数割引は時間整合的である。

    Attributes:
        delta: 割引因子。0より大きく1以下でなければならない。
    """

    def __init__(
        self, delta: float | None = None, discount_rate: float | None = None
    ) -> None:
        if delta is None and discount_rate is None:
            raise ValueError("Either delta or discount_rate must be specified")
        if delta is not None and discount_rate is not None:
            raise ValueError("Only one of delta or discount_rate may be specified")

        if discount_rate is not None:
            if discount_rate < 0:
                raise ValueError("discount_rate must be non-negative")
            delta = 1.0 / (1.0 + discount_rate)

        assert delta is not None
        if not (0.0 < delta <= 1.0):
            raise ValueError("delta must satisfy 0 < delta <= 1")
        self.delta = delta

    def evaluate(self, period: int) -> float:  # type: ignore[override]
        return self.delta**period

    def get_expression(self) -> sympy.Expr:
        return sympy.Float(self.delta) ** _PERIOD_SYMBOL

    @property
    def is_time_consistent(self) -> bool:
        return True


class QuasiHyperbolicDiscounting(BaseDiscountFunction):
    """準双曲割引（beta-delta割引）モデル.

    D(0) = 1, D(t) = beta * delta^t (t >= 1)

    beta < 1 の場合、現在バイアス（現時点に近い期間を過度に割り引く）を
    表現し、時間非整合な選好となる。beta == 1 の場合は指数割引に一致し、
    時間整合的となる。

    Attributes:
        beta: 現在バイアス・パラメータ。0より大きく1以下でなければならない。
        delta: 長期割引因子。0より大きく1以下でなければならない。
    """

    def __init__(self, beta: float, delta: float) -> None:
        if not (0.0 < beta <= 1.0):
            raise ValueError("beta must satisfy 0 < beta <= 1")
        if not (0.0 < delta <= 1.0):
            raise ValueError("delta must satisfy 0 < delta <= 1")
        self.beta = beta
        self.delta = delta

    def evaluate(self, period: int) -> float:  # type: ignore[override]
        if period == 0:
            return 1.0
        return self.beta * self.delta**period

    def get_expression(self) -> sympy.Expr:
        return sympy.Piecewise(
            (sympy.Integer(1), sympy.Eq(_PERIOD_SYMBOL, 0)),
            (sympy.Float(self.beta) * sympy.Float(self.delta) ** _PERIOD_SYMBOL, True),
        )

    @property
    def is_time_consistent(self) -> bool:
        return self.beta == 1.0
