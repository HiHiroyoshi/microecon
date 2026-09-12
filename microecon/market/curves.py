"""需要曲線・供給曲線の抽象基底クラスと具象実装.

v0.4.0では、需要側 :class:`BaseDemandCurve` と供給側 :class:`BaseSupplyCurve` が
それぞれ「数量→価格」の順方向 (:meth:`~BaseCurve.get_quantity`) と
「価格→数量」の逆需要・逆供給関数 (:meth:`~BaseCurve.get_inverse_price`) の
双方を自ら実装するポリモーフィックな設計を採る。
:class:`~microecon.market.equilibrium.MarketEquilibrium` や
:class:`~microecon.market.welfare.WelfareAnalyzer` はこれらのメソッドへの
委譲のみを行い、曲線の型ごとの `isinstance` 分岐は一切持たない
（均衡は :meth:`~BaseCurve.get_expression` が返すSymPy記号式の代数解法に、
余剰積分は :meth:`~BaseCurve.get_inverse_price` の数値評価にそれぞれ委譲する）。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sympy

from microecon.exceptions import InvalidEconomicParameterError


class BaseCurve(ABC):
    """需要曲線・供給曲線に共通するインターフェース."""

    @abstractmethod
    def get_quantity(self, price: float) -> float:
        """価格 `price` に対応する数量 Q(P) を返す（順方向の需要・供給関数）."""
        raise NotImplementedError

    @abstractmethod
    def get_inverse_price(self, quantity: float) -> float:
        """数量 `quantity` に対応する価格 P(Q) を返す（逆需要・逆供給関数）.

        この戻り値は、ある数量水準における曲線上の価格という「関数としての」
        P(Q) であり、:class:`~microecon.market.welfare.WelfareAnalyzer` が
        引数として受け取る評価点のスカラー価格（例えば売り手が実際に受け取る
        単一の価格 seller_price）とは異なる概念である点に注意すること。
        """
        raise NotImplementedError

    @abstractmethod
    def get_price_elasticity(self, price: float) -> float:
        """価格 `price` における価格弾力性（絶対値、0以上）を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_expression(self) -> sympy.Expr:
        """順方向の需要・供給関数 Q(P) のSymPy記号表現を返す.

        価格を表すシンボルは全ての具象クラスで共通の `sympy.Symbol("P", positive=True)`
        を用いる。これにより
        :class:`~microecon.market.equilibrium.MarketEquilibrium` は需要側・
        供給側の型を問わず `Eq(demand.get_expression(), supply.get_expression())`
        という単一の代数方程式として均衡条件を組み立てられる。
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def has_closed_form(self) -> bool:
        """閉形式（代数）解に対応しているか否か."""
        raise NotImplementedError


class BaseDemandCurve(BaseCurve):
    """需要曲線が実装すべきインターフェース."""

    @property
    @abstractmethod
    def is_surplus_divergent(self) -> bool:
        """消費者余剰の積分 (Q -> 0 における逆需要関数の発散) が発散するか否か.

        供給側は価格弾力性 eta > 0 であれば数量 0 で価格が 0 に収束するため
        生産者余剰は常に有限値となり、この性質は需要側にのみ定義される。
        """
        raise NotImplementedError


class BaseSupplyCurve(BaseCurve):
    """供給曲線が実装すべきインターフェース."""


_PRICE_SYMBOL = sympy.Symbol("P", positive=True)


@dataclass(frozen=True)
class LinearDemandCurve(BaseDemandCurve):
    """線形需要曲線 Q_d(P) = a - b P (a > 0, b > 0).

    逆需要関数は P_d(Q) = (a - Q) / b（予約価格 P_max = a / b）。

    Attributes:
        a: 需要の切片。正の実数でなければならない。
        b: 需要の価格に対する感応度。正の実数でなければならない。
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")

    def get_quantity(self, price: float) -> float:
        clamped_price = max(price, 0.0)
        return max(self.a - self.b * clamped_price, 0.0)

    def get_inverse_price(self, quantity: float) -> float:
        clamped_quantity = max(quantity, 0.0)
        return max((self.a - clamped_quantity) / self.b, 0.0)

    def get_price_elasticity(self, price: float) -> float:
        if price <= 0:
            raise InvalidEconomicParameterError(
                f"price must be positive, got {price}"
            )
        quantity = self.get_quantity(price)
        if quantity <= 0:
            return math.inf
        return abs(self.b * price / quantity)

    def get_expression(self) -> sympy.Expr:
        return self.a - self.b * _PRICE_SYMBOL

    @property
    def has_closed_form(self) -> bool:
        return True

    @property
    def is_surplus_divergent(self) -> bool:
        return False


@dataclass(frozen=True)
class LinearSupplyCurve(BaseSupplyCurve):
    """線形供給曲線 Q_s(P) = c + d P (d > 0).

    逆供給関数は P_s(Q) = (Q - c) / d
    （最低供給価格 P_min = max(0, -c / d)。c < 0 は「価格が -c/d を
    上回るまで供給が生じない」シャットダウン価格を表す）。

    Attributes:
        c: 供給の切片。負の値も許容する（シャットダウン価格を表現するため）。
        d: 供給の価格に対する感応度。正の実数でなければならない。
    """

    c: float
    d: float

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise InvalidEconomicParameterError(f"d must be positive, got {self.d}")

    def get_quantity(self, price: float) -> float:
        clamped_price = max(price, 0.0)
        return max(self.c + self.d * clamped_price, 0.0)

    def get_inverse_price(self, quantity: float) -> float:
        clamped_quantity = max(quantity, 0.0)
        return max((clamped_quantity - self.c) / self.d, 0.0)

    def get_price_elasticity(self, price: float) -> float:
        if price <= 0:
            raise InvalidEconomicParameterError(
                f"price must be positive, got {price}"
            )
        quantity = self.get_quantity(price)
        if quantity <= 0:
            return math.inf
        return abs(self.d * price / quantity)

    def get_expression(self) -> sympy.Expr:
        return self.c + self.d * _PRICE_SYMBOL

    @property
    def has_closed_form(self) -> bool:
        return True


@dataclass(frozen=True)
class ConstantElasticityDemandCurve(BaseDemandCurve):
    """弾力性一定型需要曲線 Q_d(P) = A P^(-epsilon) (A > 0, epsilon > 0).

    逆需要関数は P_d(Q) = (A / Q)^(1/epsilon)。

    Attributes:
        A: 規模パラメータ。正の実数でなければならない。
        epsilon: 需要の価格弾力性（価格に依存せず一定）。正の実数でなければならない。
    """

    A: float
    epsilon: float

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise InvalidEconomicParameterError(f"A must be positive, got {self.A}")
        if self.epsilon <= 0:
            raise InvalidEconomicParameterError(
                f"epsilon must be positive, got {self.epsilon}"
            )

    def get_quantity(self, price: float) -> float:
        if price <= 0:
            raise InvalidEconomicParameterError(
                f"price must be positive, got {price}"
            )
        return float(self.A * price ** (-self.epsilon))

    def get_inverse_price(self, quantity: float) -> float:
        if quantity <= 0:
            return math.inf
        return float((self.A / quantity) ** (1.0 / self.epsilon))

    def get_price_elasticity(self, price: float) -> float:
        if price <= 0:
            raise InvalidEconomicParameterError(
                f"price must be positive, got {price}"
            )
        return self.epsilon

    def get_expression(self) -> sympy.Expr:
        return self.A * _PRICE_SYMBOL ** (-self.epsilon)

    @property
    def has_closed_form(self) -> bool:
        return True

    @property
    def is_surplus_divergent(self) -> bool:
        """epsilon <= 1 のとき積分 int_0^{Q*} Q^(-1/epsilon) dQ が Q -> 0 で発散する."""
        return self.epsilon <= 1.0


@dataclass(frozen=True)
class ConstantElasticitySupplyCurve(BaseSupplyCurve):
    """弾力性一定型供給曲線 Q_s(P) = B P^eta (B > 0, eta > 0).

    逆供給関数は P_s(Q) = (Q / B)^(1/eta)。

    Attributes:
        B: 規模パラメータ。正の実数でなければならない。
        eta: 供給の価格弾力性（価格に依存せず一定）。正の実数でなければならない。
    """

    B: float
    eta: float

    def __post_init__(self) -> None:
        if self.B <= 0:
            raise InvalidEconomicParameterError(f"B must be positive, got {self.B}")
        if self.eta <= 0:
            raise InvalidEconomicParameterError(
                f"eta must be positive, got {self.eta}"
            )

    def get_quantity(self, price: float) -> float:
        if price <= 0:
            raise InvalidEconomicParameterError(
                f"price must be positive, got {price}"
            )
        return float(self.B * price**self.eta)

    def get_inverse_price(self, quantity: float) -> float:
        clamped_quantity = max(quantity, 0.0)
        return float((clamped_quantity / self.B) ** (1.0 / self.eta))

    def get_price_elasticity(self, price: float) -> float:
        if price <= 0:
            raise InvalidEconomicParameterError(
                f"price must be positive, got {price}"
            )
        return self.eta

    def get_expression(self) -> sympy.Expr:
        return self.B * _PRICE_SYMBOL**self.eta

    @property
    def has_closed_form(self) -> bool:
        return True
