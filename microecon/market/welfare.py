"""市場余剰（消費者余剰・生産者余剰・総余剰・死荷重）の分析."""

from __future__ import annotations

import math

from scipy.integrate import quad

from microecon.exceptions import InvalidEconomicParameterError
from microecon.market.curves import BaseDemandCurve, BaseSupplyCurve
from microecon.market.dataclasses import WelfareResult
from microecon.market.equilibrium import MarketEquilibrium


class WelfareAnalyzer:
    """消費者余剰・生産者余剰・政府収入・総余剰・死荷重を評価するアナライザー.

    `__init__` の時点で :class:`~microecon.market.equilibrium.MarketEquilibrium`
    を用いて歪みのない競争均衡 (P*, Q*) を一度だけ解き、`competitive_price` /
    `competitive_quantity` として内部にキャッシュする。この (P*, Q*) は
    死荷重 (DWL) の算出における不変の参照点として使われ、
    `calculate_welfare` を何度呼び出しても再計算されない。
    """

    def __init__(
        self, demand_curve: BaseDemandCurve, supply_curve: BaseSupplyCurve
    ) -> None:
        self.demand_curve = demand_curve
        self.supply_curve = supply_curve

        equilibrium = MarketEquilibrium(demand_curve, supply_curve).solve()
        self._competitive_price = equilibrium.price
        self._competitive_quantity = equilibrium.quantity

    @property
    def competitive_price(self) -> float:
        """内部キャッシュされた、歪みのない競争均衡価格 P*."""
        return self._competitive_price

    @property
    def competitive_quantity(self) -> float:
        """内部キャッシュされた、歪みのない競争均衡取引量 Q*."""
        return self._competitive_quantity

    def calculate_welfare(
        self, buyer_price: float, seller_price: float, quantity: float
    ) -> WelfareResult:
        """評価取引量 `quantity` における厚生指標を計算する.

        Args:
            buyer_price: 買い手が実際に支払うスカラーの評価価格 P_b。
            seller_price: 売り手が実際に受け取るスカラーの評価価格 P_s。
                これは供給曲線上の任意の数量に対する価格を返す関数
                （逆供給関数 P_s(q)、:meth:`~microecon.market.curves.BaseCurve.get_inverse_price`）
                とは異なり、あくまで評価点における単一のスカラー値である点に
                注意すること。非課税の競争均衡を評価する場合は
                `buyer_price = seller_price = competitive_price` を指定する。
            quantity: 実際の評価取引量 Q（例: 課税後取引量 Q_t）。

        Returns:
            消費者余剰・生産者余剰・政府収入・総余剰・死荷重を含む
            :class:`~microecon.market.dataclasses.WelfareResult`。
        """
        if quantity < 0:
            raise InvalidEconomicParameterError(
                f"quantity must be non-negative, got {quantity}"
            )

        consumer_surplus = self._calculate_consumer_surplus(buyer_price, quantity)
        producer_surplus = self._calculate_producer_surplus(seller_price, quantity)
        government_revenue = (buyer_price - seller_price) * quantity

        if math.isinf(consumer_surplus):
            total_surplus = math.inf
        else:
            total_surplus = consumer_surplus + producer_surplus + government_revenue

        deadweight_loss = self._calculate_deadweight_loss(quantity)

        return WelfareResult(
            consumer_surplus=consumer_surplus,
            producer_surplus=producer_surplus,
            government_revenue=government_revenue,
            total_surplus=total_surplus,
            deadweight_loss=deadweight_loss,
        )

    def _calculate_consumer_surplus(self, buyer_price: float, quantity: float) -> float:
        """CS = int_0^Q (P_d(q) - buyer_price) dq. 発散する需要曲線では inf を返す."""
        if self.demand_curve.is_surplus_divergent:
            return math.inf
        if quantity <= 0:
            return 0.0

        integral, _ = quad(
            lambda q: self.demand_curve.get_inverse_price(q) - buyer_price,
            0.0,
            quantity,
        )
        return float(integral)

    def _calculate_producer_surplus(
        self, seller_price: float, quantity: float
    ) -> float:
        """PS = int_0^Q (seller_price - P_s(q)) dq. eta > 0 のため常に有限値."""
        if quantity <= 0:
            return 0.0

        integral, _ = quad(
            lambda q: seller_price - self.supply_curve.get_inverse_price(q),
            0.0,
            quantity,
        )
        return float(integral)

    def _calculate_deadweight_loss(self, quantity: float) -> float:
        """DWL = |int_{quantity}^{Q*} (P_d(q) - P_s(q)) dq|."""
        q_star = self._competitive_quantity
        if math.isclose(quantity, q_star, rel_tol=1e-9, abs_tol=1e-9):
            return 0.0

        lower, upper = sorted((quantity, q_star))
        integral, _ = quad(
            lambda q: self.demand_curve.get_inverse_price(q)
            - self.supply_curve.get_inverse_price(q),
            lower,
            upper,
        )
        return float(abs(integral))
