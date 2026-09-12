"""従量税課税が市場均衡・厚生・税負担の帰属に与える影響の分析."""

from __future__ import annotations

from scipy.optimize import root_scalar

from microecon.exceptions import InvalidEconomicParameterError, NoEquilibriumError
from microecon.market.curves import BaseDemandCurve, BaseSupplyCurve
from microecon.market.dataclasses import TaxImpactResult
from microecon.market.equilibrium import _find_bracket
from microecon.market.welfare import WelfareAnalyzer


class TaxImpactAnalyzer:
    """単位あたり従量税 t が市場に与える影響（価格帰属・厚生・死荷重）を分析する.

    内部に :class:`~microecon.market.welfare.WelfareAnalyzer` を1つだけ保持し、
    課税後均衡 (P_b, P_s, Q_t) における厚生計算は必ず
    `WelfareAnalyzer.calculate_welfare` へ委譲する（余剰計算ロジックの
    再実装は行わない）。競争均衡 (P*, Q*) も `WelfareAnalyzer` が内部キャッシュ
    している値をそのまま再利用し、二重に均衡を解くことはない。
    """

    def __init__(
        self, demand_curve: BaseDemandCurve, supply_curve: BaseSupplyCurve
    ) -> None:
        self.demand_curve = demand_curve
        self.supply_curve = supply_curve
        self._welfare_analyzer = WelfareAnalyzer(demand_curve, supply_curve)

    def analyze_specific_tax(self, tax: float) -> TaxImpactResult:
        """単位あたり従量税 `tax` (>= 0) を課した場合の市場への影響を分析する.

        課税後均衡条件 P_b - P_s = tax, Q_d(P_b) = Q_s(P_s) = Q_t を満たす
        (P_b, P_s, Q_t) を求め、価格帰属（実効値・弾力性アプローチによる
        予測値）と厚生評価をまとめた :class:`~microecon.market.dataclasses.TaxImpactResult`
        を返す。
        """
        if tax < 0:
            raise InvalidEconomicParameterError(
                f"tax must be non-negative, got {tax}"
            )

        p_star = self._welfare_analyzer.competitive_price
        q_star = self._welfare_analyzer.competitive_quantity

        if tax == 0:
            buyer_price = p_star
            seller_price = p_star
            taxed_quantity = q_star
            buyer_tax_share = 0.0
            seller_tax_share = 0.0
        else:
            seller_price = self._solve_seller_price(tax)
            buyer_price = seller_price + tax
            taxed_quantity = self.demand_curve.get_quantity(buyer_price)
            buyer_tax_share = (buyer_price - p_star) / tax
            seller_tax_share = (p_star - seller_price) / tax

        epsilon = self.demand_curve.get_price_elasticity(p_star)
        eta = self.supply_curve.get_price_elasticity(p_star)
        elasticity_predicted_buyer_share = eta / (epsilon + eta)

        welfare = self._welfare_analyzer.calculate_welfare(
            buyer_price=buyer_price,
            seller_price=seller_price,
            quantity=taxed_quantity,
        )

        return TaxImpactResult(
            tax_rate=tax,
            buyer_price=buyer_price,
            seller_price=seller_price,
            taxed_quantity=taxed_quantity,
            buyer_tax_share=buyer_tax_share,
            seller_tax_share=seller_tax_share,
            elasticity_predicted_buyer_share=elasticity_predicted_buyer_share,
            welfare=welfare,
        )

    def _solve_seller_price(self, tax: float) -> float:
        """Q_d(P_s + tax) = Q_s(P_s) を満たす売り手価格 P_s を数値的に求める."""

        def excess_demand(seller_price: float) -> float:
            return self.demand_curve.get_quantity(
                seller_price + tax
            ) - self.supply_curve.get_quantity(seller_price)

        lower, upper = _find_bracket(excess_demand)
        result = root_scalar(excess_demand, bracket=[lower, upper], method="brentq")
        if not result.converged:
            raise NoEquilibriumError("No positive market equilibrium exists")
        return float(result.root)
