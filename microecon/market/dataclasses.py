"""市場均衡・厚生分析・課税影響分析の結果を保持する不可変dataclass群."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquilibriumResult:
    """競争均衡の解.

    Attributes:
        price: 均衡価格 P*。
        quantity: 均衡取引量 Q*。
        is_closed_form: 需要・供給曲線の双方が閉形式解に対応しており、
            代数的（SymPyによる記号計算）に解が求まった場合は True。
            いずれか一方でも対応しない場合は scipy.optimize.root_scalar
            による数値解法にフォールバックしており、False となる。
    """

    price: float
    quantity: float
    is_closed_form: bool


@dataclass(frozen=True)
class WelfareResult:
    """消費者余剰・生産者余剰・政府収入・総余剰・死荷重の評価結果.

    Attributes:
        consumer_surplus: 消費者余剰 CS。需要曲線が
            :attr:`~microecon.market.curves.BaseDemandCurve.is_surplus_divergent`
            を満たす場合（弾力性一定型で価格弾力性 epsilon <= 1 のとき）は
            float("inf") となる。
        producer_surplus: 生産者余剰 PS。供給側は価格弾力性 eta > 0 であれば
            数量 0 において価格が 0 に収束するため、常に有限値となる。
        government_revenue: 政府収入 GR = (buyer_price - seller_price) * quantity。
            課税を伴わない競争均衡の評価では buyer_price == seller_price のため 0。
        total_surplus: 総余剰 TS = CS + PS + GR。consumer_surplus が無限大の場合は
            float("inf")。
        deadweight_loss: 死荷重 DWL。評価取引量と、内部に保持する歪みのない
            競争均衡取引量 Q* との間の価格乖離を積分した値。
    """

    consumer_surplus: float
    producer_surplus: float
    government_revenue: float
    total_surplus: float
    deadweight_loss: float


@dataclass(frozen=True)
class TaxImpactResult:
    """従量税課税による市場への影響の分析結果.

    Attributes:
        tax_rate: 単位あたり従量税 t。
        buyer_price: 課税後に買い手が支払う価格 P_b（= 売り手価格 + t）。
        seller_price: 課税後に売り手が受け取る価格 P_s（= 買い手価格 - t）。
        taxed_quantity: 課税後の均衡取引量 Q_t。
        buyer_tax_share: 買い手の実効負担割合 (P_b - P*) / t。
        seller_tax_share: 売り手の実効負担割合 (P* - P_s) / t。
        elasticity_predicted_buyer_share: 競争均衡点 (P*, Q*) における局所弾力性
            eta / (epsilon + eta) から予測される買い手負担割合。弾力性一定型の
            需要・供給曲線の組み合わせでは弾力性が価格に依存せず一定であるため
            この予測値は実効値と厳密に一致するが、線形曲線等、弾力性が価格に
            依存する曲線の組み合わせでは、この式は t -> 0 の極限における
            限界的な負担割合の近似値であり、有限な t に対する実効値とは
            一般に一致しない（本モジュールの数値アンカーテストでは対称的な
            線形市場を用いているため偶然一致する）。
        welfare: 課税後均衡 (P_b, P_s, Q_t) における厚生評価結果。
            :class:`~microecon.market.welfare.WelfareAnalyzer.calculate_welfare`
            への委譲によって構築される。
    """

    tax_rate: float
    buyer_price: float
    seller_price: float
    taxed_quantity: float
    buyer_tax_share: float
    seller_tax_share: float
    elasticity_predicted_buyer_share: float
    welfare: WelfareResult
