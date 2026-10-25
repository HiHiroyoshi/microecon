"""産業組織論モジュールの結果を保持する不変dataclass群.

フィールドは float / int / str / bool / tuple / None のみで構成するため、
``MappingProxyType`` や ``writeable=False`` による読み取り専用化は不要である。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MonopolyResult:
    """単一価格独占または第1種価格差別の結果.

    Attributes:
        regime: ``"single_price"``（単一価格）または ``"first_degree"``（第1種価格差別）。
        price: 単一価格では P_m、第1種では最後の1単位の価格 P(Q_c)。
        quantity: 生産量。
        competitive_price: P = MC となる競争均衡価格 P_c。
        competitive_quantity: 競争均衡数量 Q_c。
        consumer_surplus: 消費者余剰。
        producer_surplus: 生産者余剰（収入 − 可変費用）。
        profit: 利潤（収入 − 総費用）。
        social_welfare: 社会厚生 W = CS + PS。
        competitive_welfare: 競争均衡の社会厚生 W_c。
        deadweight_loss: 死荷重 W_c − W（非負）。
        lerner_index: ラーナー指数 (P_m − MC(Q_m)) / P_m。第1種では ``None``。
        markdown_steps: 導出過程の Markdown/LaTeX 解説。
    """

    regime: Literal["single_price", "first_degree"]
    price: float
    quantity: float
    competitive_price: float
    competitive_quantity: float
    consumer_surplus: float
    producer_surplus: float
    profit: float
    social_welfare: float
    competitive_welfare: float
    deadweight_loss: float
    lerner_index: float | None
    markdown_steps: str


@dataclass(frozen=True)
class ThirdDegreeResult:
    """第3種価格差別の結果（費用は総生産量の関数で、市場間は非分離）."""

    prices: tuple[float, ...]
    quantities: tuple[float, ...]
    consumer_surpluses: tuple[float, ...]
    lerner_indices: tuple[float, ...]
    marginal_cost: float
    total_quantity: float
    consumer_surplus: float
    producer_surplus: float
    profit: float
    social_welfare: float
    competitive_welfare: float
    deadweight_loss: float
    markdown_steps: str


OligopolyModel = Literal[
    "cournot", "stackelberg", "bertrand_differentiated", "bertrand_homogeneous"
]


@dataclass(frozen=True)
class OligopolyResult:
    """2社寡占の均衡結果.

    ``prices`` の格納規則:

    - クールノー・シュタッケルベルク: ``(P*, P*)``、``market_price = P*``
    - 同質財ベルトラン: 各社が設定する価格。高費用企業は ``p_j = c_j``
    - 差別化財ベルトラン: ``(p1*, p2*)``、``market_price = None``
    """

    model_type: OligopolyModel
    prices: tuple[float, float]
    market_price: float | None
    quantities: tuple[float, float]
    profits: tuple[float, float]
    consumer_surplus: float | None
    producer_surplus: float
    social_welfare: float | None
    deadweight_loss: float | None
    is_corner: bool
    welfare_note: str
    markdown_steps: str


MarketStructure = Literal[
    "perfect_competition", "bertrand_homogeneous", "stackelberg", "cournot", "monopoly"
]

_STRUCTURE_NAMES_JA: dict[str, str] = {
    "perfect_competition": "完全競争",
    "bertrand_homogeneous": "ベルトラン",
    "stackelberg": "シュタッケルベルク",
    "cournot": "クールノー",
    "monopoly": "独占",
}


@dataclass(frozen=True)
class MarketStructureRow:
    """市場構造比較表の1行."""

    structure: MarketStructure
    quantity: float
    price: float
    consumer_surplus: float
    producer_surplus: float
    deadweight_loss: float


@dataclass(frozen=True)
class MarketStructureComparison:
    """市場構造比較表（行は総生産量の降順）."""

    rows: tuple[MarketStructureRow, ...]

    def to_markdown_table(self) -> str:
        """日本語の構造名（完全競争・ベルトラン・シュタッケルベルク・クールノー・独占）で表を返す."""
        lines = [
            "| 市場構造 | 総生産量 Q | 価格 P | 消費者余剰 | 生産者余剰 | 死荷重 |",
            "|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            lines.append(
                f"| {_STRUCTURE_NAMES_JA[row.structure]} | {_fmt(row.quantity)} | "
                f"{_fmt(row.price)} | {_fmt(row.consumer_surplus)} | "
                f"{_fmt(row.producer_surplus)} | {_fmt(row.deadweight_loss)} |"
            )
        return "\n".join(lines)


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")
