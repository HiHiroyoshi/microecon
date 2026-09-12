"""ドメイン固有のカスタム例外."""

from __future__ import annotations


class InvalidEconomicParameterError(ValueError):
    """経済パラメータ（価格・所得・効用関数の係数等）が定義域外である場合に送出される例外."""


class NoEquilibriumError(ValueError):
    """需要曲線と供給曲線が正の価格・数量の領域で交点（市場均衡）を持たない場合に送出される例外."""
