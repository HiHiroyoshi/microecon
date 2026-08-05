"""ドメイン固有のカスタム例外."""

from __future__ import annotations


class InvalidEconomicParameterError(ValueError):
    """経済パラメータ（価格・所得・効用関数の係数等）が定義域外である場合に送出される例外."""
