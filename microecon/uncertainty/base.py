"""不確実性・動的選好モジュール共通の抽象基底クラス."""

from __future__ import annotations

from abc import ABC, abstractmethod

import sympy


class BaseEconomicFunction(ABC):
    """リポジトリ共通の抽象基底クラス.

    :mod:`microecon.uncertainty` 配下のあらゆる関数オブジェクト
    （ベルヌーイ効用関数・割引関数）は本クラスを継承し、
    `evaluate` / `get_expression` / `has_closed_form` の命名規約に準拠する。
    """

    @abstractmethod
    def evaluate(self, *args: object, **kwargs: object) -> float:
        """主たる計算・評価メソッド."""
        raise NotImplementedError

    @abstractmethod
    def get_expression(self) -> sympy.Expr:
        """SymPyによる代数式表現を返す."""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_closed_form(self) -> bool:
        """解析解（Closed-form solution）が存在するか否か."""
        raise NotImplementedError
