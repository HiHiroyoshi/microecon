"""効用関数の抽象基底クラス."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseUtilityFunction(ABC):
    """あらゆる効用関数が実装すべきインターフェース.

    将来的にCES型・準線形型などへ拡張することを見据えた抽象基底クラス。
    v0.1.0では :class:`~microecon.consumer.models.CobbDouglasUtility` のみが
    :class:`~microecon.consumer.solver.ConsumerProblem` での数値解法に対応する。
    """

    @abstractmethod
    def evaluate(self, x: float, y: float) -> float:
        """財の組 (x, y) に対する効用水準 U(x, y) を返す."""
        raise NotImplementedError

    @abstractmethod
    def calculate_mrs(self, x: float, y: float) -> float:
        """財の組 (x, y) における限界代替率 MRS_xy を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_symbolic_expression(self) -> str:
        """SymPyで解析可能な効用関数の記号表現文字列を返す."""
        raise NotImplementedError
