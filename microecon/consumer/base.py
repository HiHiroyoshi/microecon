"""効用関数の抽象基底クラス."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from microecon.consumer.models import (
        BudgetConstraint,
        ExpenditureResult,
        OptimizationResult,
    )


class BaseUtilityFunction(ABC):
    """あらゆる効用関数が実装すべきインターフェース.

    v0.2.0では主問題（UMP: 効用最大化問題）と双対問題（EMP: 支出最小化問題）の
    双方の閉形式解を各具象クラスが自ら実装するポリモーフィックな設計を採る。
    :class:`~microecon.consumer.solver.ConsumerProblem` はこれらのメソッドへの
    委譲（Delegation）のみを行い、効用関数の型ごとの分岐は一切持たない。
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

    @abstractmethod
    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        """効用最大化問題 (UMP) を閉形式解で解き、結果を返す.

        実装は自身の :meth:`solve_emp` を用いて双対問題（EMP）の解も統合し、
        `OptimizationResult` に主・双対双方の解を格納すること。
        """
        raise NotImplementedError

    @abstractmethod
    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        """支出最小化問題 (EMP) を閉形式解で解き、結果を返す."""
        raise NotImplementedError

    @abstractmethod
    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        """間接効用関数 V(P_x, P_y, M) の数値を返す."""
        raise NotImplementedError

    @abstractmethod
    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        """支出関数 E(P_x, P_y, U) の数値を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_indirect_utility_expression(self) -> sympy.Expr:
        """間接効用関数 V(P_x, P_y, M) のSymPy記号表現を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_expenditure_expression(self) -> sympy.Expr:
        """支出関数 E(P_x, P_y, U) のSymPy記号表現を返す."""
        raise NotImplementedError
