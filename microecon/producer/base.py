"""生産関数の抽象基底クラス."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from microecon.producer.models import CostMinResult, ProfitMaxResult


class BaseProductionFunction(ABC):
    """あらゆる生産関数が実装すべきインターフェース.

    v0.3.1では利潤最大化問題（PMP: Profit Maximization Problem）と
    費用最小化問題（CMP: Cost Minimization Problem）の双方の閉形式解を
    各具象クラスが自ら実装するポリモーフィックな設計を採る。
    :class:`~microecon.producer.solver.ProducerProblem` はこれらのメソッドへの
    委譲（Delegation）のみを行い、生産関数の型ごとの分岐は一切持たない。
    """

    @abstractmethod
    def evaluate(self, labor: float, capital: float) -> float:
        """要素投入 (L, K) に対する産出量 q = f(L, K) を返す."""
        raise NotImplementedError

    @abstractmethod
    def calculate_mrts(self, labor: float, capital: float) -> float:
        """要素投入 (L, K) における技術的限界代替率 MRTS = MP_L / MP_K を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_short_run_labor(self, target_output: float, fixed_capital: float) -> float:
        """短期において資本を `fixed_capital` に固定した際、目標産出量
        `target_output` を達成するために必要な労働量 L(q; K_bar) を返す。
        """
        raise NotImplementedError

    @abstractmethod
    def solve_pmp(self, price: float, wage: float, rental: float) -> ProfitMaxResult:
        """利潤最大化問題 (PMP) を閉形式解で解き、結果を返す.

        規模に関する収穫が一定または逓増（CRS/IRS）の場合、最適解は非有界となる
        ため ValueError を送出する。
        """
        raise NotImplementedError

    @abstractmethod
    def solve_cmp(
        self, target_output: float, wage: float, rental: float
    ) -> CostMinResult:
        """費用最小化問題 (CMP) を閉形式解で解き、結果を返す."""
        raise NotImplementedError

    @abstractmethod
    def calculate_cost_function(
        self, target_output: float, wage: float, rental: float
    ) -> float:
        """総費用関数 C(q, w, r) の数値を返す."""
        raise NotImplementedError

    @abstractmethod
    def calculate_profit_function(
        self, price: float, wage: float, rental: float
    ) -> float:
        """利潤関数 Pi(p, w, r) の数値を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_cost_expression(self) -> sympy.Expr:
        """総費用関数 C(q, w, r) のSymPy記号表現を返す."""
        raise NotImplementedError

    @abstractmethod
    def get_profit_expression(self) -> sympy.Expr:
        """利潤関数 Pi(p, w, r) のSymPy記号表現を返す."""
        raise NotImplementedError
