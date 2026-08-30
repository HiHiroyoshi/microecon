"""生産者の利潤最大化問題・費用最小化問題のソルバー.

v0.3.1では、各生産関数クラス（:class:`~microecon.producer.base.BaseProductionFunction`
の具象実装）が自身のPMP（利潤最大化問題）・CMP（費用最小化問題）の閉形式解を保持する
ポリモーフィックな設計を採用する。:class:`ProducerProblem` はそれらへの委譲
（Delegation）のみを行い、生産関数の型に応じた `if isinstance(...)` 分岐は一切持たない。
"""

from __future__ import annotations

from microecon.producer.base import BaseProductionFunction
from microecon.producer.models import CostMinResult, ProfitMaxResult


class ProducerProblem:
    """利潤最大化問題 max Pi = p f(L, K) - w L - r K を表現するソルバー."""

    def __init__(self, production: BaseProductionFunction) -> None:
        self.production = production

    def solve_profit_maximization(
        self, price: float, wage: float, rental: float
    ) -> ProfitMaxResult:
        """利潤最大化問題 (PMP) を解き、最適解と解説テキストを含む結果を返す.

        実際の閉形式解の算出は `self.production.solve_pmp` へ完全に委譲する。
        """
        return self.production.solve_pmp(price, wage, rental)

    def solve_cost_minimization(
        self, target_output: float, wage: float, rental: float
    ) -> CostMinResult:
        """費用最小化問題 (CMP) を解き、最適解と解説テキストを含む結果を返す.

        実際の閉形式解の算出は `self.production.solve_cmp` へ完全に委譲する。
        """
        return self.production.solve_cmp(target_output, wage, rental)
