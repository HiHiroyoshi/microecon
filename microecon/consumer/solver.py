"""消費者の効用最大化問題・支出最小化問題のソルバー.

v0.2.0では、各効用関数クラス（:class:`~microecon.consumer.base.BaseUtilityFunction`
の具象実装）が自身のUMP（効用最大化問題）・EMP（支出最小化問題）の閉形式解を保持する
ポリモーフィックな設計を採用する。:class:`ConsumerProblem` はそれらへの委譲
（Delegation）のみを行い、効用関数の型に応じた `if isinstance(...)` 分岐は一切持たない。
"""

from __future__ import annotations

from microecon.consumer.base import BaseUtilityFunction
from microecon.consumer.models import (
    BudgetConstraint,
    ExpenditureResult,
    HicksianDecomposition,
    OptimizationResult,
)


class ConsumerProblem:
    """効用最大化問題 max U(x, y) s.t. P_x x + P_y y = M を表現するソルバー."""

    def __init__(self, utility: BaseUtilityFunction, budget: BudgetConstraint) -> None:
        self.utility = utility
        self.budget = budget

    def solve(self) -> OptimizationResult:
        """効用最大化問題 (UMP) を解き、最適解と解説テキストを含む結果を返す.

        実際の閉形式解の算出は `self.utility.solve_ump` へ完全に委譲する。
        """
        return self.utility.solve_ump(self.budget)

    def solve_expenditure_minimization(
        self, target_utility: float
    ) -> ExpenditureResult:
        """現在の価格 (P_x, P_y) の下で、目標効用 `target_utility` を実現する
        支出最小化問題 (EMP) を解く。実際の閉形式解の算出は
        `self.utility.solve_emp` へ完全に委譲する。
        """
        return self.utility.solve_emp(
            target_utility, self.budget.price_x, self.budget.price_y
        )

    def decompose_hicksian(self, new_price_x: float) -> HicksianDecomposition:
        """財Xの価格が `new_price_x` へ変化した際の全効果をヒックス分解する.

        変化前の効用水準 U* = V(P_x, P_y, M) を基準に、価格変化後の全効果
        Δx^total を代替効果 Δx^sub と所得効果 Δx^inc に分解する:

        - 代替効果: Δx^sub = x^h(P_x', P_y, U*) - x*(P_x, P_y, M)
        - 所得効果: Δx^inc = x*(P_x', P_y, M) - x^h(P_x', P_y, U*)
        - 全効果:   Δx^total = Δx^sub + Δx^inc = x*(P_x', P_y, M) - x*(P_x, P_y, M)
        """
        original_result = self.utility.solve_ump(self.budget)
        base_utility = original_result.optimal_utility

        new_budget = BudgetConstraint(
            price_x=new_price_x,
            price_y=self.budget.price_y,
            income=self.budget.income,
        )
        new_result = self.utility.solve_ump(new_budget)

        hicksian_at_new_price = self.utility.solve_emp(
            base_utility, new_price_x, self.budget.price_y
        )

        substitution_effect = (
            hicksian_at_new_price.hicksian_x - original_result.optimal_x
        )
        income_effect = new_result.optimal_x - hicksian_at_new_price.hicksian_x
        total_effect = new_result.optimal_x - original_result.optimal_x

        markdown_steps = self._build_hicksian_markdown(
            base_utility=base_utility,
            original_x=original_result.optimal_x,
            new_x=new_result.optimal_x,
            hicksian_x=hicksian_at_new_price.hicksian_x,
            substitution_effect=substitution_effect,
            income_effect=income_effect,
            total_effect=total_effect,
        )

        return HicksianDecomposition(
            price_x_before=self.budget.price_x,
            price_x_after=new_price_x,
            optimal_x_before=original_result.optimal_x,
            optimal_x_after=new_result.optimal_x,
            hicksian_x_at_new_price=hicksian_at_new_price.hicksian_x,
            substitution_effect=substitution_effect,
            income_effect=income_effect,
            total_effect=total_effect,
            markdown_steps=markdown_steps,
        )

    @staticmethod
    def _build_hicksian_markdown(
        *,
        base_utility: float,
        original_x: float,
        new_x: float,
        hicksian_x: float,
        substitution_effect: float,
        income_effect: float,
        total_effect: float,
    ) -> str:
        sections = [
            "## ヒックス分解（代替効果と所得効果）",
            "",
            "**基準効用:**",
            f"$$U^{{*}} = V(P_x, P_y, M) = {round(base_utility, 4)}$$",
            "",
            "**代替効果 (Substitution Effect):**",
            "$$\\Delta x^{sub} = x^{h}(P_x', P_y, U^{*}) - x^{*}(P_x, P_y, M) = "
            f"{round(hicksian_x, 4)} - {round(original_x, 4)} "
            f"= {round(substitution_effect, 4)}$$",
            "",
            "**所得効果 (Income Effect):**",
            "$$\\Delta x^{inc} = x^{*}(P_x', P_y, M) - x^{h}(P_x', P_y, U^{*}) = "
            f"{round(new_x, 4)} - {round(hicksian_x, 4)} "
            f"= {round(income_effect, 4)}$$",
            "",
            "**全効果 (Total Effect):**",
            "$$\\Delta x^{total} = \\Delta x^{sub} + \\Delta x^{inc} = "
            f"{round(substitution_effect, 4)} + {round(income_effect, 4)} "
            f"= {round(total_effect, 4)}$$",
        ]
        return "\n".join(sections)
