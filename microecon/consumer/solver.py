"""消費者の効用最大化問題のソルバー.

数値解（最適消費量・効用・ラグランジュ乗数・MRS）は、コブ＝ダグラス型効用関数の
閉形式解（Closed-form solution）を用いて決定論的に算出する。SymPyはあくまで
途中式（ラグランジアン・一階の条件・微分過程）をMarkdown/LaTeX形式のテキストとして
生成するためだけに用い、数値計算ロジックそのものには利用しない。
"""

from __future__ import annotations

import sympy

from microecon.consumer.base import BaseUtilityFunction
from microecon.consumer.models import (
    BudgetConstraint,
    CobbDouglasUtility,
    OptimizationResult,
)


class ConsumerProblem:
    """効用最大化問題 max U(x, y) s.t. P_x x + P_y y = M を表現するソルバー."""

    def __init__(self, utility: BaseUtilityFunction, budget: BudgetConstraint) -> None:
        self.utility = utility
        self.budget = budget

    def solve(self) -> OptimizationResult:
        """効用最大化問題を解き、最適解と解説テキストを含む結果を返す.

        Raises:
            TypeError: `utility` が :class:`CobbDouglasUtility` 以外の場合。
        """
        if not isinstance(self.utility, CobbDouglasUtility):
            raise TypeError(
                "ConsumerProblem.solve() currently supports only "
                f"CobbDouglasUtility, got {type(self.utility).__name__}"
            )

        alpha = self.utility.alpha
        beta = self.utility.beta
        price_x = self.budget.price_x
        price_y = self.budget.price_y
        income = self.budget.income

        optimal_x = (alpha / (alpha + beta)) * (income / price_x)
        optimal_y = (beta / (alpha + beta)) * (income / price_y)
        optimal_utility = (optimal_x**alpha) * (optimal_y**beta)
        lambda_star = ((alpha + beta) * optimal_utility) / income
        mrs = self.utility.calculate_mrs(optimal_x, optimal_y)

        markdown_steps = self._build_markdown_steps(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_star=lambda_star,
        )

        return OptimizationResult(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_=lambda_star,
            mrs=mrs,
            markdown_steps=markdown_steps,
        )

    def _build_markdown_steps(
        self,
        *,
        optimal_x: float,
        optimal_y: float,
        optimal_utility: float,
        lambda_star: float,
    ) -> str:
        """SymPyを用いて、ラグランジアンおよび一階の条件の途中式をMarkdown/LaTeX形式で生成する.

        所得を表す記号には `I`（SymPyの虚数単位と衝突する）ではなく `M` を用いる。
        """
        x, y = sympy.symbols("x y", positive=True)
        price_x_sym, price_y_sym, income_sym = sympy.symbols(
            "P_x P_y M", positive=True
        )
        lam = sympy.symbols("lambda", positive=True)

        utility_expr = sympy.sympify(self.utility.get_symbolic_expression())
        lagrangian = utility_expr + lam * (
            income_sym - price_x_sym * x - price_y_sym * y
        )

        dl_dx = sympy.diff(lagrangian, x)
        dl_dy = sympy.diff(lagrangian, y)
        dl_dlambda = sympy.diff(lagrangian, lam)

        du_dx = sympy.diff(utility_expr, x)
        du_dy = sympy.diff(utility_expr, y)
        tangency_condition = sympy.Eq(
            du_dx / du_dy, price_x_sym / price_y_sym
        )

        sections = [
            "## 効用最大化問題の解法過程",
            "",
            "**効用関数:**",
            f"$$U(x, y) = {sympy.latex(utility_expr)}$$",
            "",
            "**予算制約:**",
            f"$$P_x x + P_y y = M$$",
            "",
            "**ラグランジアン:**",
            f"$$L = U(x, y) + \\lambda (M - P_x x - P_y y) "
            f"= {sympy.latex(lagrangian)}$$",
            "",
            "**一階の条件 (First-Order Conditions, FOC):**",
            f"$$\\frac{{\\partial L}}{{\\partial x}} = {sympy.latex(dl_dx)} = 0$$",
            f"$$\\frac{{\\partial L}}{{\\partial y}} = {sympy.latex(dl_dy)} = 0$$",
            f"$$\\frac{{\\partial L}}{{\\partial \\lambda}} = {sympy.latex(dl_dlambda)} = 0$$",
            "",
            "**限界代替率と価格比の一致（接点条件）:**",
            f"$$MRS_{{xy}} = \\frac{{\\partial U / \\partial x}}{{\\partial U / \\partial y}} "
            f"= \\frac{{P_x}}{{P_y}}$$",
            f"$${sympy.latex(tangency_condition)}$$",
            "",
            "**数値代入:**",
            f"$$P_x = {self.budget.price_x}, \\quad P_y = {self.budget.price_y}, "
            f"\\quad M = {self.budget.income}$$",
            "",
            "**閉形式解 (Closed-form solution):**",
            "$$x^{*} = \\frac{\\alpha}{\\alpha + \\beta} \\cdot \\frac{M}{P_x} "
            f"= {round(optimal_x, 4)}$$",
            "$$y^{*} = \\frac{\\beta}{\\alpha + \\beta} \\cdot \\frac{M}{P_y} "
            f"= {round(optimal_y, 4)}$$",
            "$$U^{*} = (x^{*})^{\\alpha} (y^{*})^{\\beta} "
            f"= {round(optimal_utility, 4)}$$",
            "$$\\lambda^{*} = \\frac{(\\alpha + \\beta) U^{*}}{M} "
            f"= {round(lambda_star, 4)}$$",
        ]
        return "\n".join(sections)
