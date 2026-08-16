"""消費者理論のドメインモデル（不可変dataclass群）.

v0.2.0では、コブ＝ダグラス型に加えて完全代替型・準線形型・CES型・レオンチェフ型の
効用関数を提供する。各クラスは :class:`BaseUtilityFunction` の全抽象メソッドを実装し、
効用最大化問題 (UMP) と支出最小化問題 (EMP) の双方を自身の閉形式解で解く。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import sympy

from microecon.consumer.base import BaseUtilityFunction
from microecon.exceptions import InvalidEconomicParameterError

SolutionType = Literal["interior", "corner", "kink", "indifferent"]


@dataclass(frozen=True)
class BudgetConstraint:
    """予算制約 P_x * x + P_y * y = M を表す不可変モデル.

    Attributes:
        price_x: 財Xの価格 P_x。正の実数でなければならない。
        price_y: 財Yの価格 P_y。正の実数でなければならない。
        income: 所得 M。正の実数でなければならない。
    """

    price_x: float
    price_y: float
    income: float

    def __post_init__(self) -> None:
        if self.price_x <= 0:
            raise InvalidEconomicParameterError(
                f"price_x must be positive, got {self.price_x}"
            )
        if self.price_y <= 0:
            raise InvalidEconomicParameterError(
                f"price_y must be positive, got {self.price_y}"
            )
        if self.income <= 0:
            raise InvalidEconomicParameterError(
                f"income must be positive, got {self.income}"
            )


@dataclass(frozen=True)
class ExpenditureResult:
    """消費者の支出最小化問題 (EMP) の解.

    Attributes:
        hicksian_x: ヒックス需要 h_x(P_x, P_y, U)。
        hicksian_y: ヒックス需要 h_y(P_x, P_y, U)。
        expenditure: 最小支出 E(P_x, P_y, U)。
        solution_type: 解の種類。"interior"（内点解）、"corner"（角点解）、
            "kink"（キンク点解）、"indifferent"（無差別解）のいずれか。
        markdown_steps: 解法過程を示すMarkdown/LaTeX形式の解説文字列。
    """

    hicksian_x: float
    hicksian_y: float
    expenditure: float
    solution_type: SolutionType
    markdown_steps: str


@dataclass(frozen=True)
class OptimizationResult:
    """消費者の効用最大化問題 (UMP) の解.

    主問題（UMP）を解いた際に、双対問題（EMP）の解（ヒックス需要・最小支出）も
    自動的に統合して保持することで、主・双対の整合性を同一オブジェクト上で可視化する。

    Attributes:
        optimal_x: 最適消費量（マーシャル需要）x*。
        optimal_y: 最適消費量（マーシャル需要）y*。
        optimal_utility: 最適効用水準 U*。
        lambda_: ラグランジュ乗数（所得の限界効用・影の価格）。
        mrs: 最適点における限界代替率 MRS_xy。
        markdown_steps: 解法過程を示すMarkdown/LaTeX形式の解説文字列。
        solution_type: 解の種類。"interior"（内点解）、"corner"（角点解）、
            "kink"（キンク点解）、"indifferent"（無差別解）のいずれか。
        hicksian_x: 同一の効用水準 U* を目標としたEMPのヒックス需要 h_x。
        hicksian_y: 同一の効用水準 U* を目標としたEMPのヒックス需要 h_y。
        expenditure: 同一の効用水準 U* を実現する最小支出 E（理論上は所得 M と一致する）。
    """

    optimal_x: float
    optimal_y: float
    optimal_utility: float
    lambda_: float
    mrs: float
    markdown_steps: str
    solution_type: SolutionType
    hicksian_x: float
    hicksian_y: float
    expenditure: float


@dataclass(frozen=True)
class HicksianDecomposition:
    """価格変化に対する全効果のヒックス分解（代替効果・所得効果）.

    Attributes:
        price_x_before: 変化前の価格 P_x。
        price_x_after: 変化後の価格 P_x'。
        optimal_x_before: 変化前のマーシャル需要 x*(P_x, P_y, M)。
        optimal_x_after: 変化後のマーシャル需要 x*(P_x', P_y, M)。
        hicksian_x_at_new_price: 変化前の効用水準を維持するヒックス需要 x^h(P_x', P_y, U*)。
        substitution_effect: 代替効果 Δx^sub = x^h(P_x', P_y, U*) - x*(P_x, P_y, M)。
        income_effect: 所得効果 Δx^inc = x*(P_x', P_y, M) - x^h(P_x', P_y, U*)。
        total_effect: 全効果 Δx^total = x*(P_x', P_y, M) - x*(P_x, P_y, M)。
        markdown_steps: 分解過程を示すMarkdown/LaTeX形式の解説文字列。
    """

    price_x_before: float
    price_x_after: float
    optimal_x_before: float
    optimal_x_after: float
    hicksian_x_at_new_price: float
    substitution_effect: float
    income_effect: float
    total_effect: float
    markdown_steps: str


# --------------------------------------------------------------------------
# Markdown解説文の生成ヘルパー（solution_typeごとにテンプレートを分岐する）。
# 各効用関数クラスの solve_ump / solve_emp から呼び出される、モジュール非公開の
# 純粋関数群。ConsumerProblem はこれらを一切参照しない（委譲のみを行うため）。
# --------------------------------------------------------------------------


def _interior_ump_markdown(
    *,
    utility_expression_str: str,
    price_x: float,
    price_y: float,
    income: float,
    optimal_x: float,
    optimal_y: float,
    optimal_utility: float,
    lambda_star: float,
    closed_form_lines: list[str],
) -> str:
    """内点解: ラグランジアン→FOC→接点条件→閉形式解の順に解説する."""
    x, y = sympy.symbols("x y", positive=True)
    price_x_sym, price_y_sym, income_sym = sympy.symbols("P_x P_y M", positive=True)
    lam = sympy.symbols("lambda", positive=True)

    utility_expr = sympy.sympify(utility_expression_str, locals={"x": x, "y": y})
    lagrangian = utility_expr + lam * (
        income_sym - price_x_sym * x - price_y_sym * y
    )

    dl_dx = sympy.diff(lagrangian, x)
    dl_dy = sympy.diff(lagrangian, y)
    dl_dlambda = sympy.diff(lagrangian, lam)

    du_dx = sympy.diff(utility_expr, x)
    du_dy = sympy.diff(utility_expr, y)
    tangency_condition = sympy.Eq(du_dx / du_dy, price_x_sym / price_y_sym)

    sections = [
        "## 効用最大化問題の解法過程（内点解）",
        "",
        "**効用関数:**",
        f"$$U(x, y) = {sympy.latex(utility_expr)}$$",
        "",
        "**予算制約:**",
        "$$P_x x + P_y y = M$$",
        "",
        "**ラグランジアン:**",
        f"$$L = U(x, y) + \\lambda (M - P_x x - P_y y) = {sympy.latex(lagrangian)}$$",
        "",
        "**一階の条件 (First-Order Conditions, FOC):**",
        f"$$\\frac{{\\partial L}}{{\\partial x}} = {sympy.latex(dl_dx)} = 0$$",
        f"$$\\frac{{\\partial L}}{{\\partial y}} = {sympy.latex(dl_dy)} = 0$$",
        f"$$\\frac{{\\partial L}}{{\\partial \\lambda}} = {sympy.latex(dl_dlambda)} = 0$$",
        "",
        "**限界代替率と価格比の一致（接点条件）:**",
        "$$MRS_{xy} = \\frac{\\partial U / \\partial x}{\\partial U / \\partial y} "
        "= \\frac{P_x}{P_y}$$",
        f"$${sympy.latex(tangency_condition)}$$",
        "",
        "**数値代入:**",
        f"$$P_x = {price_x}, \\quad P_y = {price_y}, \\quad M = {income}$$",
        "",
        "**閉形式解 (Closed-form solution):**",
        *closed_form_lines,
    ]
    return "\n".join(sections)


def _corner_ump_markdown(
    *,
    mrs_value: float,
    price_ratio: float,
    chosen_good: Literal["x", "y"],
    optimal_x: float,
    optimal_y: float,
    optimal_utility: float,
    extra_note: str = "",
) -> str:
    """角点解: MRSと価格比の不一致→境界条件での最適化の順に解説する."""
    comparison = ">" if mrs_value > price_ratio else "<"
    boundary = "y = 0" if chosen_good == "x" else "x = 0"
    sections = [
        "## 効用最大化問題の解法過程（角点解）",
        "",
        "**限界代替率と価格比の比較:**",
        f"$$MRS_{{xy}} {comparison} \\frac{{P_x}}{{P_y}}$$",
        f"（数値: $MRS_{{xy}} = {round(mrs_value, 4)}$, "
        f"$P_x / P_y = {round(price_ratio, 4)}$）",
        "",
        "限界代替率と価格比が一致しないため、接点条件を満たす内点解は存在しない。"
        f"最適消費は財の一方のみを購入する境界（${boundary}$）に生じる。",
    ]
    if extra_note:
        sections += ["", extra_note]
    sections += [
        "",
        "**角点解:**",
        f"$$x^{{*}} = {round(optimal_x, 4)}, \\quad y^{{*}} = {round(optimal_y, 4)}$$",
        f"$$U^{{*}} = {round(optimal_utility, 4)}$$",
    ]
    return "\n".join(sections)


def _kink_ump_markdown(
    *,
    a: float,
    b: float,
    price_x: float,
    price_y: float,
    income: float,
    optimal_x: float,
    optimal_y: float,
    optimal_utility: float,
) -> str:
    """キンク点解: 微分不能性→固定比率と予算線の交点の順に解説する."""
    sections = [
        "## 効用最大化問題の解法過程（キンク点解）",
        "",
        "**効用関数:**",
        f"$$U(x, y) = \\min({a} x, {b} y)$$",
        "",
        f"レオンチェフ型効用関数はキンク点 $${a} x = {b} y$$ において微分不能であるため、"
        "ラグランジュ乗数法（FOCによる接点条件）を適用できない。"
        "効用は固定比率 $ax = by$ の下でのみ増加するため、最適解は常にこのキンク点上に存在する。",
        "",
        "**固定比率条件と予算制約の連立:**",
        f"$$y = \\frac{{{a}}}{{{b}}} x, \\qquad P_x x + P_y y = M$$",
        "",
        "**数値代入:**",
        f"$$P_x = {price_x}, \\quad P_y = {price_y}, \\quad M = {income}$$",
        "",
        "**キンク点解:**",
        "$$x^{*} = \\frac{b M}{b P_x + a P_y} = " f"{round(optimal_x, 4)}$$",
        "$$y^{*} = \\frac{a M}{b P_x + a P_y} = " f"{round(optimal_y, 4)}$$",
        f"$$U^{{*}} = \\min(a x^{{*}}, b y^{{*}}) = {round(optimal_utility, 4)}$$",
    ]
    return "\n".join(sections)


def _indifferent_ump_markdown(
    *,
    mrs_value: float,
    price_ratio: float,
    optimal_x: float,
    optimal_y: float,
    optimal_utility: float,
) -> str:
    """無差別解: MRSと価格比の完全一致→予算線上の任意の点が最適解である旨を解説する."""
    sections = [
        "## 効用最大化問題の解法過程（無差別解）",
        "",
        "**限界代替率と価格比の一致:**",
        f"$$MRS_{{xy}} = {round(mrs_value, 4)} = \\frac{{P_x}}{{P_y}} "
        f"= {round(price_ratio, 4)}$$",
        "",
        "限界代替率が価格比と完全に一致するため、無差別曲線（直線）と予算制約線が完全に重なる。"
        "したがって予算線上のあらゆる点が最適解となり、解は一意に定まらない。"
        "以下は代表点として予算を両財へ均等配分した値である。",
        "",
        "**代表的な最適解:**",
        f"$$x^{{*}} = {round(optimal_x, 4)}, \\quad y^{{*}} = {round(optimal_y, 4)}$$",
        f"$$U^{{*}} = {round(optimal_utility, 4)}$$",
    ]
    return "\n".join(sections)


def _emp_markdown(
    *,
    solution_type: SolutionType,
    target_utility: float,
    price_x: float,
    price_y: float,
    hicksian_x: float,
    hicksian_y: float,
    expenditure: float,
) -> str:
    """支出最小化問題 (EMP) の解説文（全effユーティリティ型共通の簡潔なテンプレート）."""
    sections = [
        f"## 支出最小化問題の解法過程（{solution_type}）",
        "",
        "**目標効用と価格:**",
        f"$$U = {round(target_utility, 4)}, \\quad P_x = {price_x}, \\quad P_y = {price_y}$$",
        "",
        "**ヒックス需要と最小支出:**",
        f"$$h_x = {round(hicksian_x, 4)}, \\quad h_y = {round(hicksian_y, 4)}$$",
        f"$$E^{{*}} = P_x h_x + P_y h_y = {round(expenditure, 4)}$$",
    ]
    return "\n".join(sections)


# --------------------------------------------------------------------------
# 具象効用関数クラス
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CobbDouglasUtility(BaseUtilityFunction):
    """コブ＝ダグラス型効用関数 U(x, y) = A * x^alpha * y^beta.

    Attributes:
        alpha: 財Xに対する選好の強さを表す指数。正の実数でなければならない。
        beta: 財Yに対する選好の強さを表す指数。正の実数でなければならない。
        A: スケールパラメータ。正の実数でなければならない。
            v0.1.0との後方互換性のためデフォルト値は 1.0。
    """

    alpha: float
    beta: float
    A: float = 1.0

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise InvalidEconomicParameterError(f"A must be positive, got {self.A}")
        if self.alpha <= 0:
            raise InvalidEconomicParameterError(
                f"alpha must be positive, got {self.alpha}"
            )
        if self.beta <= 0:
            raise InvalidEconomicParameterError(
                f"beta must be positive, got {self.beta}"
            )

    def evaluate(self, x: float, y: float) -> float:
        return float(self.A * (x**self.alpha) * (y**self.beta))

    def calculate_mrs(self, x: float, y: float) -> float:
        """MRS_xy = (alpha * y) / (beta * x) を返す（Aには依存しない）."""
        return float((self.alpha * y) / (self.beta * x))

    def get_symbolic_expression(self) -> str:
        """SymPyで解析可能な文字列 "A * x**alpha * y**beta" を返す（小数第4位で丸め）."""
        rounded_a = round(self.A, 4)
        rounded_alpha = round(self.alpha, 4)
        rounded_beta = round(self.beta, 4)
        return f"{rounded_a} * x**{rounded_alpha} * y**{rounded_beta}"

    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        alpha, beta, a = self.alpha, self.beta, self.A
        return float(
            a
            * (alpha / (alpha + beta)) ** alpha
            * (beta / (alpha + beta)) ** beta
            * income ** (alpha + beta)
            / (price_x**alpha * price_y**beta)
        )

    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        alpha, beta, a = self.alpha, self.beta, self.A
        return float(
            (
                target_utility
                / a
                * ((alpha + beta) / alpha) ** alpha
                * ((alpha + beta) / beta) ** beta
                * price_x**alpha
                * price_y**beta
            )
            ** (1.0 / (alpha + beta))
        )

    def get_indirect_utility_expression(self) -> sympy.Expr:
        price_x, price_y, income = sympy.symbols("P_x P_y M", positive=True)
        alpha, beta, a = self.alpha, self.beta, self.A
        return (
            a
            * (alpha / (alpha + beta)) ** alpha
            * (beta / (alpha + beta)) ** beta
            * income ** (alpha + beta)
            / (price_x**alpha * price_y**beta)
        )

    def get_expenditure_expression(self) -> sympy.Expr:
        price_x, price_y = sympy.symbols("P_x P_y", positive=True)
        target_utility = sympy.symbols("U", positive=True)
        alpha, beta, a = self.alpha, self.beta, self.A
        return (
            target_utility
            / a
            * ((alpha + beta) / alpha) ** alpha
            * ((alpha + beta) / beta) ** beta
            * price_x**alpha
            * price_y**beta
        ) ** (1 / (alpha + beta))

    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        alpha, beta = self.alpha, self.beta
        price_x, price_y, income = budget.price_x, budget.price_y, budget.income

        optimal_x = (alpha / (alpha + beta)) * (income / price_x)
        optimal_y = (beta / (alpha + beta)) * (income / price_y)
        optimal_utility = self.evaluate(optimal_x, optimal_y)
        lambda_star = ((alpha + beta) * optimal_utility) / income
        mrs = self.calculate_mrs(optimal_x, optimal_y)

        closed_form_lines = [
            "$$x^{*} = \\frac{\\alpha}{\\alpha + \\beta} \\cdot \\frac{M}{P_x} = "
            f"{round(optimal_x, 4)}$$",
            "$$y^{*} = \\frac{\\beta}{\\alpha + \\beta} \\cdot \\frac{M}{P_y} = "
            f"{round(optimal_y, 4)}$$",
            "$$U^{*} = A (x^{*})^{\\alpha} (y^{*})^{\\beta} = "
            f"{round(optimal_utility, 4)}$$",
            "$$\\lambda^{*} = \\frac{(\\alpha + \\beta) U^{*}}{M} = "
            f"{round(lambda_star, 4)}$$",
        ]
        markdown_steps = _interior_ump_markdown(
            utility_expression_str=self.get_symbolic_expression(),
            price_x=price_x,
            price_y=price_y,
            income=income,
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_star=lambda_star,
            closed_form_lines=closed_form_lines,
        )

        emp = self.solve_emp(optimal_utility, price_x, price_y)

        return OptimizationResult(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_=lambda_star,
            mrs=mrs,
            markdown_steps=markdown_steps,
            solution_type="interior",
            hicksian_x=emp.hicksian_x,
            hicksian_y=emp.hicksian_y,
            expenditure=emp.expenditure,
        )

    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        alpha, beta = self.alpha, self.beta
        expenditure = self.calculate_expenditure(price_x, price_y, target_utility)
        hicksian_x = (alpha / (alpha + beta)) * (expenditure / price_x)
        hicksian_y = (beta / (alpha + beta)) * (expenditure / price_y)

        markdown_steps = _emp_markdown(
            solution_type="interior",
            target_utility=target_utility,
            price_x=price_x,
            price_y=price_y,
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
        )
        return ExpenditureResult(
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
            solution_type="interior",
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class LinearUtility(BaseUtilityFunction):
    """完全代替型効用関数 U(x, y) = a x + b y.

    Attributes:
        a: 財Xの限界効用。正の実数でなければならない。
        b: 財Yの限界効用。正の実数でなければならない。
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")

    def evaluate(self, x: float, y: float) -> float:
        return float(self.a * x + self.b * y)

    def calculate_mrs(self, x: float, y: float) -> float:
        """MRS_xy = a / b（(x, y) に依存せず常に一定）を返す."""
        return float(self.a / self.b)

    def get_symbolic_expression(self) -> str:
        rounded_a = round(self.a, 4)
        rounded_b = round(self.b, 4)
        return f"{rounded_a} * x + {rounded_b} * y"

    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        return float(income * max(self.a / price_x, self.b / price_y))

    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        return float(target_utility * min(price_x / self.a, price_y / self.b))

    def get_indirect_utility_expression(self) -> sympy.Expr:
        price_x, price_y, income = sympy.symbols("P_x P_y M", positive=True)
        a, b = self.a, self.b
        return income * sympy.Max(a / price_x, b / price_y)

    def get_expenditure_expression(self) -> sympy.Expr:
        price_x, price_y = sympy.symbols("P_x P_y", positive=True)
        target_utility = sympy.symbols("U", positive=True)
        a, b = self.a, self.b
        return target_utility * sympy.Min(price_x / a, price_y / b)

    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        price_x, price_y, income = budget.price_x, budget.price_y, budget.income
        slope_ratio = self.a / self.b
        price_ratio = price_x / price_y
        mrs = self.calculate_mrs(0.0, 0.0)

        if math.isclose(slope_ratio, price_ratio, rel_tol=1e-9, abs_tol=1e-9):
            optimal_x = income / (2.0 * price_x)
            optimal_y = income / (2.0 * price_y)
            solution_type: SolutionType = "indifferent"
        elif slope_ratio > price_ratio:
            optimal_x = income / price_x
            optimal_y = 0.0
            solution_type = "corner"
        else:
            optimal_x = 0.0
            optimal_y = income / price_y
            solution_type = "corner"

        optimal_utility = self.evaluate(optimal_x, optimal_y)
        lambda_star = (
            self.a / price_x if slope_ratio >= price_ratio else self.b / price_y
        )

        if solution_type == "indifferent":
            markdown_steps = _indifferent_ump_markdown(
                mrs_value=mrs,
                price_ratio=price_ratio,
                optimal_x=optimal_x,
                optimal_y=optimal_y,
                optimal_utility=optimal_utility,
            )
        else:
            chosen_good: Literal["x", "y"] = "x" if optimal_y == 0.0 else "y"
            markdown_steps = _corner_ump_markdown(
                mrs_value=mrs,
                price_ratio=price_ratio,
                chosen_good=chosen_good,
                optimal_x=optimal_x,
                optimal_y=optimal_y,
                optimal_utility=optimal_utility,
                extra_note=(
                    "完全代替型効用関数の下では、限界代替率が価格比を上回る財へ"
                    "所得を全額投下することが最適となる（線形計画の端点解）。"
                ),
            )

        emp = self.solve_emp(optimal_utility, price_x, price_y)

        return OptimizationResult(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_=lambda_star,
            mrs=mrs,
            markdown_steps=markdown_steps,
            solution_type=solution_type,
            hicksian_x=emp.hicksian_x,
            hicksian_y=emp.hicksian_y,
            expenditure=emp.expenditure,
        )

    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        cost_per_util_x = price_x / self.a
        cost_per_util_y = price_y / self.b

        if math.isclose(cost_per_util_x, cost_per_util_y, rel_tol=1e-9, abs_tol=1e-9):
            hicksian_x = target_utility / (2.0 * self.a)
            hicksian_y = target_utility / (2.0 * self.b)
            solution_type: SolutionType = "indifferent"
        elif cost_per_util_x < cost_per_util_y:
            hicksian_x = target_utility / self.a
            hicksian_y = 0.0
            solution_type = "corner"
        else:
            hicksian_x = 0.0
            hicksian_y = target_utility / self.b
            solution_type = "corner"

        expenditure = price_x * hicksian_x + price_y * hicksian_y
        markdown_steps = _emp_markdown(
            solution_type=solution_type,
            target_utility=target_utility,
            price_x=price_x,
            price_y=price_y,
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
        )
        return ExpenditureResult(
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
            solution_type=solution_type,
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class QuasiLinearUtility(BaseUtilityFunction):
    """準線形型効用関数 U(x, y) = alpha * ln(x) + y.

    Attributes:
        alpha: 財Xに対する選好の強さを表す係数。正の実数でなければならない。
    """

    alpha: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise InvalidEconomicParameterError(
                f"alpha must be positive, got {self.alpha}"
            )

    def evaluate(self, x: float, y: float) -> float:
        return float(self.alpha * math.log(x) + y)

    def calculate_mrs(self, x: float, y: float) -> float:
        """MRS_xy = alpha / x を返す."""
        return float(self.alpha / x)

    def get_symbolic_expression(self) -> str:
        rounded_alpha = round(self.alpha, 4)
        return f"{rounded_alpha} * log(x) + y"

    def _income_threshold(self, price_y: float) -> float:
        """内点解が成立する所得の下限 alpha * P_y."""
        return self.alpha * price_y

    def _utility_threshold(self, price_x: float, price_y: float) -> float:
        """内点解が成立する効用の下限 alpha * ln(alpha * P_y / P_x)."""
        return self.alpha * math.log(self.alpha * price_y / price_x)

    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        alpha = self.alpha
        if income >= self._income_threshold(price_y):
            return float(
                alpha * math.log(alpha * price_y / price_x)
                + (income - alpha * price_y) / price_y
            )
        return float(alpha * math.log(income / price_x))

    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        alpha = self.alpha
        threshold = self._utility_threshold(price_x, price_y)
        if target_utility >= threshold:
            return float(price_y * (target_utility - threshold + alpha))
        return float(price_x * math.exp(target_utility / alpha))

    def get_indirect_utility_expression(self) -> sympy.Expr:
        price_x, price_y, income = sympy.symbols("P_x P_y M", positive=True)
        alpha = self.alpha
        interior = (
            alpha * sympy.log(alpha * price_y / price_x)
            + (income - alpha * price_y) / price_y
        )
        corner = alpha * sympy.log(income / price_x)
        return sympy.Piecewise((interior, income >= alpha * price_y), (corner, True))

    def get_expenditure_expression(self) -> sympy.Expr:
        price_x, price_y = sympy.symbols("P_x P_y", positive=True)
        target_utility = sympy.symbols("U", positive=True)
        alpha = self.alpha
        threshold = alpha * sympy.log(alpha * price_y / price_x)
        interior = price_y * (target_utility - threshold + alpha)
        corner = price_x * sympy.exp(target_utility / alpha)
        return sympy.Piecewise(
            (interior, target_utility >= threshold), (corner, True)
        )

    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        alpha = self.alpha
        price_x, price_y, income = budget.price_x, budget.price_y, budget.income

        if income >= self._income_threshold(price_y):
            optimal_x = alpha * price_y / price_x
            optimal_y = (income - alpha * price_y) / price_y
            lambda_star = 1.0 / price_y
            solution_type: SolutionType = "interior"
        else:
            optimal_x = income / price_x
            optimal_y = 0.0
            lambda_star = alpha / income
            solution_type = "corner"

        optimal_utility = self.evaluate(optimal_x, optimal_y)
        mrs = self.calculate_mrs(optimal_x, optimal_y)

        if solution_type == "interior":
            closed_form_lines = [
                "$$x^{*} = \\frac{\\alpha P_y}{P_x} = " f"{round(optimal_x, 4)}$$",
                "$$y^{*} = \\frac{M - \\alpha P_y}{P_y} = "
                f"{round(optimal_y, 4)}$$",
                "$$U^{*} = \\alpha \\ln(x^{*}) + y^{*} = "
                f"{round(optimal_utility, 4)}$$",
                "$$\\lambda^{*} = \\frac{1}{P_y} = " f"{round(lambda_star, 4)}$$",
            ]
            markdown_steps = _interior_ump_markdown(
                utility_expression_str=self.get_symbolic_expression(),
                price_x=price_x,
                price_y=price_y,
                income=income,
                optimal_x=optimal_x,
                optimal_y=optimal_y,
                optimal_utility=optimal_utility,
                lambda_star=lambda_star,
                closed_form_lines=closed_form_lines,
            )
        else:
            price_ratio = price_x / price_y
            markdown_steps = _corner_ump_markdown(
                mrs_value=mrs,
                price_ratio=price_ratio,
                chosen_good="x",
                optimal_x=optimal_x,
                optimal_y=optimal_y,
                optimal_utility=optimal_utility,
                extra_note=(
                    f"所得 $M = {income}$ が閾値 $\\alpha P_y = "
                    f"{round(alpha * price_y, 4)}$ を下回るため、"
                    "財Yの消費はゼロ ($y = 0$) に張り付き、"
                    "全所得が財Xの購入に充てられる。"
                ),
            )

        emp = self.solve_emp(optimal_utility, price_x, price_y)

        return OptimizationResult(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_=lambda_star,
            mrs=mrs,
            markdown_steps=markdown_steps,
            solution_type=solution_type,
            hicksian_x=emp.hicksian_x,
            hicksian_y=emp.hicksian_y,
            expenditure=emp.expenditure,
        )

    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        alpha = self.alpha
        threshold = self._utility_threshold(price_x, price_y)

        if target_utility >= threshold:
            hicksian_x = alpha * price_y / price_x
            hicksian_y = target_utility - threshold
            solution_type: SolutionType = "interior"
        else:
            hicksian_x = math.exp(target_utility / alpha)
            hicksian_y = 0.0
            solution_type = "corner"

        expenditure = price_x * hicksian_x + price_y * hicksian_y
        markdown_steps = _emp_markdown(
            solution_type=solution_type,
            target_utility=target_utility,
            price_x=price_x,
            price_y=price_y,
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
        )
        return ExpenditureResult(
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
            solution_type=solution_type,
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class CESUtility(BaseUtilityFunction):
    """CES型効用関数 U(x, y) = (a x^rho + b y^rho)^(1/rho).

    Attributes:
        a: 財Xに対する選好の強さを表す係数。正の実数でなければならない。
        b: 財Yに対する選好の強さを表す係数。正の実数でなければならない。
        rho: 代替の弾力性を規定するパラメータ。rho != 0 かつ rho < 1 でなければならない
            （rho = 0 はコブ＝ダグラス型の極限、rho >= 1 は凸性が破れるため定義域外）。
    """

    a: float
    b: float
    rho: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")
        if self.rho == 0 or self.rho >= 1:
            raise InvalidEconomicParameterError(
                f"rho must satisfy rho != 0 and rho < 1, got {self.rho}"
            )

    @property
    def sigma(self) -> float:
        """代替の弾力性 sigma = 1 / (1 - rho)."""
        return 1.0 / (1.0 - self.rho)

    def evaluate(self, x: float, y: float) -> float:
        return float((self.a * x**self.rho + self.b * y**self.rho) ** (1.0 / self.rho))

    def calculate_mrs(self, x: float, y: float) -> float:
        """MRS_xy = (a / b) * (x / y)^(rho - 1) を返す."""
        return float((self.a / self.b) * (x / y) ** (self.rho - 1.0))

    def get_symbolic_expression(self) -> str:
        rounded_a = round(self.a, 4)
        rounded_b = round(self.b, 4)
        rounded_rho = round(self.rho, 4)
        rounded_power = round(1.0 / self.rho, 8)
        return (
            f"({rounded_a} * x**{rounded_rho} + {rounded_b} * y**{rounded_rho})"
            f"**{rounded_power}"
        )

    def _price_index(self, price_x: float, price_y: float) -> float:
        sigma = self.sigma
        return float(
            self.a**sigma * price_x ** (1.0 - sigma)
            + self.b**sigma * price_y ** (1.0 - sigma)
        )

    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        sigma = self.sigma
        return float(
            income * self._price_index(price_x, price_y) ** (1.0 / (sigma - 1.0))
        )

    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        sigma = self.sigma
        return float(
            target_utility
            * self._price_index(price_x, price_y) ** (1.0 / (1.0 - sigma))
        )

    def get_indirect_utility_expression(self) -> sympy.Expr:
        price_x, price_y, income = sympy.symbols("P_x P_y M", positive=True)
        a, b, sigma = self.a, self.b, self.sigma
        price_index = a**sigma * price_x ** (1 - sigma) + b**sigma * price_y ** (
            1 - sigma
        )
        return income * price_index ** (1 / (sigma - 1))

    def get_expenditure_expression(self) -> sympy.Expr:
        price_x, price_y = sympy.symbols("P_x P_y", positive=True)
        target_utility = sympy.symbols("U", positive=True)
        a, b, sigma = self.a, self.b, self.sigma
        price_index = a**sigma * price_x ** (1 - sigma) + b**sigma * price_y ** (
            1 - sigma
        )
        return target_utility * price_index ** (1 / (1 - sigma))

    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        price_x, price_y, income = budget.price_x, budget.price_y, budget.income
        sigma = self.sigma
        price_index = self._price_index(price_x, price_y)

        optimal_x = (self.a**sigma * price_x**-sigma / price_index) * income
        optimal_y = (self.b**sigma * price_y**-sigma / price_index) * income
        optimal_utility = self.evaluate(optimal_x, optimal_y)
        lambda_star = (
            self.a
            * optimal_x ** (self.rho - 1.0)
            * optimal_utility ** (1.0 - self.rho)
            / price_x
        )
        mrs = self.calculate_mrs(optimal_x, optimal_y)

        closed_form_lines = [
            "$$x^{*} = \\frac{a^{\\sigma} P_x^{-\\sigma}}"
            "{a^{\\sigma} P_x^{1-\\sigma} + b^{\\sigma} P_y^{1-\\sigma}} M = "
            f"{round(optimal_x, 4)}$$",
            "$$y^{*} = \\frac{b^{\\sigma} P_y^{-\\sigma}}"
            "{a^{\\sigma} P_x^{1-\\sigma} + b^{\\sigma} P_y^{1-\\sigma}} M = "
            f"{round(optimal_y, 4)}$$",
            f"$$\\sigma = \\frac{{1}}{{1 - \\rho}} = {round(sigma, 4)}$$",
            f"$$U^{{*}} = {round(optimal_utility, 4)}$$",
        ]
        markdown_steps = _interior_ump_markdown(
            utility_expression_str=self.get_symbolic_expression(),
            price_x=price_x,
            price_y=price_y,
            income=income,
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_star=lambda_star,
            closed_form_lines=closed_form_lines,
        )

        emp = self.solve_emp(optimal_utility, price_x, price_y)

        return OptimizationResult(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_=lambda_star,
            mrs=mrs,
            markdown_steps=markdown_steps,
            solution_type="interior",
            hicksian_x=emp.hicksian_x,
            hicksian_y=emp.hicksian_y,
            expenditure=emp.expenditure,
        )

    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        sigma = self.sigma
        expenditure = self.calculate_expenditure(price_x, price_y, target_utility)
        hicksian_x = (
            self.a**sigma
            * price_x**-sigma
            * expenditure**sigma
            * target_utility ** (1.0 - sigma)
        )
        hicksian_y = (
            self.b**sigma
            * price_y**-sigma
            * expenditure**sigma
            * target_utility ** (1.0 - sigma)
        )
        markdown_steps = _emp_markdown(
            solution_type="interior",
            target_utility=target_utility,
            price_x=price_x,
            price_y=price_y,
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
        )
        return ExpenditureResult(
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
            solution_type="interior",
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class LeontiefUtility(BaseUtilityFunction):
    """レオンチェフ型（完全補完型）効用関数 U(x, y) = min(a x, b y).

    Attributes:
        a: 財Xの必要比率係数。正の実数でなければならない。
        b: 財Yの必要比率係数。正の実数でなければならない。
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")

    def evaluate(self, x: float, y: float) -> float:
        return float(min(self.a * x, self.b * y))

    def calculate_mrs(self, x: float, y: float) -> float:
        """キンク点 (a x = b y) では微分不能なため 0.0 / math.inf / math.nan を返す."""
        a_x = self.a * x
        b_y = self.b * y
        if math.isclose(a_x, b_y, rel_tol=1e-9, abs_tol=1e-9):
            return math.nan
        if a_x < b_y:
            return 0.0
        return math.inf

    def get_symbolic_expression(self) -> str:
        rounded_a = round(self.a, 4)
        rounded_b = round(self.b, 4)
        return f"Min({rounded_a} * x, {rounded_b} * y)"

    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        a, b = self.a, self.b
        return float((a * b * income) / (b * price_x + a * price_y))

    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        a, b = self.a, self.b
        return float(target_utility * (price_x / a + price_y / b))

    def get_indirect_utility_expression(self) -> sympy.Expr:
        price_x, price_y, income = sympy.symbols("P_x P_y M", positive=True)
        a, b = self.a, self.b
        return (a * b * income) / (b * price_x + a * price_y)

    def get_expenditure_expression(self) -> sympy.Expr:
        price_x, price_y = sympy.symbols("P_x P_y", positive=True)
        target_utility = sympy.symbols("U", positive=True)
        a, b = self.a, self.b
        return target_utility * (price_x / a + price_y / b)

    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        a, b = self.a, self.b
        price_x, price_y, income = budget.price_x, budget.price_y, budget.income

        denominator = b * price_x + a * price_y
        optimal_x = (b * income) / denominator
        optimal_y = (a * income) / denominator
        optimal_utility = self.evaluate(optimal_x, optimal_y)
        lambda_star = (a * b) / denominator
        mrs = self.calculate_mrs(optimal_x, optimal_y)

        markdown_steps = _kink_ump_markdown(
            a=a,
            b=b,
            price_x=price_x,
            price_y=price_y,
            income=income,
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
        )

        emp = self.solve_emp(optimal_utility, price_x, price_y)

        return OptimizationResult(
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_utility=optimal_utility,
            lambda_=lambda_star,
            mrs=mrs,
            markdown_steps=markdown_steps,
            solution_type="kink",
            hicksian_x=emp.hicksian_x,
            hicksian_y=emp.hicksian_y,
            expenditure=emp.expenditure,
        )

    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        a, b = self.a, self.b
        hicksian_x = target_utility / a
        hicksian_y = target_utility / b
        expenditure = price_x * hicksian_x + price_y * hicksian_y

        markdown_steps = _emp_markdown(
            solution_type="kink",
            target_utility=target_utility,
            price_x=price_x,
            price_y=price_y,
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
        )
        return ExpenditureResult(
            hicksian_x=hicksian_x,
            hicksian_y=hicksian_y,
            expenditure=expenditure,
            solution_type="kink",
            markdown_steps=markdown_steps,
        )
