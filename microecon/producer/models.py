"""生産者理論のドメインモデル（不可変dataclass群）.

v0.3.1では、コブ＝ダグラス型・完全代替型（線形）・レオンチェフ型・CES型の
生産関数を提供する。各クラスは :class:`BaseProductionFunction` の全抽象メソッドを
実装し、利潤最大化問題 (PMP) と費用最小化問題 (CMP) の双方を自身の閉形式解で解く。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import sympy

from microecon.exceptions import InvalidEconomicParameterError
from microecon.producer.base import BaseProductionFunction

SolutionType = Literal["interior", "corner", "kink", "indifferent"]
ReturnsToScale = Literal["DRS", "CRS", "IRS"]

_UNBOUNDED_CRS_IRS = (
    "Profit maximization problem (PMP) is unbounded under constant or "
    "increasing returns to scale ({detail}); no interior solution exists."
)


@dataclass(frozen=True)
class ProfitMaxResult:
    """生産者の利潤最大化問題 (PMP) の解.

    Attributes:
        optimal_labor: 最適要素投入（労働）L*。
        optimal_capital: 最適要素投入（資本）K*。
        optimal_output: 最適産出量 q* = f(L*, K*)。
        optimal_profit: 最大化された利潤 Pi* = p q* - w L* - r K*。
        solution_type: 解の種類。DRS下の内点解は常に "interior"。
        markdown_steps: 解法過程を示すMarkdown/LaTeX形式の解説文字列。
    """

    optimal_labor: float
    optimal_capital: float
    optimal_output: float
    optimal_profit: float
    solution_type: SolutionType
    markdown_steps: str


@dataclass(frozen=True)
class CostMinResult:
    """生産者の費用最小化問題 (CMP) の解.

    Attributes:
        optimal_labor: 条件付要素需要 L^c(q, w, r)。
        optimal_capital: 条件付要素需要 K^c(q, w, r)。
        minimum_cost: 最小化された総費用 C(q, w, r)。
        solution_type: 解の種類。"interior"（内点解）、"corner"（角点解）、
            "kink"（キンク点解）、"indifferent"（無差別解）のいずれか。
        markdown_steps: 解法過程を示すMarkdown/LaTeX形式の解説文字列。
    """

    optimal_labor: float
    optimal_capital: float
    minimum_cost: float
    solution_type: SolutionType
    markdown_steps: str


def _pmp_markdown(
    *,
    title: str,
    price: float,
    wage: float,
    rental: float,
    optimal_labor: float,
    optimal_capital: float,
    optimal_output: float,
    optimal_profit: float,
) -> str:
    """利潤最大化問題 (PMP) の解説文（全生産関数型共通の簡潔なテンプレート）."""
    sections = [
        f"## 利潤最大化問題の解法過程（{title}）",
        "",
        "**利潤関数:**",
        "$$\\Pi(L, K) = p f(L, K) - w L - r K$$",
        "",
        "**一階の条件 (FOC):**",
        "$$p \\cdot MP_L = w, \\qquad p \\cdot MP_K = r$$",
        "",
        "**数値代入:**",
        f"$$p = {price}, \\quad w = {wage}, \\quad r = {rental}$$",
        "",
        "**閉形式解:**",
        f"$$L^{{*}} = {round(optimal_labor, 4)}, \\quad K^{{*}} = "
        f"{round(optimal_capital, 4)}$$",
        f"$$q^{{*}} = {round(optimal_output, 4)}, \\quad \\Pi^{{*}} = "
        f"{round(optimal_profit, 4)}$$",
    ]
    return "\n".join(sections)


def _cmp_markdown(
    *,
    title: str,
    solution_type: SolutionType,
    target_output: float,
    wage: float,
    rental: float,
    optimal_labor: float,
    optimal_capital: float,
    minimum_cost: float,
) -> str:
    """費用最小化問題 (CMP) の解説文（全生産関数型共通の簡潔なテンプレート）."""
    sections = [
        f"## 費用最小化問題の解法過程（{title} / {solution_type}）",
        "",
        "**目標産出量と要素価格:**",
        f"$$q = {round(target_output, 4)}, \\quad w = {wage}, \\quad r = {rental}$$",
        "",
        "**条件付要素需要と最小費用:**",
        f"$$L^{{c}} = {round(optimal_labor, 4)}, \\quad K^{{c}} = "
        f"{round(optimal_capital, 4)}$$",
        f"$$C^{{*}} = w L^{{c}} + r K^{{c}} = {round(minimum_cost, 4)}$$",
    ]
    return "\n".join(sections)


# --------------------------------------------------------------------------
# 具象生産関数クラス
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CobbDouglasProduction(BaseProductionFunction):
    """コブ＝ダグラス型生産関数 q = A L^alpha K^beta.

    Attributes:
        A: 全要素生産性を表すスケールパラメータ。正の実数でなければならない。
        alpha: 労働に対する産出弾力性。正の実数でなければならない。
        beta: 資本に対する産出弾力性。正の実数でなければならない。
    """

    A: float
    alpha: float
    beta: float

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

    @property
    def returns_to_scale(self) -> ReturnsToScale:
        """alpha + beta の値に基づく規模に関する収穫の自動判定."""
        total = self.alpha + self.beta
        if math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return "CRS"
        return "DRS" if total < 1.0 else "IRS"

    def evaluate(self, labor: float, capital: float) -> float:
        return float(self.A * labor**self.alpha * capital**self.beta)

    def calculate_mrts(self, labor: float, capital: float) -> float:
        """MRTS = MP_L / MP_K = (alpha * K) / (beta * L) を返す."""
        return float((self.alpha * capital) / (self.beta * labor))

    def get_short_run_labor(self, target_output: float, fixed_capital: float) -> float:
        return float(
            (target_output / (self.A * fixed_capital**self.beta)) ** (1.0 / self.alpha)
        )

    def calculate_cost_function(
        self, target_output: float, wage: float, rental: float
    ) -> float:
        alpha, beta, a = self.alpha, self.beta, self.A
        s = alpha + beta
        bracket = (alpha / beta) ** (beta / s) + (beta / alpha) ** (alpha / s)
        return float(
            (target_output / a) ** (1.0 / s)
            * bracket
            * wage ** (alpha / s)
            * rental ** (beta / s)
        )

    def calculate_profit_function(
        self, price: float, wage: float, rental: float
    ) -> float:
        return self.solve_pmp(price, wage, rental).optimal_profit

    def get_cost_expression(self) -> sympy.Expr:
        q, w, r = sympy.symbols("q w r", positive=True)
        alpha, beta, a = self.alpha, self.beta, self.A
        s = alpha + beta
        bracket = (alpha / beta) ** (beta / s) + (beta / alpha) ** (alpha / s)
        return (q / a) ** (1 / s) * bracket * w ** (alpha / s) * r ** (beta / s)

    def get_profit_expression(self) -> sympy.Expr:
        if self.returns_to_scale != "DRS":
            raise ValueError(
                _UNBOUNDED_CRS_IRS.format(
                    detail=f"alpha + beta = {self.alpha + self.beta}"
                )
            )
        p, w, r = sympy.symbols("p w r", positive=True)
        alpha, beta, a = self.alpha, self.beta, self.A
        s = alpha + beta
        labor_expr = (
            a * alpha ** (1 - beta) * beta**beta * p / (w ** (1 - beta) * r**beta)
        ) ** (1 / (1 - s))
        capital_expr = (
            a * alpha**alpha * beta ** (1 - alpha) * p / (w**alpha * r ** (1 - alpha))
        ) ** (1 / (1 - s))
        output_expr = a * labor_expr**alpha * capital_expr**beta
        return p * output_expr - w * labor_expr - r * capital_expr

    def solve_pmp(self, price: float, wage: float, rental: float) -> ProfitMaxResult:
        if self.returns_to_scale != "DRS":
            raise ValueError(
                _UNBOUNDED_CRS_IRS.format(
                    detail=f"alpha + beta = {self.alpha + self.beta}"
                )
            )
        alpha, beta, a = self.alpha, self.beta, self.A
        s = alpha + beta

        optimal_labor = (
            a * alpha ** (1 - beta) * beta**beta * price / (wage ** (1 - beta) * rental**beta)
        ) ** (1 / (1 - s))
        optimal_capital = (
            a * alpha**alpha * beta ** (1 - alpha) * price / (wage**alpha * rental ** (1 - alpha))
        ) ** (1 / (1 - s))
        optimal_output = self.evaluate(optimal_labor, optimal_capital)
        optimal_profit = price * optimal_output - wage * optimal_labor - rental * optimal_capital

        markdown_steps = _pmp_markdown(
            title="コブ＝ダグラス型 / DRS",
            price=price,
            wage=wage,
            rental=rental,
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            optimal_output=optimal_output,
            optimal_profit=optimal_profit,
        )
        return ProfitMaxResult(
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            optimal_output=optimal_output,
            optimal_profit=optimal_profit,
            solution_type="interior",
            markdown_steps=markdown_steps,
        )

    def solve_cmp(
        self, target_output: float, wage: float, rental: float
    ) -> CostMinResult:
        alpha, beta = self.alpha, self.beta
        s = alpha + beta
        minimum_cost = self.calculate_cost_function(target_output, wage, rental)
        optimal_labor = (alpha / s) * minimum_cost / wage
        optimal_capital = (beta / s) * minimum_cost / rental

        markdown_steps = _cmp_markdown(
            title="コブ＝ダグラス型",
            solution_type="interior",
            target_output=target_output,
            wage=wage,
            rental=rental,
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
        )
        return CostMinResult(
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
            solution_type="interior",
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class LinearProduction(BaseProductionFunction):
    """完全代替型（線形）生産関数 q = a L + b K.

    労働と資本が完全代替であり、常に規模に関する収穫は一定（CRS）となるため、
    利潤最大化問題 (PMP) は常に非有界（ValueError）となる。

    Attributes:
        a: 労働の限界生産性。正の実数でなければならない。
        b: 資本の限界生産性。正の実数でなければならない。
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")

    def evaluate(self, labor: float, capital: float) -> float:
        return float(self.a * labor + self.b * capital)

    def calculate_mrts(self, labor: float, capital: float) -> float:
        """MRTS = a / b（(L, K) に依存せず常に一定）を返す."""
        return float(self.a / self.b)

    def get_short_run_labor(self, target_output: float, fixed_capital: float) -> float:
        if target_output <= self.b * fixed_capital:
            return 0.0
        return float((target_output - self.b * fixed_capital) / self.a)

    def calculate_cost_function(
        self, target_output: float, wage: float, rental: float
    ) -> float:
        return float(target_output * min(wage / self.a, rental / self.b))

    def calculate_profit_function(
        self, price: float, wage: float, rental: float
    ) -> float:
        raise ValueError(_UNBOUNDED_CRS_IRS.format(detail="perfect substitutes, CRS"))

    def get_cost_expression(self) -> sympy.Expr:
        q, w, r = sympy.symbols("q w r", positive=True)
        return q * sympy.Min(w / self.a, r / self.b)

    def get_profit_expression(self) -> sympy.Expr:
        raise ValueError(_UNBOUNDED_CRS_IRS.format(detail="perfect substitutes, CRS"))

    def solve_pmp(self, price: float, wage: float, rental: float) -> ProfitMaxResult:
        raise ValueError(_UNBOUNDED_CRS_IRS.format(detail="perfect substitutes, CRS"))

    def solve_cmp(
        self, target_output: float, wage: float, rental: float
    ) -> CostMinResult:
        a, b = self.a, self.b
        cost_per_a = wage / a
        cost_per_b = rental / b

        if math.isclose(cost_per_a, cost_per_b, rel_tol=1e-9, abs_tol=1e-9):
            optimal_labor = target_output / (2.0 * a)
            optimal_capital = target_output / (2.0 * b)
            solution_type: SolutionType = "indifferent"
        elif cost_per_a < cost_per_b:
            optimal_labor = target_output / a
            optimal_capital = 0.0
            solution_type = "corner"
        else:
            optimal_labor = 0.0
            optimal_capital = target_output / b
            solution_type = "corner"

        minimum_cost = wage * optimal_labor + rental * optimal_capital
        markdown_steps = _cmp_markdown(
            title="完全代替型",
            solution_type=solution_type,
            target_output=target_output,
            wage=wage,
            rental=rental,
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
        )
        return CostMinResult(
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
            solution_type=solution_type,
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class LeontiefProduction(BaseProductionFunction):
    """レオンチェフ型（固定比率）生産関数 q = min(a L, b K).

    労働と資本が完全補完であり、常に規模に関する収穫は一定（CRS）となるため、
    利潤最大化問題 (PMP) は常に非有界（ValueError）となる。

    Attributes:
        a: 労働の必要比率係数。正の実数でなければならない。
        b: 資本の必要比率係数。正の実数でなければならない。
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")

    def evaluate(self, labor: float, capital: float) -> float:
        return float(min(self.a * labor, self.b * capital))

    def calculate_mrts(self, labor: float, capital: float) -> float:
        """キンク点 (a L = b K) では微分不能なため 0.0 / math.inf / math.nan を返す."""
        a_labor = self.a * labor
        b_capital = self.b * capital
        if math.isclose(a_labor, b_capital, rel_tol=1e-9, abs_tol=1e-9):
            return math.nan
        if a_labor < b_capital:
            return 0.0
        return math.inf

    def get_short_run_labor(self, target_output: float, fixed_capital: float) -> float:
        if target_output > self.b * fixed_capital:
            raise ValueError(
                "Target output exceeds fixed capital capacity in the short run"
            )
        return float(target_output / self.a)

    def calculate_cost_function(
        self, target_output: float, wage: float, rental: float
    ) -> float:
        return float(target_output * (wage / self.a + rental / self.b))

    def calculate_profit_function(
        self, price: float, wage: float, rental: float
    ) -> float:
        raise ValueError(_UNBOUNDED_CRS_IRS.format(detail="perfect complements, CRS"))

    def get_cost_expression(self) -> sympy.Expr:
        q, w, r = sympy.symbols("q w r", positive=True)
        return q * (w / self.a + r / self.b)

    def get_profit_expression(self) -> sympy.Expr:
        raise ValueError(_UNBOUNDED_CRS_IRS.format(detail="perfect complements, CRS"))

    def solve_pmp(self, price: float, wage: float, rental: float) -> ProfitMaxResult:
        raise ValueError(_UNBOUNDED_CRS_IRS.format(detail="perfect complements, CRS"))

    def solve_cmp(
        self, target_output: float, wage: float, rental: float
    ) -> CostMinResult:
        a, b = self.a, self.b
        optimal_labor = target_output / a
        optimal_capital = target_output / b
        minimum_cost = wage * optimal_labor + rental * optimal_capital

        markdown_steps = _cmp_markdown(
            title="レオンチェフ型",
            solution_type="kink",
            target_output=target_output,
            wage=wage,
            rental=rental,
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
        )
        return CostMinResult(
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
            solution_type="kink",
            markdown_steps=markdown_steps,
        )


@dataclass(frozen=True)
class CESProduction(BaseProductionFunction):
    """CES型生産関数 q = A (a L^rho + b K^rho)^(gamma/rho).

    Attributes:
        A: 全要素生産性を表すスケールパラメータ。正の実数でなければならない。
        a: 労働に対する選好の強さを表す係数。正の実数でなければならない。
        b: 資本に対する選好の強さを表す係数。正の実数でなければならない。
        rho: 代替の弾力性を規定するパラメータ。rho != 0 かつ rho < 1 でなければならない。
        gamma: 規模に関する収穫を規定する指数。正の実数でなければならない。
            gamma < 1 ならばDRS、gamma = 1 ならばCRS、gamma > 1 ならばIRS。
    """

    A: float
    a: float
    b: float
    rho: float
    gamma: float

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise InvalidEconomicParameterError(f"A must be positive, got {self.A}")
        if self.a <= 0:
            raise InvalidEconomicParameterError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise InvalidEconomicParameterError(f"b must be positive, got {self.b}")
        if self.gamma <= 0:
            raise InvalidEconomicParameterError(
                f"gamma must be positive, got {self.gamma}"
            )
        if self.rho == 0 or self.rho >= 1:
            raise InvalidEconomicParameterError(
                f"rho must satisfy rho != 0 and rho < 1, got {self.rho}"
            )

    @property
    def sigma(self) -> float:
        """代替の弾力性 sigma = 1 / (1 - rho)."""
        return 1.0 / (1.0 - self.rho)

    @property
    def returns_to_scale(self) -> ReturnsToScale:
        """gamma の値に基づく規模に関する収穫の自動判定."""
        if math.isclose(self.gamma, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return "CRS"
        return "DRS" if self.gamma < 1.0 else "IRS"

    def evaluate(self, labor: float, capital: float) -> float:
        return float(
            self.A
            * (self.a * labor**self.rho + self.b * capital**self.rho)
            ** (self.gamma / self.rho)
        )

    def calculate_mrts(self, labor: float, capital: float) -> float:
        """MRTS = MP_L / MP_K = (a / b) * (L / K)^(rho - 1) を返す."""
        return float((self.a / self.b) * (labor / capital) ** (self.rho - 1.0))

    def _phi(self, wage: float, rental: float) -> float:
        sigma = self.sigma
        return float(
            self.a**sigma * wage ** (1.0 - sigma) + self.b**sigma * rental ** (1.0 - sigma)
        )

    def get_short_run_labor(self, target_output: float, fixed_capital: float) -> float:
        rho, a, b, gamma, capital_a = self.rho, self.a, self.b, self.gamma, self.A
        threshold = (target_output / capital_a) ** (rho / gamma) - b * fixed_capital**rho

        if rho > 0:
            if threshold <= 0:
                return 0.0
            return float((threshold / a) ** (1.0 / rho))

        # rho < 0: 資本のみによる産出量には物理的な上限（漸近的容量）が存在する。
        if threshold <= 0:
            raise ValueError(
                "Target output exceeds asymptotic capacity of fixed capital for rho < 0"
            )
        return float((threshold / a) ** (1.0 / rho))

    def calculate_cost_function(
        self, target_output: float, wage: float, rental: float
    ) -> float:
        sigma = self.sigma
        phi = self._phi(wage, rental)
        return float((target_output / self.A) ** (1.0 / self.gamma) * phi ** (1.0 / (1.0 - sigma)))

    def calculate_profit_function(
        self, price: float, wage: float, rental: float
    ) -> float:
        return self.solve_pmp(price, wage, rental).optimal_profit

    def get_cost_expression(self) -> sympy.Expr:
        q, w, r = sympy.symbols("q w r", positive=True)
        sigma = self.sigma
        a, b, gamma, capital_a = self.a, self.b, self.gamma, self.A
        phi = a**sigma * w ** (1 - sigma) + b**sigma * r ** (1 - sigma)
        return (q / capital_a) ** (1 / gamma) * phi ** (1 / (1 - sigma))

    def get_profit_expression(self) -> sympy.Expr:
        if self.returns_to_scale != "DRS":
            raise ValueError(
                _UNBOUNDED_CRS_IRS.format(detail=f"gamma = {self.gamma}")
            )
        p, w, r = sympy.symbols("p w r", positive=True)
        sigma = self.sigma
        a, b, rho, gamma, capital_a = self.a, self.b, self.rho, self.gamma, self.A

        phi = a**sigma * w ** (1 - sigma) + b**sigma * r ** (1 - sigma)
        inner = (gamma * p) / (capital_a ** (-1 / gamma) * phi ** (1 / (1 - sigma)))
        base = inner ** (1 / (1 - gamma))
        labor_expr = base * a**sigma * w**-sigma / phi
        capital_expr = base * b**sigma * r**-sigma / phi
        output_expr = capital_a * (a * labor_expr**rho + b * capital_expr**rho) ** (
            gamma / rho
        )
        return p * output_expr - w * labor_expr - r * capital_expr

    def solve_pmp(self, price: float, wage: float, rental: float) -> ProfitMaxResult:
        if self.returns_to_scale != "DRS":
            raise ValueError(
                _UNBOUNDED_CRS_IRS.format(detail=f"gamma = {self.gamma}")
            )
        sigma = self.sigma
        a, b, rho, gamma, capital_a = self.a, self.b, self.rho, self.gamma, self.A
        phi = self._phi(wage, rental)

        inner = (gamma * price) / (capital_a ** (-1.0 / gamma) * phi ** (1.0 / (1.0 - sigma)))
        base = inner ** (1.0 / (1.0 - gamma))
        optimal_labor = base * a**sigma * wage**-sigma / phi
        optimal_capital = base * b**sigma * rental**-sigma / phi
        optimal_output = self.evaluate(optimal_labor, optimal_capital)
        optimal_profit = (
            price * optimal_output - wage * optimal_labor - rental * optimal_capital
        )

        markdown_steps = _pmp_markdown(
            title="CES型 / DRS",
            price=price,
            wage=wage,
            rental=rental,
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            optimal_output=optimal_output,
            optimal_profit=optimal_profit,
        )
        return ProfitMaxResult(
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            optimal_output=optimal_output,
            optimal_profit=optimal_profit,
            solution_type="interior",
            markdown_steps=markdown_steps,
        )

    def solve_cmp(
        self, target_output: float, wage: float, rental: float
    ) -> CostMinResult:
        sigma = self.sigma
        a, b = self.a, self.b
        phi = self._phi(wage, rental)
        minimum_cost = self.calculate_cost_function(target_output, wage, rental)
        optimal_labor = minimum_cost * a**sigma * wage**-sigma / phi
        optimal_capital = minimum_cost * b**sigma * rental**-sigma / phi

        markdown_steps = _cmp_markdown(
            title="CES型",
            solution_type="interior",
            target_output=target_output,
            wage=wage,
            rental=rental,
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
        )
        return CostMinResult(
            optimal_labor=optimal_labor,
            optimal_capital=optimal_capital,
            minimum_cost=minimum_cost,
            solution_type="interior",
            markdown_steps=markdown_steps,
        )
