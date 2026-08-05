"""microecon.consumer の単体テスト."""

from __future__ import annotations

import re

import pytest

from microecon.consumer.base import BaseUtilityFunction
from microecon.consumer.models import (
    BudgetConstraint,
    CobbDouglasUtility,
    OptimizationResult,
)
from microecon.consumer.solver import ConsumerProblem
from microecon.exceptions import InvalidEconomicParameterError

# SymPyの虚数単位 `I` が誤って解説文に混入していないかを検出する正規表現。
# 英単語中の "I" (例: "Utility", "First") は対象外とし、独立したトークンとしての
# "I" や "I*" のみを検出する。
_STRAY_IMAGINARY_UNIT_PATTERN = re.compile(r"(?<![A-Za-z])I(?![A-Za-z])")


@pytest.fixture
def standard_result() -> OptimizationResult:
    utility = CobbDouglasUtility(alpha=0.5, beta=0.5)
    budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)
    problem = ConsumerProblem(utility=utility, budget=budget)
    return problem.solve()


class TestStandardSolution:
    """標準的なパラメータでの閉形式解の検証."""

    def test_optimal_x(self, standard_result: OptimizationResult) -> None:
        assert standard_result.optimal_x == pytest.approx(25.0, abs=1e-4)

    def test_optimal_y(self, standard_result: OptimizationResult) -> None:
        assert standard_result.optimal_y == pytest.approx(12.5, abs=1e-4)

    def test_optimal_utility(self, standard_result: OptimizationResult) -> None:
        assert standard_result.optimal_utility == pytest.approx(17.67766, abs=1e-4)

    def test_lambda(self, standard_result: OptimizationResult) -> None:
        assert standard_result.lambda_ == pytest.approx(0.17677, abs=1e-4)

    def test_mrs_equals_price_ratio(self, standard_result: OptimizationResult) -> None:
        assert standard_result.mrs == pytest.approx(0.5, abs=1e-4)


class TestMarkdownSteps:
    """途中式（markdown_steps）の質の検証."""

    def test_not_empty(self, standard_result: OptimizationResult) -> None:
        assert len(standard_result.markdown_steps) > 0

    def test_no_stray_imaginary_unit(self, standard_result: OptimizationResult) -> None:
        match = _STRAY_IMAGINARY_UNIT_PATTERN.search(standard_result.markdown_steps)
        assert match is None, (
            f"Found stray SymPy imaginary unit 'I' in markdown_steps: {match}"
        )

    def test_contains_lagrangian_and_foc_symbols(
        self, standard_result: OptimizationResult
    ) -> None:
        assert "\\lambda" in standard_result.markdown_steps
        assert "\\partial" in standard_result.markdown_steps


class TestInvalidParameters:
    """異常な経済パラメータに対するガード節の検証."""

    def test_non_positive_price_x_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            BudgetConstraint(price_x=0.0, price_y=4.0, income=100.0)

    def test_negative_price_x_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            BudgetConstraint(price_x=-2.0, price_y=4.0, income=100.0)

    def test_non_positive_price_y_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            BudgetConstraint(price_x=2.0, price_y=0.0, income=100.0)

    def test_non_positive_income_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            BudgetConstraint(price_x=2.0, price_y=4.0, income=0.0)

    def test_negative_income_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            BudgetConstraint(price_x=2.0, price_y=4.0, income=-100.0)

    def test_non_positive_alpha_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CobbDouglasUtility(alpha=0.0, beta=0.5)

    def test_non_positive_beta_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CobbDouglasUtility(alpha=0.5, beta=0.0)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CobbDouglasUtility(alpha=-0.5, beta=0.5)


class DummyUtility(BaseUtilityFunction):
    """v0.1.0では未対応の効用関数を模した、ConsumerProblem.solve()のTypeError検証用ダミークラス."""

    def evaluate(self, x: float, y: float) -> float:
        return x + y

    def calculate_mrs(self, x: float, y: float) -> float:
        return 1.0

    def get_symbolic_expression(self) -> str:
        return "x + y"


class TestUnsupportedUtilityType:
    """CobbDouglasUtility以外の効用関数が渡された場合のTypeError検証."""

    def test_dummy_utility_raises_type_error(self) -> None:
        utility = DummyUtility()
        budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)
        problem = ConsumerProblem(utility=utility, budget=budget)

        with pytest.raises(TypeError):
            problem.solve()
