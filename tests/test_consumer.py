"""microecon.consumer の単体テスト."""

from __future__ import annotations

import math
import re

import pytest

from microecon.consumer.base import BaseUtilityFunction
from microecon.consumer.models import (
    BudgetConstraint,
    CESUtility,
    CobbDouglasUtility,
    ExpenditureResult,
    LeontiefUtility,
    LinearUtility,
    OptimizationResult,
    QuasiLinearUtility,
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
    """UMP/EMPの閉形式解を未実装の効用関数を模したダミークラス.

    v0.2.0ではConsumerProblemがisinstance分岐を持たず、常に
    `utility.solve_ump()` へ委譲するポリモーフィックな設計となった。
    そのため「未対応の効用関数」という概念はConsumerProblem側ではなく、
    効用関数クラス自身がsolve_ump/solve_empを実装しているかどうかに移る。
    """

    def evaluate(self, x: float, y: float) -> float:
        return x + y

    def calculate_mrs(self, x: float, y: float) -> float:
        return 1.0

    def get_symbolic_expression(self) -> str:
        return "x + y"

    def solve_ump(self, budget: BudgetConstraint) -> OptimizationResult:
        raise NotImplementedError("DummyUtility does not implement solve_ump")

    def solve_emp(
        self, target_utility: float, price_x: float, price_y: float
    ) -> ExpenditureResult:
        raise NotImplementedError("DummyUtility does not implement solve_emp")

    def calculate_indirect_utility(
        self, price_x: float, price_y: float, income: float
    ) -> float:
        raise NotImplementedError

    def calculate_expenditure(
        self, price_x: float, price_y: float, target_utility: float
    ) -> float:
        raise NotImplementedError

    def get_indirect_utility_expression(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_expenditure_expression(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class TestUnsupportedUtilityType:
    """solve_ump/solve_empを実装しない効用関数に対するNotImplementedErrorの検証."""

    def test_dummy_utility_raises_not_implemented_error(self) -> None:
        utility = DummyUtility()
        budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)
        problem = ConsumerProblem(utility=utility, budget=budget)

        with pytest.raises(NotImplementedError):
            problem.solve()


class TestNumericAnchors:
    """手計算で確定した数値ケースを直接アサートする確定的アンカーテスト."""

    def test_cobb_douglas(self) -> None:
        utility = CobbDouglasUtility(A=1.0, alpha=1.0, beta=1.0)
        budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.optimal_x == pytest.approx(25.0, abs=1e-4)
        assert result.optimal_y == pytest.approx(12.5, abs=1e-4)
        assert result.optimal_utility == pytest.approx(312.5, abs=1e-4)
        assert result.solution_type == "interior"

    def test_linear_perfect_substitutes(self) -> None:
        utility = LinearUtility(a=1.0, b=2.0)
        budget = BudgetConstraint(price_x=2.0, price_y=3.0, income=100.0)
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.optimal_x == pytest.approx(0.0, abs=1e-4)
        assert result.optimal_y == pytest.approx(33.3333, abs=1e-4)
        assert result.optimal_utility == pytest.approx(66.6667, abs=1e-4)
        assert result.solution_type == "corner"

    def test_quasi_linear_interior(self) -> None:
        utility = QuasiLinearUtility(alpha=10.0)
        budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.optimal_x == pytest.approx(20.0, abs=1e-4)
        assert result.optimal_y == pytest.approx(15.0, abs=1e-4)
        assert result.optimal_utility == pytest.approx(44.9573, abs=1e-4)
        assert result.solution_type == "interior"

    def test_quasi_linear_corner(self) -> None:
        utility = QuasiLinearUtility(alpha=10.0)
        budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=15.0)
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.optimal_x == pytest.approx(7.5, abs=1e-4)
        assert result.optimal_y == pytest.approx(0.0, abs=1e-4)
        assert result.optimal_utility == pytest.approx(20.1490, abs=1e-4)
        assert result.solution_type == "corner"

    def test_ces(self) -> None:
        utility = CESUtility(a=1.0, b=1.0, rho=-1.0)
        budget = BudgetConstraint(price_x=2.0, price_y=2.0, income=100.0)
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.optimal_x == pytest.approx(25.0, abs=1e-4)
        assert result.optimal_y == pytest.approx(25.0, abs=1e-4)
        assert result.optimal_utility == pytest.approx(12.5, abs=1e-4)
        assert result.solution_type == "interior"

    def test_leontief(self) -> None:
        utility = LeontiefUtility(a=1.0, b=1.0)
        budget = BudgetConstraint(price_x=2.0, price_y=3.0, income=100.0)
        result = ConsumerProblem(utility=utility, budget=budget).solve()

        assert result.optimal_x == pytest.approx(20.0, abs=1e-4)
        assert result.optimal_y == pytest.approx(20.0, abs=1e-4)
        assert result.optimal_utility == pytest.approx(20.0, abs=1e-4)
        assert result.solution_type == "kink"


class TestCESGuardClause:
    """CES型のパラメータ制約（rho != 0 かつ rho < 1）の検証."""

    def test_rho_zero_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CESUtility(a=1.0, b=1.0, rho=0.0)

    def test_rho_greater_equal_one_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CESUtility(a=1.0, b=1.0, rho=1.0)

    def test_non_positive_a_raises(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CESUtility(a=0.0, b=1.0, rho=-1.0)


class TestLeontiefMrsKink:
    """レオンチェフ型のキンク点におけるMRSの分岐（0.0 / inf / nan）の検証."""

    def test_mrs_zero_when_ax_less_than_by(self) -> None:
        utility = LeontiefUtility(a=1.0, b=1.0)
        assert utility.calculate_mrs(1.0, 10.0) == 0.0

    def test_mrs_inf_when_ax_greater_than_by(self) -> None:
        utility = LeontiefUtility(a=1.0, b=1.0)
        assert utility.calculate_mrs(10.0, 1.0) == math.inf

    def test_mrs_nan_at_kink(self) -> None:
        utility = LeontiefUtility(a=1.0, b=1.0)
        assert math.isnan(utility.calculate_mrs(5.0, 5.0))
