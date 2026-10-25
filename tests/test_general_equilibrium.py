"""microecon.general_equilibrium の理論整合性テスト.

純粋交換経済（Edgeworth Box）・N 財 Arrow-Debreu 競争均衡・N 財コブ＝ダグラス
効用関数・入力検証ガードを検証する。
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from microecon.general_equilibrium import (
    ArrowDebreuEconomy,
    BaseConsumer,
    CompetitiveEquilibriumResult,
    EdgeworthBox,
    EdgeworthBoxResult,
    NGoodCobbDouglasUtility,
)

APPROX_ABS = 1e-4


def _consumer(
    name: str, weights: tuple[float, ...], endowment: tuple[float, ...]
) -> BaseConsumer:
    return BaseConsumer(name, NGoodCobbDouglasUtility(weights), endowment)


# --------------------------------------------------------------------------
# テスト1: 対称コブ＝ダグラス純粋交換経済
# --------------------------------------------------------------------------


class TestSymmetricEdgeworthBox:
    """u = x1^0.5 x2^0.5、ω_A = (10, 0)、ω_B = (0, 10)."""

    @pytest.fixture
    def eb(self) -> EdgeworthBox:
        return EdgeworthBox(
            _consumer("Consumer A", (0.5, 0.5), (10.0, 0.0)),
            _consumer("Consumer B", (0.5, 0.5), (0.0, 10.0)),
        )

    @pytest.fixture
    def res(self, eb: EdgeworthBox) -> EdgeworthBoxResult:
        return eb.analyze(numeraire_idx=0)

    def test_returns_edgeworth_box_result(self, res: EdgeworthBoxResult) -> None:
        assert isinstance(res, EdgeworthBoxResult)
        assert isinstance(res.competitive_equilibrium, CompetitiveEquilibriumResult)

    def test_prices(self, res: EdgeworthBoxResult) -> None:
        prices = res.competitive_equilibrium.prices
        assert prices[0] == 1.0
        assert prices[1] == pytest.approx(1.0, abs=APPROX_ABS)

    def test_allocations_and_utilities(self, res: EdgeworthBoxResult) -> None:
        ce = res.competitive_equilibrium
        assert ce.allocations["Consumer A"] == pytest.approx((5.0, 5.0), abs=APPROX_ABS)
        assert ce.allocations["Consumer B"] == pytest.approx((5.0, 5.0), abs=APPROX_ABS)
        assert ce.utilities["Consumer A"] == pytest.approx(5.0, abs=APPROX_ABS)
        assert ce.utilities["Consumer B"] == pytest.approx(5.0, abs=APPROX_ABS)
        assert ce.market_clearing_error < 1e-6

    def test_contract_curve_is_diagonal(
        self, eb: EdgeworthBox, res: EdgeworthBoxResult
    ) -> None:
        x1A = eb.x1A
        assert sp.simplify(res.contract_curve_expression - x1A) == 0

    def test_result_is_immutable(self, res: EdgeworthBoxResult) -> None:
        ce = res.competitive_equilibrium
        with pytest.raises((TypeError, AttributeError)):
            ce.allocations["Consumer A"] = (0.0, 0.0)  # type: ignore[index]
        with pytest.raises((TypeError, AttributeError)):
            ce.allocations["Consumer A"][0] = 0.0  # type: ignore[index]
        with pytest.raises((TypeError, AttributeError)):
            ce.utilities["Consumer A"] = 0.0  # type: ignore[index]


# --------------------------------------------------------------------------
# テスト2・3: 非対称初期保有・選好、基準財の変更
# --------------------------------------------------------------------------


@pytest.fixture
def asymmetric_box() -> EdgeworthBox:
    """u_A = x1^0.6 x2^0.4, ω_A = (8, 2); u_B = x1^0.2 x2^0.8, ω_B = (2, 8)."""
    return EdgeworthBox(
        _consumer("Consumer A", (0.6, 0.4), (8.0, 2.0)),
        _consumer("Consumer B", (0.2, 0.8), (2.0, 8.0)),
    )


class TestAsymmetricEdgeworthBox:
    def test_total_endowment(self, asymmetric_box: EdgeworthBox) -> None:
        res = asymmetric_box.analyze(numeraire_idx=0)
        assert res.total_endowment == pytest.approx((10.0, 10.0), abs=APPROX_ABS)

    def test_competitive_equilibrium(self, asymmetric_box: EdgeworthBox) -> None:
        ce = asymmetric_box.analyze(numeraire_idx=0).competitive_equilibrium
        assert ce.prices == pytest.approx((1.0, 12.0 / 7.0), abs=APPROX_ABS)
        assert ce.allocations["Consumer A"] == pytest.approx(
            (48.0 / 7.0, 8.0 / 3.0), abs=APPROX_ABS
        )
        assert ce.allocations["Consumer B"] == pytest.approx(
            (22.0 / 7.0, 22.0 / 3.0), abs=APPROX_ABS
        )
        assert ce.utilities["Consumer A"] == pytest.approx(4.6997, abs=APPROX_ABS)
        assert ce.utilities["Consumer B"] == pytest.approx(6.1902, abs=APPROX_ABS)
        assert ce.market_clearing_error < 1e-6

    @pytest.mark.parametrize(
        ("x1a_value", "expected"),
        [(2.0, 0.4), (5.0, 10.0 / 7.0), (48.0 / 7.0, 8.0 / 3.0)],
    )
    def test_contract_curve_points(
        self, asymmetric_box: EdgeworthBox, x1a_value: float, expected: float
    ) -> None:
        expr = asymmetric_box.get_contract_curve()
        x1A = asymmetric_box.x1A
        assert float(expr.subs(x1A, x1a_value)) == pytest.approx(expected, abs=APPROX_ABS)

    def test_contract_curve_has_no_float_coefficients(
        self, asymmetric_box: EdgeworthBox
    ) -> None:
        expr = asymmetric_box.get_contract_curve()
        x1A = asymmetric_box.x1A
        assert sp.simplify(expr - 2 * x1A / (12 - x1A)) == 0


class TestNumeraireChange:
    def test_numeraire_good_two(self, asymmetric_box: EdgeworthBox) -> None:
        ce1 = asymmetric_box.solve_competitive_equilibrium(numeraire_idx=1)
        ce0 = asymmetric_box.solve_competitive_equilibrium(numeraire_idx=0)
        assert ce1.prices[0] == pytest.approx(7.0 / 12.0, abs=APPROX_ABS)
        assert ce1.prices[1] == 1.0
        for name in ("Consumer A", "Consumer B"):
            assert ce1.allocations[name] == pytest.approx(
                ce0.allocations[name], abs=APPROX_ABS
            )

    @pytest.mark.parametrize("numeraire_idx", [2, -1])
    def test_out_of_range_numeraire(
        self, asymmetric_box: EdgeworthBox, numeraire_idx: int
    ) -> None:
        with pytest.raises(ValueError):
            asymmetric_box.solve_competitive_equilibrium(numeraire_idx=numeraire_idx)


# --------------------------------------------------------------------------
# テスト4: 3財 Arrow-Debreu 経済
# --------------------------------------------------------------------------


class TestThreeGoodEconomy:
    """両消費者 U = x1^0.5 x2^0.3 x3^0.2、ω_A = (10, 5, 0)、ω_B = (0, 5, 10)."""

    @pytest.fixture
    def economy(self) -> ArrowDebreuEconomy:
        utility = NGoodCobbDouglasUtility((0.5, 0.3, 0.2))
        return ArrowDebreuEconomy(
            (
                BaseConsumer("Consumer A", utility, (10.0, 5.0, 0.0)),
                BaseConsumer("Consumer B", utility, (0.0, 5.0, 10.0)),
            )
        )

    def test_equilibrium(self, economy: ArrowDebreuEconomy) -> None:
        ce = economy.solve(numeraire_idx=0)
        assert ce.prices == pytest.approx((1.0, 0.6, 0.4), abs=APPROX_ABS)
        assert ce.allocations["Consumer A"] == pytest.approx(
            (6.5, 6.5, 6.5), abs=APPROX_ABS
        )
        assert ce.allocations["Consumer B"] == pytest.approx(
            (3.5, 3.5, 3.5), abs=APPROX_ABS
        )
        assert ce.market_clearing_error < 1e-6

    def test_numeraire_invariance(self, economy: ArrowDebreuEconomy) -> None:
        ce = economy.solve(numeraire_idx=2)
        assert ce.prices == pytest.approx((2.5, 1.5, 1.0), abs=APPROX_ABS)
        assert ce.allocations["Consumer A"] == pytest.approx(
            (6.5, 6.5, 6.5), abs=APPROX_ABS
        )


# --------------------------------------------------------------------------
# テスト5: NGoodCobbDouglasUtility 単体
# --------------------------------------------------------------------------


class TestNGoodCobbDouglasUtility:
    @pytest.mark.parametrize("weights", [(0.5, 0.0), (-0.5, 0.5), (0.5, float("nan"))])
    def test_invalid_weights(self, weights: tuple[float, ...]) -> None:
        with pytest.raises(ValueError):
            NGoodCobbDouglasUtility(weights)

    def test_rational_exponents(self) -> None:
        utility = NGoodCobbDouglasUtility((0.6, 0.4))
        expr = utility.get_expression()
        x1, x2 = utility.utility_variables
        assert expr.atoms(sp.Float) == set()
        assert expr == x1 ** sp.Rational(3, 5) * x2 ** sp.Rational(2, 5)

    def test_auto_generated_variables(self) -> None:
        utility = NGoodCobbDouglasUtility((0.5, 0.3, 0.2))
        assert utility.utility_variables == sp.symbols("x1:4", positive=True)

    def test_variable_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            NGoodCobbDouglasUtility((0.5, 0.5), utility_variables=sp.symbols("a b c"))

    def test_evaluate_and_demand(self) -> None:
        utility = NGoodCobbDouglasUtility((0.5, 0.3, 0.2))
        assert utility.evaluate(6.5, 6.5, 6.5) == pytest.approx(6.5, abs=APPROX_ABS)
        assert utility.get_demand((1.0, 0.6, 0.4), 13.0) == pytest.approx(
            (6.5, 6.5, 6.5), abs=APPROX_ABS
        )

    @pytest.mark.parametrize(
        ("prices", "income"),
        [((1.0, -1.0), 10.0), ((1.0, 0.0), 10.0), ((1.0,), 10.0), ((1.0, 1.0), 0.0)],
    )
    def test_invalid_demand_inputs(
        self, prices: tuple[float, ...], income: float
    ) -> None:
        utility = NGoodCobbDouglasUtility((0.5, 0.5))
        with pytest.raises(ValueError):
            utility.get_demand(prices, income)


# --------------------------------------------------------------------------
# テスト6: 入力検証・例外ガード
# --------------------------------------------------------------------------


class TestValidationGuards:
    def test_duplicate_names_economy(self) -> None:
        a = _consumer("Same", (0.5, 0.5), (1.0, 0.0))
        b = _consumer("Same", (0.5, 0.5), (0.0, 1.0))
        with pytest.raises(ValueError, match="Consumer names must be unique"):
            ArrowDebreuEconomy((a, b))
        with pytest.raises(ValueError, match="Consumer names must be unique"):
            EdgeworthBox(a, b)

    def test_endowment_dimension_mismatch_in_consumer(self) -> None:
        with pytest.raises(ValueError):
            BaseConsumer("A", NGoodCobbDouglasUtility((0.5, 0.5)), (1.0, 1.0, 1.0))

    def test_endowment_dimension_mismatch_in_economy(self) -> None:
        a = _consumer("A", (0.5, 0.5), (1.0, 1.0))
        b = _consumer("B", (0.4, 0.3, 0.3), (1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="same endowment dimension"):
            ArrowDebreuEconomy((a, b))

    @pytest.mark.parametrize("endowment", [(0.0, 0.0), (-1.0, 5.0)])
    def test_invalid_endowment(self, endowment: tuple[float, ...]) -> None:
        with pytest.raises(ValueError):
            _consumer("A", (0.5, 0.5), endowment)

    def test_zero_total_endowment_good(self) -> None:
        a = _consumer("A", (0.4, 0.3, 0.3), (1.0, 0.0, 1.0))
        b = _consumer("B", (0.4, 0.3, 0.3), (1.0, 0.0, 2.0))
        with pytest.raises(ValueError):
            ArrowDebreuEconomy((a, b))

    def test_fewer_than_two_goods(self) -> None:
        a = _consumer("A", (1.0,), (5.0,))
        b = _consumer("B", (1.0,), (3.0,))
        with pytest.raises(ValueError, match="at least 2 goods"):
            ArrowDebreuEconomy((a, b))

    def test_edgeworth_requires_two_goods(self) -> None:
        a = _consumer("A", (0.4, 0.3, 0.3), (1.0, 1.0, 1.0))
        b = _consumer("B", (0.4, 0.3, 0.3), (1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            EdgeworthBox(a, b)

    def test_free_symbols_outside_utility_variables(self) -> None:
        class ParametricUtility(NGoodCobbDouglasUtility):
            def get_expression(self) -> sp.Expr:
                return sp.Symbol("theta") * super().get_expression()

        with pytest.raises(ValueError):
            BaseConsumer("A", ParametricUtility((0.5, 0.5)), (1.0, 1.0))

    def test_empty_economy(self) -> None:
        with pytest.raises(ValueError):
            ArrowDebreuEconomy(())


class TestMarginalRateOfSubstitution:
    def test_mrs_expression_and_value(self) -> None:
        consumer = _consumer("A", (0.6, 0.4), (8.0, 2.0))
        x1, x2 = consumer.utility_function.utility_variables
        assert sp.simplify(consumer.get_mrs_expression(0, 1) - 3 * x2 / (2 * x1)) == 0
        assert consumer.get_mrs((2.0, 4.0)) == pytest.approx(3.0, abs=APPROX_ABS)

    def test_mrs_at_zero_consumption_returns_inf(self) -> None:
        consumer = _consumer("A", (0.5, 0.5), (1.0, 1.0))
        assert consumer.get_mrs((0.0, 1.0)) == math.inf
        assert consumer.get_mrs((0.0, 0.0)) == math.inf
