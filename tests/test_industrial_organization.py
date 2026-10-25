"""microecon.industrial_organization の理論整合性テスト.

独占・第1種/第3種価格差別・ベルトラン・シュタッケルベルク・クールノー・市場構造比較・
入力検証・図を、解析解のアンカー値で検証する。
"""

from __future__ import annotations

import dataclasses

import matplotlib

matplotlib.use("Agg")

import pytest
import sympy as sp
from matplotlib.figure import Figure

from microecon.exceptions import InvalidEconomicParameterError, NoEquilibriumError
from microecon.industrial_organization import (
    CournotDuopoly,
    DifferentiatedBertrand,
    HomogeneousBertrand,
    IOPlotter,
    Monopoly,
    MonopolyResult,
    OligopolyResult,
    StackelbergDuopoly,
    ThirdDegreePriceDiscrimination,
    compare_market_structures,
)

APPROX_ABS = 1e-4

Q = sp.Symbol("Q")
q1, q2 = sp.symbols("q1 q2")
p1, p2, q = sp.symbols("p1 p2 q")


def approx(value: object) -> object:
    return pytest.approx(value, abs=APPROX_ABS)


# --------------------------------------------------------------------------
# テスト1: 単一価格独占（P = 100 − Q）
# --------------------------------------------------------------------------


class TestSinglePriceMonopoly:
    @pytest.fixture
    def result(self) -> MonopolyResult:
        return Monopoly(100 - Q, Q, 20 * Q).solve_single_price()

    def test_equilibrium(self, result: MonopolyResult) -> None:
        assert result.regime == "single_price"
        assert (result.quantity, result.price) == approx((40.0, 60.0))
        assert (result.competitive_quantity, result.competitive_price) == approx((80.0, 20.0))

    def test_welfare(self, result: MonopolyResult) -> None:
        assert result.consumer_surplus == approx(800.0)
        assert (result.producer_surplus, result.profit) == approx((1600.0, 1600.0))
        assert (result.social_welfare, result.competitive_welfare) == approx((2400.0, 3200.0))
        assert result.deadweight_loss == approx(800.0)
        assert result.competitive_welfare - result.social_welfare == approx(
            result.deadweight_loss
        )
        integral = float(sp.integrate((100 - Q) - 20, (Q, 40, 80)))
        assert result.deadweight_loss == approx(integral)

    def test_lerner_index_equals_inverse_elasticity(self, result: MonopolyResult) -> None:
        assert result.lerner_index == approx(2.0 / 3.0)
        elasticity = 1.0 * result.price / result.quantity  # |dQ/dP| = 1
        assert result.lerner_index == approx(1.0 / elasticity)

    def test_fixed_cost_separates_ps_and_profit(self) -> None:
        result = Monopoly(100 - Q, Q, 20 * Q + 500).solve_single_price()
        assert (result.quantity, result.price) == approx((40.0, 60.0))
        assert result.consumer_surplus == approx(800.0)
        assert result.producer_surplus == approx(1600.0)
        assert result.profit == approx(1100.0)
        assert result.deadweight_loss == approx(800.0)

    def test_first_degree_price_discrimination(self) -> None:
        result = Monopoly(100 - Q, Q, 20 * Q).solve_first_degree_price_discrimination()
        assert result.regime == "first_degree"
        assert (result.quantity, result.price) == approx((80.0, 20.0))
        assert result.consumer_surplus == approx(0.0)
        assert (result.producer_surplus, result.profit) == approx((3200.0, 3200.0))
        assert result.deadweight_loss == approx(0.0)
        assert result.lerner_index is None

    def test_increasing_cubic_cost_is_not_natural_monopoly(self) -> None:
        """P = MC の負の根で MC' < 0 となっても、自然独占と誤判定しない."""
        result = Monopoly(100 - Q, Q, 20 * Q + Q**3 / 3).solve_single_price()
        assert result.competitive_quantity > 0
        assert result.deadweight_loss >= 0


# --------------------------------------------------------------------------
# テスト2: 第3種価格差別（P1 = 100 − q1、P2 = 80 − 2 q2）
# --------------------------------------------------------------------------


class TestThirdDegreePriceDiscrimination:
    def test_constant_marginal_cost(self) -> None:
        result = ThirdDegreePriceDiscrimination(
            (100 - q1, 80 - 2 * q2), (q1, q2), 20 * Q, Q
        ).solve()
        assert result.quantities == approx((40.0, 15.0))
        assert result.prices == approx((60.0, 50.0))
        assert result.total_quantity == approx(55.0)
        assert result.consumer_surpluses == approx((800.0, 225.0))
        assert result.consumer_surplus == approx(1025.0)
        assert result.profit == approx(2050.0)
        assert result.deadweight_loss == approx(1025.0)

    def test_increasing_marginal_cost_links_markets(self) -> None:
        result = ThirdDegreePriceDiscrimination(
            (100 - q1, 80 - 2 * q2), (q1, q2), Q**2 / 2, Q
        ).solve()
        assert result.quantities == approx((30.0, 10.0))
        assert result.total_quantity == approx(40.0)
        assert result.prices == approx((70.0, 60.0))
        assert result.marginal_cost == approx(40.0)
        assert result.consumer_surpluses == approx((450.0, 100.0))
        assert result.consumer_surplus == approx(550.0)
        assert (result.profit, result.producer_surplus) == approx((1900.0, 1900.0))
        assert (result.social_welfare, result.competitive_welfare) == approx((2450.0, 2680.0))
        assert result.deadweight_loss == approx(230.0)

    def test_lower_elasticity_market_has_higher_price(self) -> None:
        result = ThirdDegreePriceDiscrimination(
            (100 - q1, 80 - 2 * q2), (q1, q2), 20 * Q, Q
        ).solve()
        assert result.lerner_indices[0] > result.lerner_indices[1]
        assert result.prices[0] > result.prices[1]


# --------------------------------------------------------------------------
# テスト3: ベルトラン
# --------------------------------------------------------------------------


class TestDifferentiatedBertrand:
    def test_symmetric_affine_demand(self) -> None:
        result = DifferentiatedBertrand(
            (100 - 2 * p1 + p2, 100 - 2 * p2 + p1), (p1, p2), (10 * q, 10 * q), q,
            ((0, 100), (0, 100)),
        ).find_pure_nash()
        assert result.model_type == "bertrand_differentiated"
        assert result.prices == approx((40.0, 40.0))
        assert result.market_price is None
        assert result.quantities == approx((60.0, 60.0))
        assert result.profits == approx((1800.0, 1800.0))
        assert result.consumer_surplus == approx(3600.0)
        assert result.social_welfare == approx(7200.0)
        assert result.deadweight_loss == approx(900.0)
        assert result.is_corner is False
        assert result.welfare_note == ""

    def test_asymmetric_jacobian_has_no_consumer_surplus(self) -> None:
        result = DifferentiatedBertrand(
            (100 - 2 * p1 + p2, 100 - 2 * p2 + sp.Rational(1, 2) * p1), (p1, p2),
            (10 * q, 10 * q), q, ((0, 100), (0, 100)),
        ).find_pure_nash()
        assert result.prices == approx((1200.0 / 31.0, 1080.0 / 31.0))
        assert result.consumer_surplus is None
        assert result.social_welfare is None
        assert result.deadweight_loss is None
        assert result.welfare_note != ""


class TestHomogeneousBertrand:
    @pytest.mark.parametrize(
        ("costs", "market_price", "prices", "quantities", "profits", "cs", "dwl", "corner"),
        [
            ((10.0, 10.0), 10.0, (10.0, 10.0), (45.0, 45.0), (0.0, 0.0), 4050.0, 0.0, False),
            ((10.0, 30.0), 30.0, (30.0, 30.0), (70.0, 0.0), (1400.0, 0.0), 2450.0, 200.0, True),
            ((10.0, 70.0), 55.0, (55.0, 70.0), (45.0, 0.0), (2025.0, 0.0), 1012.5, 1012.5, True),
        ],
    )
    def test_cases(
        self,
        costs: tuple[float, float],
        market_price: float,
        prices: tuple[float, float],
        quantities: tuple[float, float],
        profits: tuple[float, float],
        cs: float,
        dwl: float,
        corner: bool,
    ) -> None:
        result = HomogeneousBertrand(100 - Q, Q, costs).solve()
        assert result.market_price == approx(market_price)
        assert result.prices == approx(prices)
        assert result.quantities == approx(quantities)
        assert result.profits == approx(profits)
        assert result.consumer_surplus == approx(cs)
        assert result.deadweight_loss == approx(dwl)
        assert result.is_corner is corner


# --------------------------------------------------------------------------
# テスト4: シュタッケルベルクとクールノー（P = 100 − (q1 + q2)）
# --------------------------------------------------------------------------


@pytest.fixture
def stackelberg() -> StackelbergDuopoly:
    return StackelbergDuopoly(100 - (q1 + q2), (q1, q2), (10 * q1, 10 * q2))


@pytest.fixture
def cournot() -> CournotDuopoly:
    return CournotDuopoly(100 - (q1 + q2), (q1, q2), (10 * q1, 10 * q2))


class TestStackelbergAndCournot:
    def test_follower_reaction_function(self, stackelberg: StackelbergDuopoly) -> None:
        reaction = stackelberg.follower_reaction_function()
        assert float(reaction.subs(q1, 20)) == approx(35.0)
        assert float(reaction.subs(q1, 100)) == approx(0.0)
        interior = stackelberg.reaction_functions()[1]
        assert sp.simplify(interior - (90 - q1) / 2) == 0

    def test_stackelberg_equilibrium(self, stackelberg: StackelbergDuopoly) -> None:
        result = stackelberg.solve()
        assert result.quantities == approx((45.0, 22.5))
        assert sum(result.quantities) == approx(67.5)
        assert result.market_price == approx(32.5)
        assert result.prices == approx((32.5, 32.5))
        assert result.profits == approx((1012.5, 506.25))
        assert result.consumer_surplus == approx(2278.125)
        assert result.deadweight_loss == approx(253.125)
        assert result.is_corner is False

    def test_cournot_equilibrium(self, cournot: CournotDuopoly) -> None:
        result = cournot.solve()
        assert result.quantities == approx((30.0, 30.0))
        assert result.market_price == approx(40.0)
        assert result.profits == approx((900.0, 900.0))
        assert result.consumer_surplus == approx(1800.0)
        assert result.deadweight_loss == approx(450.0)

    def test_first_mover_advantage(
        self, stackelberg: StackelbergDuopoly, cournot: CournotDuopoly
    ) -> None:
        stack, cour = stackelberg.solve(), cournot.solve()
        assert stack.profits[0] > cour.profits[0]
        assert stack.profits[1] < cour.profits[1]

    def test_follower_exit_uses_piecewise_reaction(self) -> None:
        result = StackelbergDuopoly(100 - (q1 + q2), (q1, q2), (10 * q1, 60 * q2)).solve()
        assert result.quantities == approx((45.0, 0.0))
        assert result.market_price == approx(55.0)
        assert result.profits == approx((2025.0, 0.0))
        assert result.is_corner is True

    def test_cournot_with_nonconstant_marginal_cost(self) -> None:
        result = CournotDuopoly(100 - (q1 + q2), (q1, q2), (10 * q1, q2**2)).solve()
        assert result.quantities == approx((260.0 / 7.0, 110.0 / 7.0))
        assert isinstance(result.consumer_surplus, float)
        assert isinstance(result.social_welfare, float)
        assert result.deadweight_loss is None
        assert result.welfare_note != ""


# --------------------------------------------------------------------------
# テスト5: 市場構造比較（P = 100 − Q、MC = 10）
# --------------------------------------------------------------------------


class TestMarketStructureComparison:
    EXPECTED = {
        "perfect_competition": (90.0, 10.0, 4050.0, 0.0, 0.0),
        "bertrand_homogeneous": (90.0, 10.0, 4050.0, 0.0, 0.0),
        "stackelberg": (67.5, 32.5, 2278.125, 1518.75, 253.125),
        "cournot": (60.0, 40.0, 1800.0, 1800.0, 450.0),
        "monopoly": (45.0, 55.0, 1012.5, 2025.0, 1012.5),
    }

    def test_rows(self) -> None:
        comparison = compare_market_structures(100 - Q, Q, 10.0)
        assert len(comparison.rows) == 5
        quantities = [row.quantity for row in comparison.rows]
        assert quantities == sorted(quantities, reverse=True)
        assert {row.structure for row in comparison.rows[:2]} == {
            "perfect_competition", "bertrand_homogeneous",
        }
        assert [row.structure for row in comparison.rows[2:]] == [
            "stackelberg", "cournot", "monopoly",
        ]
        for row in comparison.rows:
            expected = self.EXPECTED[row.structure]
            actual = (
                row.quantity, row.price, row.consumer_surplus, row.producer_surplus,
                row.deadweight_loss,
            )
            assert actual == approx(expected)

    def test_markdown_table(self) -> None:
        table = compare_market_structures(100 - Q, Q, 10.0).to_markdown_table()
        lines = table.splitlines()
        assert lines[0].startswith("| 市場構造")
        data_rows = lines[2:]
        assert len(data_rows) == 5
        for name in ("完全競争", "ベルトラン", "シュタッケルベルク", "クールノー", "独占"):
            assert any(name in row for row in data_rows)

    def test_negative_marginal_cost(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            compare_market_structures(100 - Q, Q, -1.0)


# --------------------------------------------------------------------------
# テスト6: 入力検証・不変性・解説文
# --------------------------------------------------------------------------


class TestValidation:
    def test_undeclared_symbol_in_demand(self) -> None:
        a = sp.Symbol("a")
        with pytest.raises(InvalidEconomicParameterError):
            Monopoly(a - Q, Q, 20 * Q)
        with pytest.raises(InvalidEconomicParameterError):
            CournotDuopoly(a - (q1 + q2), (q1, q2), (10 * q1, 10 * q2))
        with pytest.raises(InvalidEconomicParameterError):
            DifferentiatedBertrand(
                (a - 2 * p1 + p2, 100 - 2 * p2 + p1), (p1, p2), (10 * q, 10 * q), q,
                ((0, 100), (0, 100)),
            )

    def test_negative_marginal_cost(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            HomogeneousBertrand(100 - Q, Q, (-1.0, 10.0))
        with pytest.raises(InvalidEconomicParameterError):
            Monopoly(100 - Q, Q, -5 * Q + Q**2)

    def test_non_finite_price_bounds(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            DifferentiatedBertrand(
                (100 - 2 * p1 + p2, 100 - 2 * p2 + p1), (p1, p2), (10 * q, 10 * q), q,
                ((0, float("inf")), (0, 100)),
            )

    def test_inverted_price_bounds_do_not_leak_plain_value_error(self) -> None:
        with pytest.raises(InvalidEconomicParameterError) as exc_info:
            DifferentiatedBertrand(
                (100 - 2 * p1 + p2, 100 - 2 * p2 + p1), (p1, p2), (10 * q, 10 * q), q,
                ((50, 10), (0, 100)),
            )
        assert type(exc_info.value) is not ValueError

    def test_natural_monopoly(self) -> None:
        with pytest.raises(NoEquilibriumError, match="自然独占"):
            Monopoly(100 - Q, Q, 60 * Q - sp.Rational(1, 4) * Q**2).solve_single_price()
        with pytest.raises(NoEquilibriumError, match="自然独占"):
            Monopoly(
                100 - Q, Q, 60 * Q - sp.Rational(1, 4) * Q**2
            ).solve_first_degree_price_discrimination()

    def test_third_degree_market_closure(self) -> None:
        with pytest.raises(NoEquilibriumError, match="市場閉鎖"):
            ThirdDegreePriceDiscrimination((100 - q1, 30 - q2), (q1, q2), 40 * Q, Q).solve()

    def test_third_degree_invalid_inputs(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            ThirdDegreePriceDiscrimination((100 - q1,), (q1,), 20 * Q, Q)
        with pytest.raises(InvalidEconomicParameterError):
            ThirdDegreePriceDiscrimination((100 - q1, 80 - 2 * q2), (q1,), 20 * Q, Q)

    def test_cournot_demand_must_depend_on_total_output(self) -> None:
        with pytest.raises(InvalidEconomicParameterError):
            CournotDuopoly(100 - 2 * q1 - q2, (q1, q2), (10 * q1, 10 * q2))

    def test_no_demand_in_homogeneous_bertrand(self) -> None:
        with pytest.raises(NoEquilibriumError):
            HomogeneousBertrand(100 - Q, Q, (120.0, 130.0)).solve()

    def test_results_are_frozen(self, cournot: CournotDuopoly) -> None:
        monopoly = Monopoly(100 - Q, Q, 20 * Q).solve_single_price()
        with pytest.raises(dataclasses.FrozenInstanceError):
            monopoly.price = 0.0  # type: ignore[misc]
        oligopoly = cournot.solve()
        with pytest.raises(dataclasses.FrozenInstanceError):
            oligopoly.quantities = (0.0, 0.0)  # type: ignore[misc]
        comparison = compare_market_structures(100 - Q, Q, 10.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            comparison.rows[0].price = 0.0  # type: ignore[misc]


class TestMarkdownSteps:
    def test_monopoly(self) -> None:
        steps = Monopoly(100 - Q, Q, 20 * Q).solve_single_price().markdown_steps
        for keyword in ("限界収入", "限界費用", "2階条件", "消費者余剰", "死荷重", "ラーナー指数"):
            assert keyword in steps
        assert "$$" in steps and "**" in steps and steps.startswith("## ")

    def test_first_degree(self) -> None:
        steps = Monopoly(100 - Q, Q, 20 * Q).solve_first_degree_price_discrimination().markdown_steps
        assert "留保価格" in steps and "生産者余剰" in steps

    def test_third_degree(self) -> None:
        steps = ThirdDegreePriceDiscrimination(
            (100 - q1, 80 - 2 * q2), (q1, q2), Q**2 / 2, Q
        ).solve().markdown_steps
        for keyword in ("限界収入", "ヘッセ行列", "弾力性が低い市場ほど高価格"):
            assert keyword in steps

    def test_cournot(self, cournot: CournotDuopoly) -> None:
        steps = cournot.solve().markdown_steps
        assert "FOC" in steps and "反応関数" in steps and "均衡" in steps

    def test_stackelberg(self, stackelberg: StackelbergDuopoly) -> None:
        steps = stackelberg.solve().markdown_steps
        for keyword in ("ステップ1", "ステップ2", "均衡"):
            assert keyword in steps

    def test_differentiated_bertrand(self) -> None:
        steps = DifferentiatedBertrand(
            (100 - 2 * p1 + p2, 100 - 2 * p2 + p1), (p1, p2), (10 * q, 10 * q), q,
            ((0, 100), (0, 100)),
        ).find_pure_nash().markdown_steps
        for keyword in ("利潤", "FOC", "反応関数", "戦略的補完", "消費者余剰"):
            assert keyword in steps

    @pytest.mark.parametrize(
        ("costs", "keyword"),
        [((10.0, 10.0), "対称"), ((10.0, 30.0), "費用優位"), ((10.0, 70.0), "抜本的費用優位")],
    )
    def test_homogeneous_bertrand(self, costs: tuple[float, float], keyword: str) -> None:
        assert keyword in HomogeneousBertrand(100 - Q, Q, costs).solve().markdown_steps


# --------------------------------------------------------------------------
# テスト7: 図
# --------------------------------------------------------------------------


class TestPlotter:
    def test_plot_monopoly(self) -> None:
        monopoly = Monopoly(100 - Q, Q, 20 * Q)
        for result in (
            monopoly.solve_single_price(),
            monopoly.solve_first_degree_price_discrimination(),
        ):
            fig = IOPlotter.plot_monopoly(monopoly, result)
            assert isinstance(fig, Figure)
            assert fig.axes[0].get_xlabel() and fig.axes[0].get_ylabel()

    def test_plot_reaction_curves(
        self, cournot: CournotDuopoly, stackelberg: StackelbergDuopoly
    ) -> None:
        bertrand = DifferentiatedBertrand(
            (100 - 2 * p1 + p2, 100 - 2 * p2 + p1), (p1, p2), (10 * q, 10 * q), q,
            ((0, 100), (0, 100)),
        )
        pairs: list[tuple[CournotDuopoly | StackelbergDuopoly | DifferentiatedBertrand,
                          OligopolyResult]] = [
            (cournot, cournot.solve()),
            (stackelberg, stackelberg.solve()),
            (bertrand, bertrand.find_pure_nash()),
        ]
        for game, result in pairs:
            fig = IOPlotter.plot_reaction_curves(game, result)
            assert isinstance(fig, Figure)
            assert fig.axes[0].get_xlabel() and fig.axes[0].get_ylabel()
