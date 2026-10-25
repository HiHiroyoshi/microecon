"""M 消費者・N 財の純粋交換経済における Arrow-Debreu 競争均衡."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from scipy.optimize import root

from microecon.general_equilibrium.consumer import BaseConsumer
from microecon.general_equilibrium.dataclasses import CompetitiveEquilibriumResult

logger = logging.getLogger(__name__)

_MARKET_CLEARING_TOL = 1e-6
_CORNER_TOL = 1e-9
_MRS_REL_TOL = 1e-4
# exp のオーバーフローを防ぐための対数価格の探索上限
_MAX_LOG_PRICE = 700.0


class ArrowDebreuEconomy:
    """一般の M 消費者・N 財純粋交換経済.

    Args:
        consumers: 経済を構成する消費者のタプル。

    Raises:
        ValueError: 消費者が存在しない場合、消費者名が重複する場合、
            初期保有の次元数が消費者間で一致しない場合、財の数が 2 未満の場合、
            または総初期保有量が 0 の財が存在する場合。
    """

    def __init__(self, consumers: Sequence[BaseConsumer]) -> None:
        consumers_tuple = tuple(consumers)
        if len(consumers_tuple) < 1:
            raise ValueError("Economy must have at least 1 consumer.")

        names = [c.name for c in consumers_tuple]
        if len(set(names)) != len(names):
            raise ValueError("Consumer names must be unique.")

        n_goods = consumers_tuple[0].n_goods
        if any(c.n_goods != n_goods for c in consumers_tuple):
            raise ValueError("All consumers must have the same endowment dimension.")
        if n_goods < 2:
            raise ValueError("Economy must have at least 2 goods.")

        total = np.sum([c.endowment for c in consumers_tuple], axis=0)
        if np.any(total <= 0.0):
            zero_goods = [int(k) for k in np.flatnonzero(total <= 0.0)]
            raise ValueError(
                f"Total endowment must be strictly positive for every good; "
                f"goods {zero_goods} have zero total endowment."
            )

        self._consumers = consumers_tuple
        self._n_goods = n_goods
        self._total_endowment = tuple(float(t) for t in total)

    @property
    def consumers(self) -> tuple[BaseConsumer, ...]:
        """経済を構成する消費者."""
        return self._consumers

    @property
    def n_goods(self) -> int:
        """財の数 ``N``."""
        return self._n_goods

    @property
    def total_endowment(self) -> tuple[float, ...]:
        """総初期保有量 ``Σ_m ω_m``."""
        return self._total_endowment

    def get_excess_demand(self, prices: Sequence[float] | np.ndarray) -> np.ndarray:
        """超過需要ベクトル ``z(p) = Σ_m x_m(p) - Σ_m ω_m`` を返す."""
        price_tuple = tuple(float(p) for p in prices)
        demand = np.sum([c.get_demand(price_tuple) for c in self._consumers], axis=0)
        return np.asarray(demand - np.asarray(self._total_endowment), dtype=float)

    def solve(self, numeraire_idx: int = 0) -> CompetitiveEquilibriumResult:
        """基準財 ``numeraire_idx`` の価格を 1.0 に固定した競争均衡を求める.

        非基準財の価格を対数価格 ``θ_k`` により ``p_k = exp(θ_k)`` とパラメータ化して
        探索中の価格の正値性を保証し、ワルラス法則により基準財を除く N-1 本の
        超過需要方程式 ``z_k(p(θ)) = 0`` を ``scipy.optimize.root(method='hybr')``
        （初期値 ``θ = 0``）で解く。

        Raises:
            ValueError: ``numeraire_idx`` が ``0 <= idx < N`` の範囲外の場合。
            RuntimeError: 求解が収束しない、市場清算誤差が ``1e-6`` を超える、
                または内点解の財について限界代替率と価格比が一致しない場合。
        """
        if not 0 <= numeraire_idx < self._n_goods:
            raise ValueError(
                f"numeraire_idx must satisfy 0 <= idx < {self._n_goods}, got {numeraire_idx}."
            )

        non_numeraire = [k for k in range(self._n_goods) if k != numeraire_idx]

        def prices_from_theta(theta: np.ndarray) -> np.ndarray:
            prices = np.ones(self._n_goods)
            prices[non_numeraire] = np.exp(np.clip(theta, -_MAX_LOG_PRICE, _MAX_LOG_PRICE))
            return prices

        def residual(theta: np.ndarray) -> np.ndarray:
            return self.get_excess_demand(prices_from_theta(theta))[non_numeraire]

        sol = root(residual, np.zeros(len(non_numeraire)), method="hybr")
        prices = prices_from_theta(sol.x)
        excess_demand = self.get_excess_demand(prices)
        market_clearing_error = float(np.linalg.norm(excess_demand))
        if (
            not sol.success
            or not np.isfinite(market_clearing_error)
            or market_clearing_error > _MARKET_CLEARING_TOL
        ):
            raise RuntimeError(
                "Failed to converge to a competitive equilibrium with acceptable precision."
            )

        price_tuple = tuple(float(p) for p in prices)
        allocations = {c.name: c.get_demand(price_tuple) for c in self._consumers}
        for consumer in self._consumers:
            self._verify_first_order_conditions(
                consumer, allocations[consumer.name], price_tuple, numeraire_idx
            )

        utilities = {
            c.name: c.utility_function.evaluate(*allocations[c.name]) for c in self._consumers
        }
        return CompetitiveEquilibriumResult(
            prices=price_tuple,
            allocations=allocations,
            utilities=utilities,
            market_clearing_error=market_clearing_error,
        )

    @staticmethod
    def _verify_first_order_conditions(
        consumer: BaseConsumer,
        allocation: tuple[float, ...],
        prices: tuple[float, ...],
        numeraire_idx: int,
    ) -> None:
        """内点解の財について ``MRS_{k,numeraire} = p_k / p_numeraire`` を検証する.

        消費量 ``< 1e-9`` の財は角解として検証をスキップし、基準財自体が角解の
        場合は当該消費者の検証全体をスキップする。
        """
        if allocation[numeraire_idx] < _CORNER_TOL:
            logger.warning(
                "Consumer '%s' is at a corner for the numeraire good; "
                "skipping first-order condition check.",
                consumer.name,
            )
            return

        for k in range(len(allocation)):
            if k == numeraire_idx:
                continue
            if allocation[k] < _CORNER_TOL:
                logger.warning(
                    "Consumer '%s' is at a corner for good %d; skipping its "
                    "first-order condition check.",
                    consumer.name,
                    k,
                )
                continue
            price_ratio = prices[k] / prices[numeraire_idx]
            mrs = consumer.get_mrs(allocation, k, numeraire_idx)
            if abs(mrs - price_ratio) / max(1.0, price_ratio) > _MRS_REL_TOL:
                raise RuntimeError(
                    f"First-order condition violated for consumer '{consumer.name}', "
                    f"good {k}: MRS={mrs}, price ratio={price_ratio}."
                )
