"""Matplotlib による産業組織論モデルの可視化モジュール.

各メソッドは :class:`matplotlib.figure.Figure` を返す（保存は呼び出し側が
``fig.savefig`` で行う。Jupyter ではそのまま表示できる）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from matplotlib.figure import Figure

from microecon.industrial_organization.bertrand import DifferentiatedBertrand
from microecon.industrial_organization.cournot import CournotDuopoly
from microecon.industrial_organization.dataclasses import MonopolyResult, OligopolyResult
from microecon.industrial_organization.monopoly import Monopoly
from microecon.industrial_organization.stackelberg import StackelbergDuopoly

_DEMAND_COLOR = "#1f2937"
_MR_COLOR = "#2563eb"
_MC_COLOR = "#dc2626"
_CS_COLOR = "#93c5fd"
_PS_COLOR = "#86efac"
_DWL_COLOR = "#fca5a5"
_GRID_POINTS = 400


def _vectorize(expr: sp.Expr, var: sp.Symbol) -> Any:
    func = sp.lambdify(var, expr, "numpy")

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(func(x), dtype=float), x.shape)

    return evaluate


class IOPlotter:
    """独占・寡占モデルの図を描画するプロッター."""

    @staticmethod
    def plot_monopoly(monopoly: Monopoly, result: MonopolyResult) -> Figure:
        """逆需要・MR・MC 曲線、独占点・競争点、CS・PS・DWL の塗り分けを描画する."""
        q = monopoly.quantity_variable
        demand = _vectorize(monopoly.inverse_demand, q)
        marginal_revenue = _vectorize(monopoly.marginal_revenue, q)
        marginal_cost = _vectorize(monopoly.marginal_cost, q)

        q_max = max(result.competitive_quantity, result.quantity) * 1.25
        grid = np.linspace(0.0, q_max, _GRID_POINTS)

        fig = Figure(figsize=(8, 6))
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(grid, demand(grid), color=_DEMAND_COLOR, label="Inverse demand P(Q)")
        axes.plot(grid, marginal_cost(grid), color=_MC_COLOR, label="MC(Q)")

        if result.regime == "single_price":
            axes.plot(grid, marginal_revenue(grid), color=_MR_COLOR, linestyle="--", label="MR(Q)")
            q_m, p_m = result.quantity, result.price
            left = np.linspace(0.0, q_m, _GRID_POINTS)
            middle = np.linspace(q_m, result.competitive_quantity, _GRID_POINTS)
            axes.fill_between(left, demand(left), p_m, color=_CS_COLOR, alpha=0.6, label="CS")
            axes.fill_between(left, p_m, marginal_cost(left), color=_PS_COLOR, alpha=0.6, label="PS")
            axes.fill_between(
                middle, demand(middle), marginal_cost(middle), color=_DWL_COLOR, alpha=0.6,
                label="DWL",
            )
            axes.scatter([q_m], [p_m], color=_MR_COLOR, zorder=5, label="Monopoly (Q_m, P_m)")
            title = "Single-Price Monopoly"
        else:
            full = np.linspace(0.0, result.competitive_quantity, _GRID_POINTS)
            axes.fill_between(
                full, demand(full), marginal_cost(full), color=_PS_COLOR, alpha=0.6,
                label="PS (= total surplus)",
            )
            title = "First-Degree Price Discrimination"

        axes.scatter(
            [result.competitive_quantity], [result.competitive_price],
            color=_MC_COLOR, zorder=5, label="Competitive (Q_c, P_c)",
        )
        axes.set_xlim(0.0, q_max)
        axes.set_ylim(bottom=0.0)
        axes.set_xlabel("Quantity Q")
        axes.set_ylabel("Price P")
        axes.set_title(title)
        axes.legend()
        return fig

    @staticmethod
    def plot_reaction_curves(
        game: CournotDuopoly | StackelbergDuopoly | DifferentiatedBertrand,
        result: OligopolyResult,
    ) -> Figure:
        """両社の反応曲線と均衡点を描画する（シュタッケルベルクでは先導者の選択点も表示）."""
        br1, br2 = game.reaction_functions()
        if isinstance(game, DifferentiatedBertrand):
            s1, s2 = game.price_variables
            strategies = result.prices
            (lo1, hi1), (lo2, hi2) = game.price_bounds
            label = "Price"
            clip = False
        else:
            s1, s2 = game.quantity_variables
            strategies = result.quantities
            upper = max(max(strategies) * 2.0, 1.0)
            lo1, hi1, lo2, hi2 = 0.0, upper, 0.0, upper
            label = "Quantity"
            clip = True

        grid1 = np.linspace(lo1, hi1, _GRID_POINTS)
        grid2 = np.linspace(lo2, hi2, _GRID_POINTS)
        r1 = _vectorize(br1, s2)(grid2)
        r2 = _vectorize(br2, s1)(grid1)
        if clip:
            r1, r2 = np.maximum(r1, 0.0), np.maximum(r2, 0.0)

        fig = Figure(figsize=(7, 6))
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(r1, grid2, color=_MR_COLOR, label="Firm 1 reaction R1")
        axes.plot(grid1, r2, color=_MC_COLOR, label="Firm 2 reaction R2")
        if isinstance(game, StackelbergDuopoly):
            axes.scatter(
                [strategies[0]], [strategies[1]], color="#7c3aed", marker="*", s=200, zorder=5,
                label="Stackelberg (leader's choice on R2)",
            )
            title = "Stackelberg Duopoly"
        else:
            axes.scatter([strategies[0]], [strategies[1]], color=_DEMAND_COLOR, zorder=5,
                         label="Nash equilibrium")
            title = (
                "Differentiated Bertrand Duopoly"
                if isinstance(game, DifferentiatedBertrand)
                else "Cournot Duopoly"
            )
        axes.set_xlim(lo1, hi1)
        axes.set_ylim(lo2, hi2)
        axes.set_xlabel(f"{label} of firm 1")
        axes.set_ylabel(f"{label} of firm 2")
        axes.set_title(title)
        axes.legend()
        return fig
