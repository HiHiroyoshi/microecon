"""Matplotlibによる効用最大化問題の可視化モジュール."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from microecon.consumer.models import BudgetConstraint, OptimizationResult

_BUDGET_LINE_COLOR = "#1f2937"
_INDIFFERENCE_CURVE_COLOR = "#2563eb"
_OPTIMAL_POINT_COLOR = "#dc2626"


class ConsumerPlotter:
    """効用最大化問題の解を予算制約線・無差別曲線とともに描画するプロッター."""

    @staticmethod
    def save_plot(
        result: OptimizationResult,
        budget: BudgetConstraint,
        file_path: str,
        width: int = 8,
        height: int = 6,
    ) -> None:
        """予算制約線・無差別曲線・最適点を含むグラフを`file_path`へPNG等として保存する."""
        x_intercept = budget.income / budget.price_x
        y_intercept = budget.income / budget.price_y

        x_max = max(result.optimal_x * 2.0, x_intercept * 1.2)
        y_max = max(result.optimal_y * 2.0, y_intercept * 1.2)

        alpha, beta = _recover_exponents(result)

        figure: Figure = plt.figure(figsize=(width, height))
        axes = figure.add_subplot(1, 1, 1)

        budget_x = np.array([0.0, x_intercept])
        budget_y = np.array([y_intercept, 0.0])
        axes.plot(
            budget_x,
            budget_y,
            color=_BUDGET_LINE_COLOR,
            linewidth=2,
            label="Budget Constraint",
        )

        curve_x = np.linspace(x_max * 1e-4, x_max, 500)
        curve_y = (result.optimal_utility / curve_x**alpha) ** (1.0 / beta)
        axes.plot(
            curve_x,
            curve_y,
            color=_INDIFFERENCE_CURVE_COLOR,
            linewidth=2,
            label="Indifference Curve",
        )

        axes.plot(
            result.optimal_x,
            result.optimal_y,
            marker="o",
            markersize=8,
            color=_OPTIMAL_POINT_COLOR,
            linestyle="none",
            zorder=5,
        )
        axes.annotate(
            "Optimal",
            xy=(result.optimal_x, result.optimal_y),
            xytext=(10, 10),
            textcoords="offset points",
            color=_OPTIMAL_POINT_COLOR,
            fontweight="bold",
        )

        axes.set_xlim(0, x_max)
        axes.set_ylim(0, y_max)
        axes.set_xlabel("Goods X")
        axes.set_ylabel("Goods Y")
        axes.set_title("Utility Maximization (Cobb-Douglas)")
        axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        axes.legend(loc="upper right")

        figure.tight_layout()
        figure.savefig(file_path)
        plt.close(figure)


def _recover_exponents(result: OptimizationResult) -> tuple[float, float]:
    """OptimizationResultからコブ＝ダグラス型効用関数の指数 (alpha, beta) を逆算する.

    `OptimizationResult` は alpha, beta を直接保持しないため、以下の連立方程式を解く:
      1. MRS_xy = (alpha * y*) / (beta * x*)  より  alpha / beta = MRS_xy * x* / y*
      2. U* = (x*)^alpha * (y*)^beta より  ln(U*) = alpha * ln(x*) + beta * ln(y*)
    """
    ratio = result.mrs * result.optimal_x / result.optimal_y
    beta = np.log(result.optimal_utility) / (
        ratio * np.log(result.optimal_x) + np.log(result.optimal_y)
    )
    alpha = ratio * beta
    return float(alpha), float(beta)
