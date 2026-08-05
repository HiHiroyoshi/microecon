# microecon

ミクロ経済学の基礎理論を、型安全かつ直感的に計算・可視化できる教育・実務向け Python ライブラリです。

## v0.1.0 スコープ

現バージョンではコブ＝ダグラス型効用関数 $U(x, y) = x^\alpha y^\beta$ を用いた
消費者の効用最大化問題（Consumer Problem）のみを完全実装対象としています。

- 数値解（最適消費量・効用・ラグランジュ乗数・MRS）は閉形式解によって決定論的に算出されます。
- `sympy` は途中式（ラグランジアン・一階の条件・微分過程）の Markdown/LaTeX 表現の生成にのみ使用され、
  数値計算そのものには利用されません。
- `matplotlib` によって予算制約線・無差別曲線・最適点を含むグラフを画像として出力できます。

## インストール

```bash
poetry install
```

## 使い方

```python
from microecon.consumer.models import CobbDouglasUtility, BudgetConstraint
from microecon.consumer.solver import ConsumerProblem
from microecon.consumer.plotter import ConsumerPlotter

utility = CobbDouglasUtility(alpha=0.5, beta=0.5)
budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)

problem = ConsumerProblem(utility=utility, budget=budget)
result = problem.solve()

print(result.optimal_x, result.optimal_y, result.optimal_utility)
print(result.markdown_steps)

ConsumerPlotter.save_plot(result, budget, "optimal.png")
```

## 開発

```bash
poetry run mypy microecon
poetry run pytest
```

---

## English

A type-safe, intuitive Python library for computing and visualizing the
foundational theory of microeconomics, aimed at both education and practical use.

### v0.1.0 Scope

The current version fully supports the consumer utility-maximization problem
for the Cobb-Douglas utility function $U(x, y) = x^\alpha y^\beta$ only.

- Numerical results (optimal consumption, utility, the Lagrange multiplier,
  and MRS) are computed deterministically via closed-form solutions.
- `sympy` is used only to generate the Markdown/LaTeX explanation of the
  derivation (the Lagrangian, first-order conditions, and differentiation
  steps) — it is never part of the numerical computation path.
- `matplotlib` renders the budget line, indifference curve, and optimal point
  as an image file.

### Installation

```bash
poetry install
```

### Usage

```python
from microecon.consumer.models import CobbDouglasUtility, BudgetConstraint
from microecon.consumer.solver import ConsumerProblem
from microecon.consumer.plotter import ConsumerPlotter

utility = CobbDouglasUtility(alpha=0.5, beta=0.5)
budget = BudgetConstraint(price_x=2.0, price_y=4.0, income=100.0)

problem = ConsumerProblem(utility=utility, budget=budget)
result = problem.solve()

print(result.optimal_x, result.optimal_y, result.optimal_utility)
print(result.markdown_steps)

ConsumerPlotter.save_plot(result, budget, "optimal.png")
```

### Development

```bash
poetry run mypy microecon
poetry run pytest
```
