# microecon

ミクロ経済学の基礎理論を、型安全かつ直感的に計算・可視化できる教育・実務向け Python ライブラリです。

## v0.3.1 スコープ

**消費者理論 (Consumer Theory)** と **生産者理論 (Producer Theory)** の双方を、
ポリモーフィックな設計のもとで統一的に扱います。各具象クラスが主問題・双対問題双方の
閉形式解を自ら実装し、ファサードクラス（`ConsumerProblem` / `ProducerProblem`）は
それらへの委譲のみを行います（型ごとの `if isinstance(...)` 分岐は持ちません）。

### 消費者理論

効用最大化問題 (UMP) と支出最小化問題 (EMP) の双対性を、以下5種類の効用関数について
閉形式解で解きます。

- コブ＝ダグラス型 `CobbDouglasUtility`
- 完全代替型（線形） `LinearUtility`
- 準線形型 `QuasiLinearUtility`
- CES型 `CESUtility`
- レオンチェフ型（完全補完） `LeontiefUtility`

価格変化に対するヒックス分解（代替効果・所得効果）、Royの恒等式・Shephardの補題の
記号微分による検証にも対応します。

### 生産者理論（v0.3.1で追加）

利潤最大化問題 (PMP) と費用最小化問題 (CMP) の双対性を、以下4種類の生産関数について
閉形式解で解きます。

- コブ＝ダグラス型 `CobbDouglasProduction`（規模に関する収穫 DRS/CRS/IRS を自動判定）
- 完全代替型（線形） `LinearProduction`
- レオンチェフ型（固定比率） `LeontiefProduction`
- CES型 `CESProduction`

規模に関する収穫が一定・逓増（CRS/IRS）の場合、利潤最大化問題は非有界となるため
`ValueError` を送出します（内点解はDRSケースのみ存在）。

また、総費用関数の記号微分から長期の限界費用・平均費用を導出する `CostAnalyzer`、
資本量を固定した下で固定費用・可変費用を分解する `ShortRunProduction` により、
長期・短期双方の費用構造分析を提供します。Hotellingの補題・Shephardの補題（生産者版）の
記号微分による検証にも対応します。

### 共通の設計方針

- 数値解は全て閉形式解によって決定論的に算出されます。
- `sympy` は途中式の解説（Markdown/LaTeX）の生成や、各種補題の記号微分による理論検証
  にのみ使用され、数値計算そのものには利用されません。
- `matplotlib` によって消費者理論の予算制約線・無差別曲線・最適点をグラフとして
  出力できます。

## インストール

```bash
poetry install
```

## 使い方

### 消費者理論

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

### 生産者理論

```python
from microecon.producer import CobbDouglasProduction, ProducerProblem
from microecon.producer.cost import CostAnalyzer, ShortRunProduction

production = CobbDouglasProduction(A=1.0, alpha=0.25, beta=0.25)
problem = ProducerProblem(production)

# 利潤最大化問題 (PMP) — alpha + beta < 1 (DRS) の場合のみ内点解が存在
pmp_result = problem.solve_profit_maximization(price=16.0, wage=2.0, rental=2.0)
print(pmp_result.optimal_labor, pmp_result.optimal_capital, pmp_result.optimal_profit)

# 費用最小化問題 (CMP)
cmp_result = problem.solve_cost_minimization(target_output=12.0, wage=4.0, rental=9.0)
print(cmp_result.optimal_labor, cmp_result.optimal_capital, cmp_result.minimum_cost)

# 長期費用指標（限界費用・平均費用）
analyzer = CostAnalyzer(production, wage=4.0, rental=9.0)
print(analyzer.marginal_cost(12.0), analyzer.average_cost(12.0))

# 短期費用分解（資本量を4に固定）
short_run = ShortRunProduction(production, fixed_capital=4.0, wage=2.0, rental=3.0)
print(short_run.fixed_cost(), short_run.variable_cost(8.0))
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

### v0.3.1 Scope

Both **Consumer Theory** and **Producer Theory** are supported under a unified,
polymorphic design. Each concrete class implements the closed-form solutions for
both its primal and dual problems, while the facade classes (`ConsumerProblem` /
`ProducerProblem`) do nothing but delegate to them — no `if isinstance(...)`
branching by type ever appears in the solvers.

#### Consumer Theory

The duality between the utility-maximization problem (UMP) and the
expenditure-minimization problem (EMP) is solved in closed form for five
utility function types:

- Cobb-Douglas: `CobbDouglasUtility`
- Perfect substitutes (linear): `LinearUtility`
- Quasi-linear: `QuasiLinearUtility`
- CES: `CESUtility`
- Leontief (perfect complements): `LeontiefUtility`

Hicksian decomposition (substitution and income effects) under a price change,
and symbolic verification of Roy's identity and Shephard's lemma, are also
supported.

#### Producer Theory (new in v0.3.1)

The duality between the profit-maximization problem (PMP) and the
cost-minimization problem (CMP) is solved in closed form for four production
function types:

- Cobb-Douglas: `CobbDouglasProduction` (automatically classifies returns to
  scale as DRS/CRS/IRS)
- Perfect substitutes (linear): `LinearProduction`
- Leontief (fixed proportions): `LeontiefProduction`
- CES: `CESProduction`

Under constant or increasing returns to scale (CRS/IRS), profit maximization
is unbounded and raises `ValueError` — an interior solution exists only under
decreasing returns to scale (DRS).

`CostAnalyzer` derives long-run marginal and average cost from the symbolic
derivative of the total cost function, while `ShortRunProduction` decomposes
fixed and variable cost with capital held fixed, covering both long-run and
short-run cost structure analysis. Symbolic verification of Hotelling's lemma
and the producer-side Shephard's lemma are also supported.

#### Shared design principles

- All numerical results are computed deterministically via closed-form
  solutions.
- `sympy` is used only to generate the Markdown/LaTeX explanation of the
  derivation and to verify the various lemmas via symbolic differentiation —
  it is never part of the numerical computation path.
- `matplotlib` renders the budget line, indifference curve, and optimal point
  for consumer theory as an image file.

### Installation

```bash
poetry install
```

### Usage

#### Consumer Theory

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

#### Producer Theory

```python
from microecon.producer import CobbDouglasProduction, ProducerProblem
from microecon.producer.cost import CostAnalyzer, ShortRunProduction

production = CobbDouglasProduction(A=1.0, alpha=0.25, beta=0.25)
problem = ProducerProblem(production)

# Profit-maximization problem (PMP) — an interior solution exists only under
# decreasing returns to scale (alpha + beta < 1)
pmp_result = problem.solve_profit_maximization(price=16.0, wage=2.0, rental=2.0)
print(pmp_result.optimal_labor, pmp_result.optimal_capital, pmp_result.optimal_profit)

# Cost-minimization problem (CMP)
cmp_result = problem.solve_cost_minimization(target_output=12.0, wage=4.0, rental=9.0)
print(cmp_result.optimal_labor, cmp_result.optimal_capital, cmp_result.minimum_cost)

# Long-run cost metrics (marginal cost, average cost)
analyzer = CostAnalyzer(production, wage=4.0, rental=9.0)
print(analyzer.marginal_cost(12.0), analyzer.average_cost(12.0))

# Short-run cost decomposition (capital fixed at 4)
short_run = ShortRunProduction(production, fixed_capital=4.0, wage=2.0, rental=3.0)
print(short_run.fixed_cost(), short_run.variable_cost(8.0))
```

### Development

```bash
poetry run mypy microecon
poetry run pytest
```
