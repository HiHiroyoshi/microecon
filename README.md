# microecon

ミクロ経済学の基礎理論を、型安全かつ直感的に計算・可視化できる教育・実務向け Python ライブラリです。

## v0.5.0 スコープ

**不確実性下の意思決定・リスク選好・動的（異時点間）選好分析 (Uncertainty, Risk
Preference & Dynamic Preference Analysis)** を追加しました。期待効用理論に基づく
リスク指標の算出と、割引関数による異時点間の消費選択評価を、v0.4.0までと同様の
ポリモーフィックな設計のもとで提供します。

### 不確実性・リスク選好・動的選好（v0.5.0で追加）

ベルヌーイ効用関数 `BaseBernoulliUtility` と割引関数 `BaseDiscountFunction` は
共通の抽象基底クラス `BaseEconomicFunction`（`evaluate` / `get_expression` /
`has_closed_form` の命名規約）に準拠します。

- CRRA型（相対的リスク回避度一定） `CRRAUtility`（リスク中立ケース gamma=0 も許容）
- CARA型（絶対的リスク回避度一定） `CARAUtility`
- 指数割引 `ExponentialDiscounting`（時間整合的）
- 準双曲割引（beta-delta割引） `QuasiHyperbolicDiscounting`（beta<1で時間非整合）

`ExpectedUtilityAnalyzer` はくじ `Lottery`（状態別所得とその生起確率の組）を
効用関数で評価し、期待所得・期待効用・確実性等価（CE）・リスク・プレミアム（RP）・
絶対的/相対的リスク回避度（ARA/RRA、期待所得における評価値）を
`RiskAnalysisResult` として返します。確実性等価は効用の逆関数による解析解を
優先し、対応しない場合のみ `scipy.optimize.root_scalar` にフォールバックします。

`IntertemporalChoice` は割引関数と期間内効用関数を組み合わせ、消費ストリーム
`(c_0, c_1, ..., c_T)` の割引現在価値 `sum_t D(t) u(c_t)` と時間整合性を
`DiscountedUtilityResult` として返します。

## v0.4.0 スコープ

**消費者理論 (Consumer Theory)**・**生産者理論 (Producer Theory)**・
**市場均衡と厚生経済学 (Market Equilibrium & Welfare Economics)** の3領域を、
ポリモーフィックな設計のもとで統一的に扱います。各具象クラスが主問題・双対問題双方の
閉形式解（あるいは需要・供給関数）を自ら実装し、ファサードクラス
（`ConsumerProblem` / `ProducerProblem` / `MarketEquilibrium` / `WelfareAnalyzer` /
`TaxImpactAnalyzer`）はそれらへの委譲のみを行います（型ごとの
`if isinstance(...)` 分岐は持ちません）。

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

### 市場均衡・厚生経済学（v0.4.0で追加）

需要曲線・供給曲線から競争均衡を解き、税による市場の歪みを厚生（消費者余剰・
生産者余剰・政府収入・総余剰・死荷重）の観点から定量化します。

- 線形 `LinearDemandCurve` / `LinearSupplyCurve`
- 弾力性一定型 `ConstantElasticityDemandCurve` / `ConstantElasticitySupplyCurve`

`MarketEquilibrium` は需要・供給曲線の双方が閉形式解に対応する場合、両曲線の
SymPy記号表現を連立させて代数的に均衡を解きます（`is_closed_form=True`）。
いずれか一方でも対応しない場合のみ `scipy.optimize.root_scalar` による数値解法に
フォールバックします。

`WelfareAnalyzer` は初期化時に歪みのない競争均衡 `(P*, Q*)` を内部にキャッシュし、
これを死荷重 (DWL) 算出の参照点として用います。消費者余剰は弾力性一定型の需要曲線で
価格弾力性 `epsilon <= 1` の場合に理論通り発散し（`is_surplus_divergent`）、
`float("inf")` を返します。

`TaxImpactAnalyzer` は単位あたり従量税の課税による価格帰属（買い手・売り手の実効
負担割合と、局所弾力性 `eta / (epsilon + eta)` による予測値）を分析し、厚生評価は
内部に保持する `WelfareAnalyzer` へ委譲します（余剰計算ロジックの再実装は行いません）。

### 共通の設計方針

- 消費者理論・生産者理論の数値解は全て閉形式解によって決定論的に算出されます。
  市場均衡・厚生経済学モジュールは、両曲線が閉形式解に対応する場合はSymPyによる
  代数解法を、対応しない場合は`scipy.optimize`による数値解法を用いる二重構成を
  採ります。余剰・死荷重の積分評価には `scipy.integrate` を用います。
- `sympy` は途中式の解説（Markdown/LaTeX）の生成、各種補題の記号微分による理論検証、
  および市場均衡の代数解法にのみ使用され、消費者理論・生産者理論の数値計算そのもの
  には利用されません。
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

### 市場均衡・厚生経済学

```python
from microecon.market import LinearDemandCurve, LinearSupplyCurve, TaxImpactAnalyzer

demand = LinearDemandCurve(a=100.0, b=2.0)
supply = LinearSupplyCurve(c=-20.0, d=2.0)

analyzer = TaxImpactAnalyzer(demand, supply)

# 単位あたり従量税 t = 8 を課した場合の価格帰属・厚生への影響
result = analyzer.analyze_specific_tax(tax=8.0)
print(result.buyer_price, result.seller_price, result.taxed_quantity)
print(result.buyer_tax_share, result.elasticity_predicted_buyer_share)
print(result.welfare.deadweight_loss)
```

### 不確実性・リスク選好・動的選好

```python
from microecon.uncertainty import (
    CRRAUtility,
    ExpectedUtilityAnalyzer,
    IntertemporalChoice,
    Lottery,
    QuasiHyperbolicDiscounting,
)

# 期待効用理論に基づくリスク・プレミアムの評価
utility = CRRAUtility(gamma=1.0)  # 対数効用
lottery = Lottery(outcomes=((0.5, 100.0), (0.5, 400.0)))
result = ExpectedUtilityAnalyzer(utility).analyze(lottery)
print(result.certainty_equivalent, result.risk_premium)

# beta-delta 準双曲割引による異時点間消費計画の評価
discounting = QuasiHyperbolicDiscounting(beta=0.8, delta=0.95)
choice = IntertemporalChoice(discounting, CRRAUtility(gamma=0.0))
stream_result = choice.evaluate_stream((100.0, 100.0, 100.0))
print(stream_result.present_value, stream_result.is_time_consistent)
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

### v0.5.0 Scope

Adds **Uncertainty, Risk Preference & Dynamic Preference Analysis**: risk
metrics grounded in expected-utility theory, and intertemporal consumption
evaluation via discount functions, following the same polymorphic design as
the modules through v0.4.0.

#### Uncertainty, Risk & Dynamic Preference (new in v0.5.0)

`BaseBernoulliUtility` and `BaseDiscountFunction` both conform to the shared
abstract base `BaseEconomicFunction` (the `evaluate` / `get_expression` /
`has_closed_form` naming convention).

- CRRA (constant relative risk aversion): `CRRAUtility` (also covers the
  risk-neutral case, gamma=0)
- CARA (constant absolute risk aversion): `CARAUtility`
- Exponential discounting: `ExponentialDiscounting` (time-consistent)
- Quasi-hyperbolic (beta-delta) discounting: `QuasiHyperbolicDiscounting`
  (time-inconsistent when beta < 1)

`ExpectedUtilityAnalyzer` evaluates a `Lottery` (state-contingent payoffs and
their probabilities) under a given utility function, returning expected
wealth, expected utility, the certainty equivalent (CE), the risk premium
(RP), and absolute/relative risk aversion (ARA/RRA, evaluated at expected
wealth) as a `RiskAnalysisResult`. The certainty equivalent prefers the
closed-form inverse utility function and falls back to
`scipy.optimize.root_scalar` only when one isn't available.

`IntertemporalChoice` combines a discount function with a period utility
function to evaluate a consumption stream `(c_0, c_1, ..., c_T)`, returning
its discounted present value `sum_t D(t) u(c_t)` and time consistency as a
`DiscountedUtilityResult`.

### v0.4.0 Scope

**Consumer Theory**, **Producer Theory**, and **Market Equilibrium & Welfare
Economics** are all supported under a unified, polymorphic design. Each
concrete class implements the closed-form solutions (or demand/supply
functions) for its own problem, while the facade classes (`ConsumerProblem` /
`ProducerProblem` / `MarketEquilibrium` / `WelfareAnalyzer` /
`TaxImpactAnalyzer`) do nothing but delegate to them — no `if isinstance(...)`
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

#### Market Equilibrium & Welfare Economics (new in v0.4.0)

Solves the competitive equilibrium from a demand curve and a supply curve, and
quantifies how a tax distorts the market in terms of welfare (consumer
surplus, producer surplus, government revenue, total surplus, and deadweight
loss).

- Linear: `LinearDemandCurve` / `LinearSupplyCurve`
- Constant elasticity: `ConstantElasticityDemandCurve` /
  `ConstantElasticitySupplyCurve`

When both the demand and supply curves support a closed-form solution,
`MarketEquilibrium` solves for the equilibrium algebraically by equating their
SymPy symbolic expressions (`is_closed_form=True`). Only when either curve
lacks a closed form does it fall back to a numeric solution via
`scipy.optimize.root_scalar`.

`WelfareAnalyzer` caches the undistorted competitive equilibrium `(P*, Q*)` at
construction time and uses it as the reference point for the deadweight loss
(DWL) calculation. Consumer surplus correctly diverges to `float("inf")` for a
constant-elasticity demand curve whose price elasticity satisfies
`epsilon <= 1` (`is_surplus_divergent`), matching economic theory.

`TaxImpactAnalyzer` analyzes the incidence of a per-unit specific tax (the
buyer's and seller's effective tax shares, along with the value predicted by
the local elasticity formula `eta / (epsilon + eta)`), and delegates its
welfare evaluation to the `WelfareAnalyzer` it holds internally (it never
re-implements the surplus calculation logic).

#### Shared design principles

- All numerical results for consumer theory and producer theory are computed
  deterministically via closed-form solutions. The market equilibrium and
  welfare economics module uses a dual approach: an algebraic solution via
  SymPy when both curves support a closed form, and a numeric solution via
  `scipy.optimize` otherwise. Surplus and deadweight-loss integrals are
  evaluated with `scipy.integrate`.
- `sympy` is used to generate the Markdown/LaTeX explanation of the
  derivation, to verify the various lemmas via symbolic differentiation, and
  to solve market equilibria algebraically — it is never part of the
  numerical computation path for consumer theory or producer theory.
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

#### Market Equilibrium & Welfare Economics

```python
from microecon.market import LinearDemandCurve, LinearSupplyCurve, TaxImpactAnalyzer

demand = LinearDemandCurve(a=100.0, b=2.0)
supply = LinearSupplyCurve(c=-20.0, d=2.0)

analyzer = TaxImpactAnalyzer(demand, supply)

# Impact on price incidence and welfare of a per-unit specific tax t = 8
result = analyzer.analyze_specific_tax(tax=8.0)
print(result.buyer_price, result.seller_price, result.taxed_quantity)
print(result.buyer_tax_share, result.elasticity_predicted_buyer_share)
print(result.welfare.deadweight_loss)
```

#### Uncertainty, Risk & Dynamic Preference

```python
from microecon.uncertainty import (
    CRRAUtility,
    ExpectedUtilityAnalyzer,
    IntertemporalChoice,
    Lottery,
    QuasiHyperbolicDiscounting,
)

# Risk premium under expected-utility theory
utility = CRRAUtility(gamma=1.0)  # logarithmic utility
lottery = Lottery(outcomes=((0.5, 100.0), (0.5, 400.0)))
result = ExpectedUtilityAnalyzer(utility).analyze(lottery)
print(result.certainty_equivalent, result.risk_premium)

# Intertemporal consumption plan under beta-delta quasi-hyperbolic discounting
discounting = QuasiHyperbolicDiscounting(beta=0.8, delta=0.95)
choice = IntertemporalChoice(discounting, CRRAUtility(gamma=0.0))
stream_result = choice.evaluate_stream((100.0, 100.0, 100.0))
print(stream_result.present_value, stream_result.is_time_consistent)
```

### Development

```bash
poetry run mypy microecon
poetry run pytest
```
