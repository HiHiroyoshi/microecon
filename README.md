# microecon

ミクロ経済学の基礎理論を、型安全かつ直感的に計算・可視化できる教育・実務向け Python ライブラリです。

## v0.8.0 スコープ

**一般均衡理論（純粋交換経済・Edgeworth Box・Arrow-Debreu競争均衡）** を
`microecon.general_equilibrium` として追加しました。外部ライブラリには依存せず、
Pure Python + NumPy / SciPy + SymPy のみで実装しています。

### 一般均衡理論（v0.8.0で追加）

- `NGoodCobbDouglasUtility`: N 財コブ＝ダグラス効用 `U(x) = Π x_k^{α_k}`。既存の2財専用
  `BaseUtilityFunction` は継承せず、`typing.Protocol` による `UtilityProtocol`
  （`utility_variables` / `get_expression` / `evaluate` / `get_demand`）を満たします。
  指数は `sp.Rational` に変換され、記号計算で Float 係数が残りません。
- `BaseConsumer`: 効用関数と初期保有を統合し、需要・限界代替率（記号式・数値）を算出します。
- `ArrowDebreuEconomy`: M 消費者・N 財の純粋交換経済。基準財の価格を 1.0 に固定し、
  非基準財を対数価格 `p_k = exp(θ_k)` でパラメータ化して正値性を保証した上で、
  ワルラス法則により N-1 本の超過需要方程式を `scipy.optimize.root(method='hybr')`
  で解きます。市場清算誤差（超過需要の L2 ノルム）が `1e-6` を超える場合や、
  内点解の財で限界代替率と価格比が一致しない場合は `RuntimeError` を送出します。
- `EdgeworthBox`: 2消費者・2財モデル。`MRS_A = MRS_B` から契約曲線 `x2A = f(x1A)` を
  SymPy で導出し（`EdgeworthBox.x1A` で同一シンボルを参照可能）、競争均衡と合わせて
  `EdgeworthBoxResult` として返します。

結果は `CompetitiveEquilibriumResult`（価格・配分・効用・市場清算誤差）として返り、
配分・効用のマッピングは `MappingProxyType` により読み取り専用です。

## v0.7.0 スコープ

**一般形ゲーム解法・動的ゲーム・連続戦略ゲーム** を追加しました。外部のゲーム理論
ライブラリには依存せず、Pure Python + NumPy / SciPy / SymPy のみで実装しています。

### Lemke-Howson アルゴリズム（v0.7.0で追加）

`LemkeHowsonSolver` は任意サイズ `M x N` の双行列ゲームについて、線形相補性問題の
Tableau ピボット操作により1つのナッシュ均衡 `(p*, q*)` を求めます。

- 退化ゲームでの無限巡回を防ぐため、辞書式最小比テストで行選択を一意化
- 正値化定数 `k` は内部計算のみに使用し、期待利得は元の利得行列で `p^T A q`, `p^T B q` を評価
- 初期ラベルが `0 <= label < M + N` の範囲外なら `ValueError`、得られた解が最適反応条件
  （支持集合 `> 1e-6`、許容誤差 `1e-4`）を満たさなければ `RuntimeError`

`NashSolver.find_mixed_nash` は `2 x 2` ゲームでは解析解で完全混合戦略
（`0 < p, q < 1`）を、それ以外のサイズでは全初期ラベルの Lemke-Howson 探索を用い、
純粋戦略を除外・重複排除（許容誤差 `1e-4`）した混合戦略均衡を返します
（v0.6.0 の `NotImplementedError` は廃止）。純粋戦略均衡は `pure_nash_equilibria` にのみ
格納されます。

> **注意**: Lemke-Howson アルゴリズムが保証するのは「少なくとも1つのナッシュ均衡」の
> 発見であり、全ナッシュ均衡の完全列挙は保証しません（例: 男女の争いの完全混合均衡には
> どの初期ラベルからも到達しません）。均衡が連続体（線分等）となる退化ゲームでは、
> 到達した連続体の代表点（頂点解）のみを返却します。

### 連続戦略ゲーム（v0.7.0で追加）

`ContinuousGame` は SymPy 記号式の利得関数を持つ2人連続戦略ゲームを表現します。
`get_best_response_expression` は FOC を解き SOC を満たす最適反応関数を返し、
`find_pure_nash` は内点 FOC 解と有限な境界（無限大は除外）の全組み合わせを候補として
相互最適反応を検証し、角点解（`is_corner=True`）を含む全均衡を返します。

### 展開型ゲーム・後方帰納法（v0.7.0で追加）

`ExtensiveFormGame` は完備情報の動的ゲーム（ゲーム木）を表現し、構築時に厳格な木構造
バリデーション（ルートの存在、親が厳密に1つ、全ノードへの到達可能性、プレイヤー
インデックス・利得長・行動と子ノードの一致）を行います。`solve_spne` は後方帰納法により
部分ゲーム完全ナッシュ均衡を求め、`tie_breaking_rule="first"` では無差別な行動のうち
`actions` の先頭を選択します。結果は `MappingProxyType` により読み取り専用です。

## v0.6.0 スコープ

**戦略型ゲーム・ナッシュ均衡分析 (Strategic-Form Games & Nash Equilibrium)** を
追加しました。2人有限戦略型ゲームにおける支配戦略の判定、純粋・混合戦略ナッシュ均衡の
算出を、v0.5.0までと同様の不変（immutable）データクラスによる結果表現のもとで提供します。

### 戦略型ゲーム・ナッシュ均衡（v0.6.0で追加）

`NormalFormGame` は利得行列のペア `(A, B)` によって2人有限戦略型ゲームを表現し、
任意の相手戦略に対して厳密に（あるいは弱く）優る厳密支配戦略・弱支配戦略の判定
（許容誤差 `atol=1e-6`）、ゼロサムゲーム判定（`A + B = 0`）を提供します。

`NashSolver` は以下の2種類のアルゴリズムでナッシュ均衡を算出します。

- 純粋戦略ナッシュ均衡: 最適反応（Best Response）比較により、任意の `m x n`
  サイズのゲームに対応（`find_pure_nash`）
- 混合戦略ナッシュ均衡: v0.6.0時点では `2 x 2` ゲームの無差別条件による解析解のみ
  対応（`find_mixed_nash`）。対応しないサイズでは `NotImplementedError` を送出し
  （N x M ゲームへの一般化はv0.7.0でLemke-Howsonアルゴリズムにより対応予定）、
  `solve()` はこれを内部で捕捉して `mixed_nash_equilibria = ()` にフォールバック
  します。

結果を保持する `MixedStrategyProfile` の確率分布ベクトル（NumPy配列）は、複製の上で
`writeable = False` 化されており、誤った書き換えを防ぎます。

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

### 戦略型ゲーム・ナッシュ均衡

```python
import numpy as np

from microecon.game_theory import NashSolver, NormalFormGame

# 囚人のジレンマ
payoff_a = np.array([[-1.0, -3.0], [0.0, -2.0]])
payoff_b = np.array([[-1.0, 0.0], [-3.0, -2.0]])
game = NormalFormGame(payoff_matrices=(payoff_a, payoff_b))

print(game.has_dominant_strategy(0, strict=True))  # True（裏切りが支配戦略）
print(game.get_dominant_strategy(0, strict=True))  # 1

result = NashSolver(game).solve()
print(result.pure_nash_equilibria[0].strategy_indices)  # (1, 1)
print(result.pure_nash_equilibria[0].payoffs)  # (-2.0, -2.0)

# コインの表裏合わせ（ゼロサム・混合戦略ナッシュ均衡）
matching_pennies = NormalFormGame(
    payoff_matrices=(
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.array([[-1.0, 1.0], [1.0, -1.0]]),
    )
)
mixed_result = NashSolver(matching_pennies).solve()
print(mixed_result.is_zero_sum)  # True
print(mixed_result.mixed_nash_equilibria[0].probabilities)  # (array([0.5, 0.5]), array([0.5, 0.5]))
```

### 一般形ゲーム・連続戦略ゲーム・展開型ゲーム

```python
import numpy as np
import sympy as sp

from microecon.game_theory import (
    ContinuousGame,
    ExtensiveFormGame,
    ExtensiveFormNode,
    LemkeHowsonSolver,
)

# Lemke-Howson（コインの表裏合わせ）
matrix_a = np.array([[1.0, -1.0], [-1.0, 1.0]])
solver = LemkeHowsonSolver(matrix_a, -matrix_a)
p, q = solver.solve(initial_dropped_label=0)
print(p, q, solver.expected_payoffs(p, q))  # [0.5 0.5] [0.5 0.5] ≈(0.0, 0.0)

# Cournot 複占（角点解）
q1, q2 = sp.symbols("q1 q2")
price = 100 - (q1 + q2)
cournot = ContinuousGame(
    ((price - 10) * q1, (price - 120) * q2), (q1, q2), ((0.0, float("inf")),) * 2
)
print(cournot.find_pure_nash())  # strategies=(45.0, 0.0), is_corner=True

# 市場参入ゲーム（後方帰納法）
nodes = {
    "root": ExtensiveFormNode("root", 0, ("Enter", "Out"), {"Enter": "n_enter", "Out": "n_out"}),
    "n_out": ExtensiveFormNode("n_out", None, (), payoffs=(0.0, 300.0)),
    "n_enter": ExtensiveFormNode(
        "n_enter", 1, ("Fight", "Acquiesce"), {"Fight": "n_fight", "Acquiesce": "n_acq"}
    ),
    "n_fight": ExtensiveFormNode("n_fight", None, (), payoffs=(-100.0, 0.0)),
    "n_acq": ExtensiveFormNode("n_acq", None, (), payoffs=(100.0, 100.0)),
}
spne = ExtensiveFormGame("root", nodes).solve_spne()
print(dict(spne.equilibrium_strategies[1]), spne.payoffs)  # {'n_enter': 'Acquiesce'} (100.0, 100.0)
```

### 一般均衡理論（Edgeworth Box・Arrow-Debreu）

```python
from microecon.general_equilibrium import (
    ArrowDebreuEconomy,
    BaseConsumer,
    EdgeworthBox,
    NGoodCobbDouglasUtility,
)

# 2財2消費者の Edgeworth Box
consumer_a = BaseConsumer("A", NGoodCobbDouglasUtility((0.6, 0.4)), (8.0, 2.0))
consumer_b = BaseConsumer("B", NGoodCobbDouglasUtility((0.2, 0.8)), (2.0, 8.0))
box = EdgeworthBox(consumer_a, consumer_b)
result = box.analyze(numeraire_idx=0)
print(result.contract_curve_expression)  # -2*x1A/(x1A - 12)
print(result.competitive_equilibrium.prices)  # ≈(1.0, 1.7143)
print(dict(result.competitive_equilibrium.allocations))  # A ≈ (6.857, 2.667), B ≈ (3.143, 7.333)

# 3財の Arrow-Debreu 経済
utility = NGoodCobbDouglasUtility((0.5, 0.3, 0.2))
economy = ArrowDebreuEconomy(
    (
        BaseConsumer("A", utility, (10.0, 5.0, 0.0)),
        BaseConsumer("B", utility, (0.0, 5.0, 10.0)),
    )
)
equilibrium = economy.solve(numeraire_idx=0)
print(equilibrium.prices)  # ≈(1.0, 0.6, 0.4)
print(equilibrium.market_clearing_error < 1e-6)  # True
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

### v0.8.0 Scope

Adds **general equilibrium theory (pure exchange economies, the Edgeworth box,
and Arrow-Debreu competitive equilibrium)** as `microecon.general_equilibrium`,
implemented with pure Python + NumPy / SciPy + SymPy only.

#### General Equilibrium (new in v0.8.0)

- `NGoodCobbDouglasUtility`: N-good Cobb-Douglas utility `U(x) = Π x_k^{α_k}`.
  Rather than inheriting the two-good `BaseUtilityFunction`, it satisfies the
  `typing.Protocol`-based `UtilityProtocol` (`utility_variables` /
  `get_expression` / `evaluate` / `get_demand`). Exponents are converted to
  `sp.Rational`, so no Float coefficients remain in symbolic results.
- `BaseConsumer`: combines a utility function with an endowment and computes
  demand and the marginal rate of substitution (symbolic and numeric).
- `ArrowDebreuEconomy`: M-consumer, N-good pure exchange economy. The numeraire
  price is fixed at 1.0 and the other prices are parameterized as
  `p_k = exp(θ_k)` to guarantee positivity; by Walras' law, the N-1 excess
  demand equations are solved with `scipy.optimize.root(method='hybr')`.
  Raises `RuntimeError` if the market-clearing error (L2 norm of excess demand)
  exceeds `1e-6` or if MRS and price ratios disagree for interior goods.
- `EdgeworthBox`: two-consumer, two-good model. Derives the contract curve
  `x2A = f(x1A)` from `MRS_A = MRS_B` with SymPy (the shared symbol is exposed
  as `EdgeworthBox.x1A`) and returns it with the competitive equilibrium as an
  `EdgeworthBoxResult`.

Results are returned as `CompetitiveEquilibriumResult` (prices, allocations,
utilities, market-clearing error); the allocation and utility mappings are
read-only via `MappingProxyType`.

### v0.7.0 Scope

Adds **general normal-form game solving, dynamic games, and continuous-strategy
games**, implemented with pure Python + NumPy / SciPy / SymPy only (no external
game-theory libraries).

#### Lemke-Howson Algorithm (new in v0.7.0)

`LemkeHowsonSolver` finds one Nash equilibrium `(p*, q*)` of any `M x N`
bimatrix game via complementary pivoting on LCP tableaux.

- A lexicographic minimum ratio test makes row selection unique, preventing
  cycling in degenerate games
- The positivity shift `k` is internal only; expected payoffs are evaluated on
  the original matrices as `p^T A q`, `p^T B q`
- Raises `ValueError` for an initial label outside `0 <= label < M + N`, and
  `RuntimeError` if the result fails the best-response check (support
  `> 1e-6`, tolerance `1e-4`)

`NashSolver.find_mixed_nash` uses the closed-form solution for fully mixed
equilibria (`0 < p, q < 1`) in `2 x 2` games, and Lemke-Howson from every
initial label otherwise, returning only non-pure, de-duplicated (tolerance
`1e-4`) profiles (the v0.6.0 `NotImplementedError` is removed). Pure equilibria
appear only in `pure_nash_equilibria`.

> **Note**: Lemke-Howson guarantees finding *at least one* Nash equilibrium, not
> all of them (e.g. no initial label reaches the fully mixed equilibrium of the
> Battle of the Sexes). For degenerate games with a continuum of equilibria,
> only the representative vertex reached is returned.

#### Continuous-Strategy Games (new in v0.7.0)

`ContinuousGame` models two-player games with SymPy payoff expressions.
`get_best_response_expression` solves the FOC and keeps the SOC-satisfying
solution; `find_pure_nash` enumerates interior FOC solutions and finite bounds
(infinite bounds are excluded), verifies mutual best responses, and returns all
equilibria, including corner solutions (`is_corner=True`).

#### Extensive-Form Games & Backward Induction (new in v0.7.0)

`ExtensiveFormGame` models perfect-information dynamic games (game trees) with
strict tree validation at construction (root exists, exactly one parent per
non-root node, all nodes reachable, valid player indices, payoff lengths, and
actions matching children). `solve_spne` derives the subgame-perfect Nash
equilibrium by backward induction; `tie_breaking_rule="first"` picks the
earliest action among ties. Results are read-only via `MappingProxyType`.

### v0.6.0 Scope

Adds **Strategic-Form Games & Nash Equilibrium**: dominant-strategy detection
and pure/mixed Nash equilibrium computation for two-player finite normal-form
games, with results represented via the same immutable dataclass style used
through v0.5.0.

#### Strategic-Form Games & Nash Equilibrium (new in v0.6.0)

`NormalFormGame` represents a two-player finite normal-form game via a pair of
payoff matrices `(A, B)`, and detects strict and weak dominant strategies
(tolerance `atol=1e-6`) as well as zero-sum games (`A + B = 0`).

`NashSolver` computes Nash equilibria via two algorithms:

- Pure-strategy Nash equilibria: enumerated via best-response comparison,
  supporting any `m x n` game size (`find_pure_nash`)
- Mixed-strategy Nash equilibria: as of v0.6.0, only `2 x 2` games are
  supported, via the closed-form indifference-condition solution
  (`find_mixed_nash`). Other sizes raise `NotImplementedError` (general `N x M`
  support is planned for v0.7.0 via the Lemke-Howson algorithm), which
  `solve()` catches internally and falls back to
  `mixed_nash_equilibria = ()`.

The probability vectors (NumPy arrays) held by `MixedStrategyProfile` are
copied and marked `writeable = False`, preventing accidental mutation.

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

#### Strategic-Form Games & Nash Equilibrium

```python
import numpy as np

from microecon.game_theory import NashSolver, NormalFormGame

# Prisoner's Dilemma
payoff_a = np.array([[-1.0, -3.0], [0.0, -2.0]])
payoff_b = np.array([[-1.0, 0.0], [-3.0, -2.0]])
game = NormalFormGame(payoff_matrices=(payoff_a, payoff_b))

print(game.has_dominant_strategy(0, strict=True))  # True (Defect dominates)
print(game.get_dominant_strategy(0, strict=True))  # 1

result = NashSolver(game).solve()
print(result.pure_nash_equilibria[0].strategy_indices)  # (1, 1)
print(result.pure_nash_equilibria[0].payoffs)  # (-2.0, -2.0)

# Matching Pennies (zero-sum, mixed-strategy Nash equilibrium)
matching_pennies = NormalFormGame(
    payoff_matrices=(
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.array([[-1.0, 1.0], [1.0, -1.0]]),
    )
)
mixed_result = NashSolver(matching_pennies).solve()
print(mixed_result.is_zero_sum)  # True
print(mixed_result.mixed_nash_equilibria[0].probabilities)  # (array([0.5, 0.5]), array([0.5, 0.5]))
```

#### General Games, Continuous-Strategy Games & Extensive-Form Games

```python
import numpy as np
import sympy as sp

from microecon.game_theory import (
    ContinuousGame,
    ExtensiveFormGame,
    ExtensiveFormNode,
    LemkeHowsonSolver,
)

# Lemke-Howson (Matching Pennies)
matrix_a = np.array([[1.0, -1.0], [-1.0, 1.0]])
solver = LemkeHowsonSolver(matrix_a, -matrix_a)
p, q = solver.solve(initial_dropped_label=0)
print(p, q, solver.expected_payoffs(p, q))  # [0.5 0.5] [0.5 0.5] ≈(0.0, 0.0)

# Cournot duopoly (corner solution)
q1, q2 = sp.symbols("q1 q2")
price = 100 - (q1 + q2)
cournot = ContinuousGame(
    ((price - 10) * q1, (price - 120) * q2), (q1, q2), ((0.0, float("inf")),) * 2
)
print(cournot.find_pure_nash())  # strategies=(45.0, 0.0), is_corner=True

# Market entry game (backward induction)
nodes = {
    "root": ExtensiveFormNode("root", 0, ("Enter", "Out"), {"Enter": "n_enter", "Out": "n_out"}),
    "n_out": ExtensiveFormNode("n_out", None, (), payoffs=(0.0, 300.0)),
    "n_enter": ExtensiveFormNode(
        "n_enter", 1, ("Fight", "Acquiesce"), {"Fight": "n_fight", "Acquiesce": "n_acq"}
    ),
    "n_fight": ExtensiveFormNode("n_fight", None, (), payoffs=(-100.0, 0.0)),
    "n_acq": ExtensiveFormNode("n_acq", None, (), payoffs=(100.0, 100.0)),
}
spne = ExtensiveFormGame("root", nodes).solve_spne()
print(dict(spne.equilibrium_strategies[1]), spne.payoffs)  # {'n_enter': 'Acquiesce'} (100.0, 100.0)
```

#### General Equilibrium (Edgeworth Box & Arrow-Debreu)

```python
from microecon.general_equilibrium import (
    ArrowDebreuEconomy,
    BaseConsumer,
    EdgeworthBox,
    NGoodCobbDouglasUtility,
)

# Two-good, two-consumer Edgeworth box
consumer_a = BaseConsumer("A", NGoodCobbDouglasUtility((0.6, 0.4)), (8.0, 2.0))
consumer_b = BaseConsumer("B", NGoodCobbDouglasUtility((0.2, 0.8)), (2.0, 8.0))
box = EdgeworthBox(consumer_a, consumer_b)
result = box.analyze(numeraire_idx=0)
print(result.contract_curve_expression)  # -2*x1A/(x1A - 12)
print(result.competitive_equilibrium.prices)  # ≈(1.0, 1.7143)
print(dict(result.competitive_equilibrium.allocations))  # A ≈ (6.857, 2.667), B ≈ (3.143, 7.333)

# Three-good Arrow-Debreu economy
utility = NGoodCobbDouglasUtility((0.5, 0.3, 0.2))
economy = ArrowDebreuEconomy(
    (
        BaseConsumer("A", utility, (10.0, 5.0, 0.0)),
        BaseConsumer("B", utility, (0.0, 5.0, 10.0)),
    )
)
equilibrium = economy.solve(numeraire_idx=0)
print(equilibrium.prices)  # ≈(1.0, 0.6, 0.4)
print(equilibrium.market_clearing_error < 1e-6)  # True
```

### Development

```bash
poetry run mypy microecon
poetry run pytest
```
