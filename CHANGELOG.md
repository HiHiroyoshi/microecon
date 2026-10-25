# Changelog

v0.8.0 以前の変更内容は README の各スコープ節を参照してください。

## [0.9.0]

### Added

- `microecon.industrial_organization`: 産業組織論モジュールを追加。
  - `Monopoly`: 単一価格独占（MR = MC、SOC、CS・PS・利潤・DWL・ラーナー指数）と
    第1種（完全）価格差別。
  - `ThirdDegreePriceDiscrimination`: 第3種価格差別（費用は総生産量の関数、
    ヘッセ行列による SOC）。
  - `CournotDuopoly`: クールノー複占（`ContinuousGame` へのコンポジションによる委譲）。
  - `StackelbergDuopoly`: シュタッケルベルク複占（追従者の非負制約付き反応関数を
    区分的に扱う後方帰納法）。
  - `DifferentiatedBertrand`: 差別化財ベルトラン複占（可積分なアフィン需要に限り、
    代表的消費者の二次効用から消費者余剰を定義）。
  - `HomogeneousBertrand`: 同質財ベルトラン複占の解析解（対称・費用優位・
    抜本的費用優位）。
  - `compare_market_structures`: 完全競争・ベルトラン・シュタッケルベルク・
    クールノー・独占の比較表（`MarketStructureComparison.to_markdown_table()`）。
  - `IOPlotter`: 独占の余剰図・反応曲線（`matplotlib.figure.Figure` を返す）。
  - 全結果クラスに導出過程の解説 `markdown_steps` を付与。
- `tests/test_industrial_organization.py`: 解析解のアンカー値による理論整合性テスト。

### Notes

- 自然独占、第3種価格差別の市場閉鎖、第2種価格差別、3社以上の寡占などは対象外
  （README の「対象外」を参照）。
