# LogicAD Figure 2 / Figure 3 実行パイプライン

既存anomalibのモデル登録を通さず、LOCO画像からテキスト特徴・JSON・異常スコアを
生成する追加実装です。学習は行わず、カテゴリごとに正常参照画像1枚を使用します。
論文Table 1の数値を再現済みとするものではありません。

実装前の調査と対応表は [docs/FIG2_RECONSTRUCTION.md](docs/FIG2_RECONSTRUCTION.md) にあります。

## Figure 2/3との対応

```text
normal image                         query image
    |                                     |
    +-- original + GroundingDINO ROI ------+
    |                                     |
Guided category observations (GPT-4o, K=3 independent requests)
    |                                     |
text-embedding-3-large -> LOF -> seeded random surviving description
    |                                     |
    +------------- two branches ----------+
    |
    +-- Format Embedding: LLM JSON -> text embedding -> L2 normalization
    |                     score = 1 - dot(normal, query)
    |
    +-- optional Logic Reasoner: LLM formalization -> normal/query formulae
                          -> naming/functionality/domain closure/defaults
                          -> Prover9 / Mace4 -> verdict + explanation
```

参照画像とqueryはそれぞれ独立に特徴抽出します。正常画像の内容やラベルをqueryの
観察プロンプトへ混ぜません。ROI画像は元画像の詳細として同一リクエストへ渡します。

| モジュール | 役割 |
| --- | --- |
| `prompts.py` | Feature / Extraction / Format / Logicの4種を集中管理、5カテゴリのJSON schema |
| `roi.py` | 既存 `load_gdino_model` の再利用、cuda/cpu自動選択、bboxからcrop |
| `openai_client.py` | 通常のOpenAI API、timeout・backoff・usage記録、mock差し替え |
| `text_extraction.py`, `filtering.py` | K記述、段階cache、埋め込み、LOF、選択結果 |
| `format_embedding.py`, `embeddings.py` | JSON検証・正規化・埋め込み・コサインスコア |
| `dataset.py`, `evaluator.py` | LOCO探索、参照選択、繰り返し評価、再開、CSV/JSON |
| `logic_syntax.py`, `logic_reasoner.py` | 形式文法検証、公理、ATP、矛盾部分集合 |

## Linux / DLBox setup

Python 3.10を基本対象とします。新しい専用venvを使い、旧anomalibの依存関係とは
分けてインストールしてください。`pip install -e .` や旧 `requirements.txt` は不要です。

```bash
cd logicAD_
python3.10 -m venv .venv-fig2
source .venv-fig2/bin/activate
python -m pip install -r requirements_logicad_fig2.txt
export OPENAI_API_KEY='your-own-api-key'
python scripts/run_logicad_fig2.py --help
```

キーは `OPENAI_API_KEY` のみから取得します。既存の `keys/` ファイルやAzureの設定は
使用しません。GPT-4oとembeddingはAPIで実行するため、大容量GPUは不要です。
APIキー未設定でも、dry-run、offlineテスト、必要なcacheが全部揃った再実行は可能です。

### GroundingDINO（任意だがFigure 3再現には必要）

DLBoxのCUDAと互換性のあるtorch / torchvisionを
[PyTorch公式手順](https://pytorch.org/get-started/locally/)で導入した後、
[GroundingDINO公式リポジトリ](https://github.com/IDEA-Research/GroundingDINO)をインストールします。
CUDA拡張を使用する場合はCUDA toolkitとビルドツールも必要です。

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git vendor/GroundingDINO
python -m pip install -e vendor/GroundingDINO
python -m pip install omegaconf
```

checkpointは公式リポジトリの案内に従って取得し、引数で指定してください。

```bash
python scripts/run_logicad_fig2.py \
  --data-root /data/mvtec_loco_anomaly_detection \
  --category breakfast_box --max-images 10 \
  --gdino-checkpoint /models/groundingdino_swint_ogc.pth \
  --gdino-config swint --device auto
```

`--gdino-config swinb` も対応します。`auto` はtorchのCUDA可用性を確認します。
checkpoint未指定・モデル未導入・検出失敗・ROIゼロでも元画像で継続し、理由を保存します。
checkpointを指定したのに失敗した場合はwarningが出ます。`--disable-roi` は意図的に
ROIを無効化するablation用です。ROIなしの結果を完全なFigure 3再現として扱わないでください。

標準のbox/text閾値は0.2/0.3、pushpinsは0.1/0.1です。
`--box-threshold` / `--text-threshold` で上書きできます。
cropはbbox中心から `--roi-padding 1.5` 倍、境界でclipし、信頼度順で最大
`--max-rois 32` 個です。既存ローダーは絶対checkpointパスを受け取り、モデル登録を
経由せず単体ロードします。

## データ配置

```text
/data/mvtec_loco_anomaly_detection/
  breakfast_box/
    train/good/*.png
    test/good/*.png
    test/logical_anomalies/*.png
  juice_bottle/...
  pushpins/...
  screw_bag/...
  splicing_connectors/...
```

データ自体は [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco) から取得します。
PNG/JPEG/BMP/TIFF/WebPを認識します。上記root、カテゴリディレクトリ、または
`mvtec_loco_anomaly_detection/`・`original/`・`MVTec_Loco/original/` を含む親を指定できます。
複数候補がある場合は曖昧な選択をせずエラーにします。
`test/structural_anomalies`、validation、pixel maskはこの評価には使用しません。

## 実行

APIを呼ばない事前確認:

```bash
python scripts/run_logicad_fig2.py \
  --data-root /data/mvtec_loco_anomaly_detection \
  --category all --num-runs 5 --max-images 10 --dry-run
```

1カテゴリ・少数画像で実APIを確認:

```bash
python scripts/run_logicad_fig2.py \
  --data-root /data/mvtec_loco_anomaly_detection \
  --category breakfast_box --reference-index 0 --max-images 10 \
  --vlm-model gpt-4o --embedding-model text-embedding-3-large \
  --gdino-checkpoint /models/groundingdino_swint_ogc.pth
```

1カテゴリの全test画像:

```bash
python scripts/run_logicad_fig2.py \
  --data-root /data/mvtec_loco_anomaly_detection \
  --category breakfast_box --reference-index 0 \
  --gdino-checkpoint /models/groundingdino_swint_ogc.pth
```

全5カテゴリ・one-shotを5回:

```bash
python scripts/run_logicad_fig2.py \
  --data-root /data/mvtec_loco_anomaly_detection \
  --category all --num-runs 5 --seed 42 \
  --gdino-checkpoint /models/groundingdino_swint_ogc.pth
```

`--reference-index` は名前順に並べた `train/good` の0始まりindexです。複数runでは
指定indexを最初にし、残りはseedによる重複なし抽出です。未指定なら全runをseedで選びます。
参照数が不足すればAPI呼び出し前にエラーにします。queryの特徴はrun間でも再利用します。
`--max-images` はカテゴリあたりのtest画像数上限で、goodとlogicalを交互に選びます。
1にするとラベルが片方だけなのでAUROCはnullになります。

主な調整引数:

```text
--k 3 --temperature 0.05 --top-p 0.1 --lof-neighbors 2
--format-model gpt-4o --logic-model gpt-4o --max-tokens 1600
--timeout 90 --max-retries 4 --json-retries 2
--output-dir outputs
```

## Logic Reasoner

Prover9とMace4は [LADR公式配布](https://www.cs.unm.edu/~mccune/prover9/download/) の
command-line版を使います。Linuxではソースを展開したディレクトリで `make all` し、
`bin` をPATHへ追加するか、実行ファイルのパスを指定します。

```bash
python scripts/run_logicad_fig2.py \
  --data-root /data/mvtec_loco_anomaly_detection \
  --category breakfast_box --enable-reasoner \
  --prover9-path /opt/LADR-2009-11A/bin/prover9 \
  --mace4-path /opt/LADR-2009-11A/bin/mace4 \
  --gdino-checkpoint /models/groundingdino_swint_ogc.pth
```

通常は正常参照のselected natural languageをLLMで形式化し、queryを別に形式化します。
`--normal-rules legacy` はリポジトリに残ったbreakfast/juice/connectorsの正常規則を使う
比較設定です。この設定では正常規則がone-shot画像だけに由来しない点に注意してください。
pushpins/screw_bagには元の正常規則がないためlegacy設定のreasonerはskipします。

`Γ = Σnorm ∪ Σna ∪ Σfa ∪ Σdca` を構成し、欠落countを0で補完した `Σ0` に対して
Prover9で `Γ |= ¬Σ0` を確認します。既存コードのu/b/f/c規約、同義語の正規化、
機能性とdomain closureを再利用しています。LLMが返した生テキストをプログラムとして
実行せず、述語名・arity・文法を検証したASTからATP入力を組み立てます。

判定は次のとおりです。

- `abnormal`: 正常公理の整合性をモデルで確認でき、queryとの矛盾が証明された。
- `normal`: Mace4が正常公理とqueryを同時に満たす有限モデルを発見した。
- `unknown`: 時間制限、構文/実行エラー、明示的なunknown値、正常公理の不整合や整合性未確認。
- `skipped`: Prover9未導入など。Format Embedding評価は継続する。

未証明を正常とは扱いません。Mace4がない場合は正常公理の整合性を保証できず、
保守的にunknownとなります。両方の実行ファイルを設定するのが推奨です。
`--prover-timeout 10` は1回のATP呼び出しあたりの制限で、画像全体の制限ではありません。

異常時にはqueryの各式を削除して矛盾が残るかを調べ、queryに対して包含関係で極小な
矛盾集合を求めます。最小要素数の集合を保証するものではありません。
`--max-mis-checks 64` やタイムアウトで探索が不完全なら `minimality_certified=false` とし、
証明済みの矛盾集合だけを説明します。説明文・使った式・証明/モデルファイルを保存します。
Reasoner判定はFormat Embeddingスコアと混ぜず、別の出力欄に記録します。

## Cache / 途中再開 / 出力

```text
outputs/
  api_usage.jsonl
  cache/
    <category>/<image-stem>-<image-hash>/<extraction-settings-hash>/
      descriptions.json              # 各K応答のraw response、ROI metadata
      selection-<hash>.json           # 採用index、LOF scores、inliers、fallback
    embeddings/<hash>.json
    embeddings/<hash>.attempts.json
    formatted/<category>/<hash>.json
    formatted/<category>/<hash>.attempts.json
    references/<category>/<hash>.json
    formal/<category>/<hash>.json
    reasoning/<category>/<hash>.json
    prover_calls/<hash>.json          # raw stdout/stderr/return code
    prover_calls/<hash>.in
    prover_calls/<hash>.out
  results/<experiment-hash>/
    manifest.json                    # 全引数・prompt・画像SHA256・参照選択
    records/<category>/<run>/<hash>.json
    predictions.json / predictions.csv
    runs.json / runs.csv
    summary.json / summary.csv
    errors.json
```

画像内容とパス、prompt、モデル、温度などに応じたstage keyで保存し、異なる実験の
cache混入を防ぎます。通常の再実行は成功したjobを再利用し、失敗jobを再試行します。
途中終了しても、各記述・各画像の保存済み結果から再開できます。

同じコマンドに `--retry-errors` を付けると、既存の同一実験があることを確認してから
失敗分を再実行します。成功分は集計に残ります。引数・画像・参照を変えた場合は別実験です。
`--force` は各stageを1回再生成します。同じ参照・queryをrunごとに再課金しません。
この2つのフラグは同時使用できません。

生成済みcacheはJSON envelopeの `value` 内にあります。旧 `datasets/*/img2text*.json` は
K・モデル・prompt等の来歴が不足するため、自動でこのcacheへ混ぜません。
ROI導入前のoriginal-only結果を同じ設定で再利用したくない場合は `--force` を指定します。
Python処理のversionを変更するときはstage/manifestのversionも更新してください。
cacheと結果は単一writer用です。同じoutput-dirに複数プロセスを同時実行しないでください。

APIはtimeout・429/408/409/5xx・接続エラーに対し指数backoffで再試行し、Retry-Afterを
尊重します。SDK内蔵retryは重複課金の見積もりを分かりやすくするため無効にしています。
成功呼び出し数・試行回数・返されたusageをログに残します。サーバー受理後の通信切断は
再試行で課金が重複する可能性があり、完全なexactly-onceは保証しません。

整形はStructured Outputsを優先し、未対応のsnapshotではJSON modeへ切り替えます。
parse/schema失敗は修復リクエストを行い、すべてのraw responseを保存します。
修復できない場合、特徴を捏造せず画像をerrorにし、後から再実行可能にします。

## 指標の読み方

`predictions` のscoreは大きいほど異常です。数学的な値域は `[0,2]` で、
論文図の `[0,1]` 表示に合わせるための追加clippingは行いません。
AUROC/F1-maxは `[0,1]` 表記なので、表の%と比較する場合は100倍してください。
F1-maxは評価ラベルを使ったthreshold sweepによる報告指標です。実運用の閾値は別途必要です。

`runs` はカテゴリ・参照ごとの指標、`summary` はカテゴリごとのmean/stdです。
stdは母標準偏差 `ddof=0` です。macroは各runのカテゴリ平均を取り、そのmean/stdを算出します。
5カテゴリ指定時のみ `is_five_category_average=true` になります。
処理失敗が残るrunは `complete=false` で、成功画像だけの参考指標を `runs` に残しますが、
そのカテゴリ/全カテゴリの確定平均には使いません。片ラベルのAUROCはnullです。
終了コード0はA+B処理完了、2は設定/画像処理エラーです。Reasonerのunknown/skipは別管理です。

## テストと現時点の実行確認

```bash
python -m pytest tests/fig2 -q
python scripts/smoke_logicad_fig2.py --output-dir outputs/offline_smoke
```

実ATPの検証テストもあります。PATHまたは以下の環境変数を設定すると、欠落物の
矛盾証明と整合queryのモデル生成を実バイナリで確認します。未設定ならこの2件はskipします。

```bash
PROVER9_TEST_PATH=/opt/LADR-2009-11A/bin/prover9 \
MACE4_TEST_PATH=/opt/LADR-2009-11A/bin/mace4 \
python -m pytest tests/fig2/test_atp_integration.py -q
```

offline smokeは合成画像と模擬APIで全5カテゴリ×5参照を実行し、再開時にAPI呼び出しが
増えないことを確認します。出力には `SMOKE_TEST_ONLY.json` が付きます。
そのAUROC/F1は実データ性能を表しません。実APIに切り替えるには `run_logicad_fig2.py` を使います。

この実装作業ではWindows / Python 3.12専用venvで45テストが通過（実ATPの2件はskip）し、5カテゴリ×5runの
offline smoke（100 image/reference pair、模擬API170回、再開時追加0回）を実行しました。
Python 3.10の文法互換チェックも実施しました。LinuxのPython 3.10/3.12向けCI定義を
追加しましたが、まだGitHub上では実行していません。

OPENAI_API_KEY、LOCO実画像、GroundingDINO checkpoint、Prover9/Mace4実行ファイルが
作業環境にないため、GPT-4o実応答、ROI実検出、実ATP、DLBoxでの実行と論文の実測性能は
未確認です。API/ROI/ATPは差し替え可能な境界でテストしました。

## Reconstruction assumptions / 復元できなかった部分

1. **Extraction prompt**: 原本の `TEXT_EXTRACTOR_PROMPTS` が欠落しています。
   repo notebook・中間JSON・補足A.1を参考に観察項目を再構成しました。文言の完全一致ではありません。
   breakfastではapple/nectarineを同一視せず、果物選択肢と個数を保持します。
2. **Detector prompt**: 補足A.4のconnector行はjuice bottleを指しています。
   この実装は `connector block` に修正します。元の表の値は
   `--feature-prompt 'fruit juice bottle'` で指定できます。
3. **Image handling**: 元画像＋複数cropを一度のmultimodal requestへ渡します。
   crop padding/上限/重複bbox除外は実装上の仮定です。元の全patch集約コードは欠落しています。
4. **Stochastic generation**: K回独立呼び出しでtemperature/top_pを固定します。
   `--seed` は参照選択とLOF後の乱択を固定し、API出力自体の再現性は保証しません。
   作業時の既定は公共APIのgpt-4oで、補足のAzure 2024-05-13と同一ではありません。
5. **LOF**: 元のmetric/neighbor数は復元できず、L2正規化後のEuclidean距離、
   `contamination=auto`、近傍数min(2,K−1)を採用します。K<3や同値時は全候補を残します。
   K=3では外れ値を検出できないことがあるため、無理に1件を除外しません。
6. **JSON**: `TEXT_SUMMATION_PROMPTS` と最終schemaが欠落しています。
   schemaはrepoの出力項目に基づく再構成で、dict keyと配列をcanonical化します。
7. **Normal logic**: デフォルトは単一参照からのground constraintsです。
   1枚から未観察の正常バリエーション（果物、色、slot等）は推定しません。
   既存の手作業で定義された広い正常規則と判定が異なる可能性があります。
8. **Vocabulary / axioms**: 3カテゴリの形式化例とu/b/f/c設定はrepoを再利用します。
   pushpins/screw_bagの述語は新規再構成です。機能性には存在性も加えています。
   同義語は既存のtask-specific mappingを固定使用し、任意語ペアへの追加LLM synonym queryは行いません。
   `Γ |= ¬Σ0` を保ちますが、自由な関数・任意の述語・LLM生成量化文はサポートしません。
9. **Explanations**: 元repoに実行可能な極小矛盾集合探索は見つかりませんでした。
   bounded deletion探索を追加し、制限到達時には極小性を主張しません。
10. **Existing code**: 古い実験ファイルの作者固有パスは新しい実行経路へ持ち込んでいません。
    元の実験コードは削除/大改変せず残しています。importする旧ファイルは副作用のない
    `formal_prompts_spec.py` と、明示checkpointを渡すGroundingDINOローダーだけです。

出典: [論文](https://arxiv.org/html/2501.01767v2)、
[著者の補足資料](https://jasonjin34.github.io/logicad.github.io/static/pdfs/LogicAD_Supplymentary_Matrials.pdf)、
[OpenAI vision](https://developers.openai.com/api/docs/guides/images-vision)、
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、
[Prover9呼び出し仕様](https://www.cs.unm.edu/~mccune/prover9/manual/2009-11A/running.html)、
[Mace4オプション](https://www.cs.unm.edu/~mccune/prover9/manual/2009-11A/m4-options.html)。
