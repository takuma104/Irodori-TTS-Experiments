# Irodori-TTS v4.1-Small(-MF) のサイズ削減 Distill：実現性検討と実験計画

## Context

- **目的**：推論に要るVRAMを主に減らしたい（速度向上は副次的）。text encoder（ModernBERT）と DiT を小さくした生徒を蒸留で作る。
- **対象**：Small-MF を主にする（4ステップで速い）。Small(RF) は副とする。
- **目標**：JKYB-Parakeet の低下を数pt以内に抑えつつ、パラメータ数を約半分にする。推論コードは変えず、`model.safetensors`（とその隣の `tokenizer/`）を差し替えるだけで動くこと。成果は HF で公開する前提。
- mainブランチの読み改善の作業（自己蒸留、評価パイプライン）の資産を流用する。`third-party/Irodori-TTS` には手を入れない。

## 検討結果（実測、2026-10-03、RTX 5090）

### 1. Drop-in の差し替えは可能（コード変更は不要）

- 推論側は safetensors のメタデータ `config_json`（ModelConfig）と `text_encoder_config_json`（HF の ModernBERT config）からモデルを組み立て、`load_state_dict` を strict で読む（`irodori_tts/inference_runtime.py:494-523, 637-656`）。
  - `num_layers` / `mlp_ratio` / `adaln_rank` / `speaker_layers` などを書き換えて、形の合う重みを保存すれば、そのまま読める。
- ModernBERT 側は config に `layer_types`（full / sliding の並び）を持つ。層を抜いても各層の attention の種類を保てる。
- ModernBERT-ja の 30m/70m/130m/310m は、`tokenizer.json` がすべて同一（ハッシュ一致）。Irodori に同梱のトークナイザとは、JSUT の 5,002 文でトークン列が完全に一致した。
  - よって backbone を **modernbert-ja-130m に差し替えても、同梱のトークナイザのまま使える**。
- Small と MF とで違うのは DiT（と MF の `delta_cond_module`）だけ。text encoder、speaker encoder、duration predictor、projector は完全に同一。
  - よって **text encoder の生徒は1つ作れば両方に使える**。

### 2. パラメータの内訳（Small 766M / MF 773M、fp32 で保存）

| 部位 | パラメータ | 割合 |
|---|---:|---:|
| ModernBERT 層（25層、768次元） | 236.0M | 31% |
| ModernBERT 埋め込み（102400×768） | 78.6M | 10% |
| DiT 12 blocks（1280次元。MLP 169.6M、self-attn 98.3M、ctx-KV 55.1M、AdaLN 35.5M） | 358.4M | 47% |
| cond_module (+MFの delta_cond_module) | 7.2M (+7.2M) | 1-2% |
| speaker encoder（8層、768次元） | 60.5M | 8% |
| duration predictor | 21.8M | 3% |
| text/caption projector | 3.4M | 0.4% |

### 3. 推論時の VRAM と速度（MF、model bf16、7.7秒の発話）

| 設定 | 重み | 常駐 | ピーク (7.7秒) | ピーク (30秒) |
|---|---:|---:|---:|---:|
| codec fp32 | 1475MB | 2041MB | 3117MB | 5469MB |
| codec bf16 | 1475MB | 1783MB | 2605MB | 4534MB |

- 処理段ごとのピーク増分（アクティベーション）：
  - DiT のサンプリング：60〜260MB
  - ModernBERT 等の条件エンコード：20MB 程度
  - **DACVAE の復号：1.07GB（7.7秒）〜3.2GB（30秒）**
- モデルを半分にすると約 0.75GB 減る。7.7秒の発話では、ピークが 3.1→2.4GB（codec bf16 なら 2.6→1.85GB）になる。
- コーデック側の対策は今回の範囲外とする（既存の `--codec-precision bf16` の利用を注記するだけ）。
- 速度（MF、bf16）：計 109ms。内訳は DiT 43ms、復号 25ms、duration+条件エンコード 14ms、参照のエンコード 14ms、透かし 12ms。DiT を半分にすると約 -20%。RF（40ステップ）なら約 -40%。

### 4. 学習せずに枝刈りしたときの感度（どちらも冗長性は小さく、回復のための蒸留学習が必須）

ModernBERT（1層だけ抜いたときの text_state の相対 L2 誤差）：
- 中間層 3〜11：約 8%。最も冗長。
- 層 12〜16：約 10%。
- 層 0（36%）、18（26%）、24（28%）は致命的。

DiT（1ブロックだけ抜いたときの速度場の相対誤差）：
- RF：14〜84%。MF：30〜91%。
- block 0、10、11 が致命的。1〜7 は比較的冗長。
- 6ブロックを貪欲に除去すると {1..6} が選ばれ、誤差は RF 54%、MF 72%。
- MLP の中間ニューロンを 50% 刈った場合（約2.8ブロック分の削減）は RF 36%、MF 49% で、3ブロックを抜いた場合（34% / 54%）と同程度。

### 5. 学習コスト（問題にならない）

- 6層の生徒 DiT（194M）を、教師の順伝播込みでバッチ 32×約10秒で回すと、1ステップ 113ms、ピーク 10.5GB。
- 5万ステップで約 1.6h。MF 教師の軌道生成（教師 4 fwd/サンプル）を足しても 3〜4h。
- text encoder の蒸留はテキストだけで済み、さらに安い。

### 再現手順

上記の数値は `scripts/feasibility/` で再現できる。結果の JSON は `outputs/feasibility/` に置く。

```bash
# JSUT の参照音声2本と文を data/jsut/ に展開する（git 管理外）
unzip -j /mnt/datasets/speech/free_distribute/jsut/origin/jsut_ver1.1.zip \
  "jsut_ver1.1/basic5000/wav/BASIC5000_000[12].wav" "jsut_ver1.1/basic5000/transcript_utf8.txt" -d data/jsut
export PYTHONPATH=third-party/Irodori-TTS:scripts/feasibility
PY=third-party/Irodori-TTS/.venv/bin/python
$PY scripts/feasibility/profile_vram.py          # §3 VRAM と速度
$PY scripts/feasibility/bert_layer_drop.py       # §4 ModernBERT の層削除
$PY scripts/feasibility/dit_prune_sensitivity.py # §4 DiT のブロック削除と MLP の刈り込み
$PY scripts/feasibility/bench_distill_step.py    # §5 学習ステップの速度
```

### 結論

- 構造上は実現可能。差し替えも計算量も問題ない。
- 未知数は品質だけ。特に DiT は密で、半分にすると音質や話者類似度の劣化が出やすい。
- JKYB（読み）は主に text 側で決まる（mainの M0 診断）。text 側の生徒の質が JKYB を左右する。
- そこで、text 側と DiT 側を**それぞれ教師のインターフェース tensor に合わせる形で別々に蒸留し、教師モデルに差し込んで個別に評価する**。そのうえで統合する。

## 生徒の構成（目標）

| 部位 | 教師 | 第一候補 | 比較候補 |
|---|---|---|---|
| text backbone | 310m・25層（315M） | **T-B**：modernbert-ja-130m 事前学習済み（132M）+ projector を新規に（512→512） | **T-A**：310m を12層に削る（192M）。中間層 3〜16 から13層を除く。`layer_types` は保つ |
| DiT | 12層×1280（373M） | **D6**：6層×1280（194M）。init は block {0,7,8,9,10,11} | **D9M**：9層×1280、mlp_ratio 1.5（223M）。{1,4,6} を除き、MLP を重要度で刈る |
| speaker encoder | 8層（60.5M） | 据え置き | 4層（30.3M）。Phase 3b |
| duration predictor | 21.8M | 据え置き | – |

合計（MF）：
- T-B + D6 = 約 411M（53%）。speaker encoder 4層を足すと約 381M（**49%**）。
- T-A + D6 = 約 471M（61%）。
- bf16 の重みは 1475MB → 約 725〜785MB。

## 実験計画

### Phase 0：基盤（評価・データ）

1. mainのスクリプトをこのブランチの `scripts/` に移植する。パスは `third-party/` 配下に直す。
   - 対象：`batch_synth.py`、`generate_jkyb_audio.py`、`compare_jkyb_runs.py`（McNemar）、`eval_general_cer.py`、`analyze_jkyb_errors.py`
   - JKYB の環境：`third-party/Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition` で `uv sync --extra asr`。
2. 評価の設定は、教師と生徒で同一にする。
   - MF 既定の4ステップ、seed 0、参照話者1名を固定する。
   - 精度は2通り：mainと同じ混合精度（DiT bf16、text/codec fp32）と、全部 bf16（VRAM を節約する配備形態）。
   - 参照音声：JVS jvs001 が手元にない。入手できればmainの数値と比べられる。入手できなければ JSUT を固定参照にする。
3. 教師のベースラインを測る。MF と Small について：
   - JKYB の全 13,536 文
   - dev 2,000 文（読み分類で層化。モデル選択用）
4. 品質の補助指標を用意する。JKYB だけでは音質の劣化が見えないため。
   - JSUT BASIC5000 の kana CER（mainの `eval_general_cer.py --mode kana`）
   - Whisper CER
   - 話者類似度（WavLM 系の SV 埋め込み、参照との cos）
   - UTMOS
   - 推定尺の比
   - VRAM と速度（`scratchpad/profile_*.py` を `scripts/profile_vram.py` に整理）
5. 蒸留データを用意する（公開前提）。
   - テキスト：日本語 Wikipedia（HF `wikimedia/wikipedia`）、青空文庫（`/mnt/datasets/.../aozorabunko/parquets`）、コーパスの書き起こし。JKYB の文と重複するものは除く。
   - caption：ローカル LLM で生成する（数万件。声質、感情、話し方の記述）。
   - 参照音声：自由に使えるコーパスに限る（JVNV、Hi-Fi-CAPTAIN ja、あみたろ、JSUT 等。**ライセンスはコーパスごとに確認**）。games 系は使わない。
   - 参照 latent は DACVAE で事前にエンコードしておく。

### Phase 1：text encoder の蒸留（テキストのみ。Small と MF で共有）

- **1a 特徴蒸留**
  - 教師の `text_norm(text_encoder(...))` と `caption_norm(caption_encoder(...))` を、トークンごとに MSE + cos で合わせる。
  - frozen の duration predictor の出力（log frames）も合わせる。
  - T-A は中間層の出力合わせを足してもよい。
  - T-A と T-B の両方を、各 1〜2h で回す。
- **1b DiT-in-the-loop**
  - frozen の教師 MF の DiT 上で、教師の条件と生徒の条件それぞれで u(x_t,t,Δ) を出し、両者を合わせる。duration の損失も入れる。
  - mainの `train_kana_distill.py` と同じ仕組みで、ターゲットを「教師条件の出力」に変える。
- 学習は fp32 で行う。推論の精度感度（mainの知見：text を bf16 にすると -0.5pt）は、評価の2通りの精度で確認する。
- **関門 G1**：教師 DiT に生徒 text を差し込んだとき、次を満たすこと。T-A と T-B のうち良い方を採る。
  - JKYB dev の低下 ≤ 1.0pt
  - JSUT kana CER の悪化 ≤ +0.2pt
  - caption のみの合成で話者類似度・ピッチのずれが小さい（mainの caption drift 手順）

### Phase 2：DiT の蒸留（MF を主とする）

- **主法：MF→MF の出力蒸留**
  1. 条件（テキスト、参照、caption）ごとに、MF 教師で4ステップの軌道を作る。
  2. 状態 x_t（t ∈ {1, .75, .5, .25}、Δ=.25）と教師の出力 u_T を得る。
  3. 生徒にその u_T を回帰させる（発話平均 MSE）。
  - 混ぜるもの：
    - 推論の点以外への一般化用：25% はランダムな (t,Δ) と補間した状態（`--num-steps` の変更に耐えるため）
    - 学習が進んだら、生徒自身の軌道上の状態に教師ターゲットを付けたもの（on-policy）
    - 条件の有無のプロファイル：ref なし、caption なし。`irodori_tts.meanflow.apply_condition_profile` / `drop_condition` を使う
  - 補助損失：
    - 中間 hidden の合わせ（弱く）
    - 終盤：4ステップを展開した最終 latent の一致
- **pilot**：D6 と D9M をそれぞれ 1万ステップ（約 40 分）回し、dev で D6 と D9M のどちらかを選ぶ。
- **本番**：5〜10万ステップ（3〜6h）。Muon+AdamW（`irodori_tts/optim.py`）、WSD スケジュール。
- **代替法（主法で品質が足りない場合）**：RF 教師から小さな MF 生徒へ、公式と同じターゲットで蒸留する。
  - `irodori_tts.meanflow.compute_teacher_meanflow_target` と `adaptive_meanflow_loss` を、生徒の config を変えた自前ループから呼ぶ。
- **関門 G2**：教師 text + 生徒 DiT で、次を満たすこと。
  - JKYB dev の低下 ≤ 1.5pt
  - CER、話者類似度、UTMOS の悪化が許容範囲内（教師の seed 間のばらつきを基準に決める）

### Phase 3：統合と書き出し

- 生徒 text と生徒 DiT を組み合わせ、元の MF 全体を教師として、両方を小さい LR で共同微調整する（3a）。
- 3b（任意）：speaker encoder を8→4層に削る。参照 latent だけで `speaker_state` を合わせる特徴蒸留をしてから、in-loop で仕上げる。
- 書き出し：mainの `export_student_checkpoint.py` と `convert_checkpoint_to_safetensors.py` の `_build_safetensors_metadata` に倣う。
  - `config_json`：`num_layers`、`mlp_ratio`、`text_tokenizer_repo` / `text_encoder_revision` を更新する。
  - `text_encoder_config_json`：130m の config、または削った 310m の config（`layer_types` 付き）を入れる。
  - 同梱の `tokenizer/` をコピーする。
- **最終評価**：
  - JKYB の全文（教師と対にした McNemar）、補助指標、VRAM と速度
  - `quantize_checkpoint.py`（int8 weight-only）との併用時の値
- Phase 4（任意）：RF Small 版の DiT 生徒。text 生徒はそのまま流用する。

## 作るファイル（このブランチ。`third-party/` は変えない）

- `scripts/`：Phase 0 で移植するもの、`build_distill_corpus.py`、`make_student.py`（層の選択、MLP の刈り込み、130m での init）、`train_text_distill.py`、`train_dit_distill.py`、`export_small_checkpoint.py`、`profile_vram.py`
- 流用するもの：
  - `irodori_tts.model.TextToLatentRFDiT`（`encode_conditions`、`forward_with_encoded_conditions`、`build_context_kv_cache`）
  - `inference_runtime._load_checkpoint_for_inference`
  - `meanflow.sample_euler_meanflow` と関連ヘルパー、`optim.py`
- `docs/reports/` に各 Phase の結果を書く（mainのレポート形式に合わせる）。この計画は承認後に `docs/plans/model_size_distill_plan.md` へ移す。
- 環境：mainと同じく、`third-party/Irodori-TTS` の venv（`uv sync --extra cu128` 済み）で `PYTHONPATH` を通して実行する。トップの `pyproject.toml` には必要な依存だけを `uv add` する。

## 検証（end-to-end）

1. 書き出した `model.safetensors` と `tokenizer/` が、無改変の `infer.py --checkpoint <path> --text ... --ref-wav ...` で、MF/RF ともに読めて合成できること。Gradio でも読めること。`quantize_checkpoint.py` も通ること。
2. 各関門で、教師との対の比較（JKYB dev と全文、McNemar）と補助指標を、レポートに記録する。
3. VRAM と速度を、教師と同じスクリプト、同じ発話（7.7秒 / 30秒）で測る。
4. 合否の目安：
   - JKYB の全文で、教師（同じ設定）からの低下 ≤ 2pt
   - JSUT kana CER、Whisper CER、話者類似度、UTMOS の悪化が許容範囲内
   - パラメータ ≤ 約 55%

## リスクと対策

- **DiT が密で、半分にすると品質が落ちる**：D9M（幅を混ぜる）、on-policy 学習、展開した出力損失、代替法（RF 教師）を試す。それでも足りなければ、DiT は 9層程度にとどめる。text 側と speaker encoder の削減で約 55% を狙う。
- **130m では読みの知識が足りない**：T-A（310m を削って埋め込みを保つ）に切り替える。
- **caption 経路のずれ**：caption の状態の蒸留を同時に行い、mainの drift チェックで確認する。
- **ベンチマークへの過適合**：JKYB の文は学習に使わない。モデル選択は dev のみで行い、最終の全文評価は1回だけにする。
