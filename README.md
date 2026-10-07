# Irodori-TTS-Experiments

Experiments on fixing kanji misreadings of
[Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small), especially
on'yomi/kun'yomi confusions (音訓の取り違え), by fine-tuning only its text encoder on
text-only data.

The result is released as
[takuma104/Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi),
a drop-in replacement checkpoint for the stock Irodori-TTS inference code.

A technical report summarizes the method and results:
[English (PDF)](docs/report/irodori_yomi_report_en.pdf) /
[Japanese (PDF)](docs/report/irodori_yomi_report_ja.pdf) (sources in [`docs/report/`](docs/report/)).
Detailed stage-by-stage reports and the experiment plan are written in Japanese under
[`docs/`](docs/). This README summarizes them.

## Results

Mixed-precision inference (DiT in bf16, text encoder and codec in fp32), 40 RF
steps, text CFG 3.0, speaker CFG 5.0, reference voice `jvs001`, seed 0. These settings
differ from the base model card (FP32, no reference, seeds 0–4), so the numbers
are not directly comparable to it. Changes are paired per item against the base model
(fixed / broken counts, exact McNemar test).

| Evaluation | Items | Base | v1 (S10) | v2 (S12) | **v3 (S12+S15 average)** | v3 fixed / broken | v3 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aozora Bunko ruby, held-out works (target reading accuracy) | 3,000 | 62.50% | 63.30% | 65.80% | **67.53%** | 200 / 49 | 8e-23 |
| Wikipedia sentences, held-out words (target reading accuracy) | 6,403 | 84.04% | 84.37% | **85.12%** | 84.98% | 126 / 66 | 2e-5 |
| Pilot dev, held-out hard words (target reading accuracy) | 898 | 80.73% | – | **82.74%** | 81.96% | 31 / 20 | 0.16 |
| JSUT BASIC5000, not used in training (sentence kana CER, lower is better) | 2,236 | 1.881% | 1.799% | 1.778% | **1.713%** | 161 / 99 sentences | – |
| JVS general sentences, not used in training (Whisper CER, lower is better) | 500 | **5.625%** | 5.650% | 5.683% | 5.717% | 13 / 18 sentences | – |
| JKYB-Parakeet, accuracy (external; see the caveat below) | 13,536 | 94.04% | 95.06% | 95.46% | **95.69%** | 320 / 96 | 3e-29 |
| JKYB-Parakeet, on'yomi / kun'yomi / appendix readings | | 92.98 / 95.87 / 83.87% | 94.47 / 96.27 / 86.29% | 94.97 / 96.45 / 88.71% | 95.17 / 96.76 / 88.17% | | |

The released checkpoints are students merged into the base checkpoint (see
[Exporting](#exporting-a-checkpoint)); the numbers are for the exported files, and fixed /
broken counts are against the base model. Precision alone moves these numbers by a few
tenths of a point. The JVS differences are mostly
orthographic variants in the Whisper transcripts (時 / とき, 全て / すべて).

- Words that appear in the training data are fixed in new contexts. Words that never
  appear in training improve only a little.
- Readings of general sentences did not measurably regress. JSUT kana CER improved
  slightly, and the Whisper CER of general sentences is on par with the base model.
- **JKYB caveat.** No JKYB-Parakeet sentence was used for training. The earlier stages
  (S1–S8) did depend on the benchmark in other ways, though:
  - They took their Joyo on/kun reading table from the benchmark's keys.
  - They held out half of the benchmark's target words (group B) from training.
  - Their settings were chosen by JKYB results.

  For v3, group A (words allowed in training) gained +1.91pt. Group B (words held out until S8;
  the production data no longer excludes them) gained +1.41pt. Read the JKYB numbers as
  partly in-domain. The production stages (S9, S10) use no JKYB data and were judged
  only by the other evaluations.

## Method

**Kana-substitution self-distillation.** When the target word is written in katakana,
the base model reads it correctly far more often. With katakana, 85% of its systematic
JKYB errors and 95% of its on/kun swaps disappear
([`docs/reports/yomi_m0_diagnostics.md`](docs/reports/yomi_m0_diagnostics.md)). Training
turns this into supervision for the kanji sentence:

1. The **teacher** is the frozen base model. It sees the sentence with the target word
   replaced by its katakana reading, and generates the audio latents.
2. The **student** is a trainable copy of the text path: the top 4 of the 25 ModernBERT
   layers, the text projector and `text_norm`. It sees the original kanji sentence.
3. The loss matches the DiT's conditional velocity of student and teacher on noised
   teacher latents. It also matches the duration predictor's output. The DiT, the duration
   predictor, the speaker encoder and the codec stay frozen.
4. **Hard mining.** The kana teacher is used only for sentences that the base model
   misreads in kanji and reads correctly in kana. These rows are oversampled 3×. For
   sentences that the base model already reads correctly, the target is the base model's
   own output for the kanji sentence, which keeps them as they are.
5. **Preservation losses.** The student's text states must stay close to the original
   encoder's in two places: on the tokens outside the target word, and on general
   Wikipedia sentences. Without these losses, words that were not trained regressed.
6. **Target-window weighting** (from S11). The velocity loss weights the frames around the
   target 5×. The target's position in the reading is mapped proportionally onto the
   latent frames, with a 10% margin on both sides.

No recorded speech is needed. All audio latents come from the base model itself,
conditioned on JVS reference voices.

Irodori-TTS v4.1 runs the reading text and the voice-design caption through one
shared ModernBERT, with separate projectors. When the student is merged into a stock
checkpoint, the caption path sees the updated layers too. Without any change, the caption
states moved by 21–24% (relative L2 per token). The caption projector is therefore refit to
the new backbone, by matching states on 5.6k LLM-written captions and Wikipedia sentences.
This brings the shift down to 6.5–7.4%, and in caption-only synthesis the median pitch
stays within about half a semitone of the base model ([Exporting](#exporting-a-checkpoint)).

## Data

The target words come from a reading lexicon:
- [JMdict](https://www.edrdg.org/jmdict/j_jmdict.html) for words, readings and frequency tags.
- [JmdictFurigana](https://github.com/Doublevil/JmdictFurigana) for the reading of each kanji.
- [KANJIDIC2](https://www.edrdg.org/wiki/index.php/KANJIDIC_Project) for the Joyo reading
  table of the production lexicon.

Words are grouped into buckets:

| Bucket | Words |
|---|---|
| A | on'yomi in compounds that the tokenizer splits |
| B | on'yomi inside one token |
| C | kun'yomi |
| D | kun'yomi with okurigana |
| E | homographs |
| F | jukujikun and words from the Joyo appendix |
| G | contrast words |
| R | Aozora ruby |

| Stage | Sentences | Hard rows (kana teacher) |
|---|---|---:|
| Pilot | LLM-generated sentences for 2.6k words | 1,895 |
| M3 | LLM-generated sentences for ~21k words | 22,515 |
| Kun | LLM-generated sentences covering each Joyo kun reading, 8 per single-kanji word | 2,146 |
| Prod: Wikipedia | 40,609 sentences for 12,060 words. Kept only when UniDic, Sudachi and JMdict agree on the reading; homographs were also checked by the LLM. | 4,784 |
| Prod: Aozora Bunko | 50,269 sentences, using human ruby readings from 1,000 works | 16,441 |
| General | JVS transcripts (JSUT-derived text) and 200k Wikipedia sentences, for preservation | – |

The LLM was Qwen3.6-27B (NVFP4) running locally on vLLM. It generated sentences and
checked homograph readings.

Evaluation sets:
- **Aozora Bunko ruby.** Ruby readings from a held-out set of works, split by a hash of
  the work ID. Only rubies that agree with a dictionary or analyzer reading are kept,
  which removes authors' idiosyncratic readings.
- **Wikipedia dev words.** Held out by a hash of the word.
- **JSUT BASIC5000.** The human kana labels of `jsut-label`, for sentences not used as
  general training sentences.
- **JVS general sentences.** 500 sentences reserved for regression checks.

## Training history

All runs use batch 32 and the learning rates 3e-4 (projector) / 1e-4 (BERT layers),
with cosine decay. Each run continues from the previous one, except where marked.

| Run | Data | Steps (total) | JKYB | Notes |
|---|---|---:|---:|---|
| S4 | pilot | – | 94.13% | recipe established (top-4 layers, hard mining, preservation losses) |
| S5 | pilot + M3 | 9k (from base) | 94.30% | |
| S6 | same | 12k (21k) | 94.56% | user listening test: fixed readings sound natural |
| S7 | same | 12k (33k) | 94.64% | |
| S8 | + kun | 12k (45k) | 94.88% | |
| S9 | + Wikipedia + Aozora | 15k (60k) | 95.08% | new data barely fit; JKYB-free evals flat |
| S10 | same, new data 2× | 30k (90k) | 95.05% | released as v1; Aozora 63.40%, Wikipedia 84.55% |
| S11 | same, target window weighted 5× | 30k (120k) | 95.32% | Aozora 65.00%, Wikipedia 84.91% |
| **S12** | same | 30k (150k) | 95.42% | released as v2; Aozora 65.87%, Wikipedia 85.09% |
| S13 | same | 60k (210k) | 95.38% | memorizes training sentences; held-out evals flat |
| S14 | S12 + LLM variety (6 sentences per hard word) + 3,000 Aozora works | 60k (210k) | 95.69% | Aozora 69.00%; pilot dev −2.0pt |
| S15 | S14 + contrast sentences for single-kanji targets | 30k (240k) | 95.60% | Aozora 69.10%; pilot dev still −2.6pt |
| **v3** | average of S12 and S15 weights | – | – | released; Aozora 67.77%, pilot dev 82.07% |

After S10, the hard training rows of Wikipedia and Aozora were read correctly only 30% and
18% of the time, against 66% for M3. A diagnosis (`run_diag_fit.sh`) looked into this. It
trained from the base model on hard rows only, with the same number of passes over each
data set:
- Wikipedia rows fit as fast as M3 rows (about 35% each). Aozora rows were harder (19%).
- The S10 gap came mostly from fewer passes over the new data, and from the velocity loss
  being averaged over all frames, which dilutes the target in long sentences.
- Weighting the frames around the target 5× (`--window-weight 5`) worked best: +14pt
  (Wikipedia) and +9pt (Aozora) at equal passes. Training the top 8 layers came second
  (+11 / +8), and weaker preservation losses helped least (+5 / +3).

With the weighting, after S12, the hard rows are read correctly 56% (Wikipedia), 42%
(Aozora) and 81% (M3) of the time. Words that never appear in training still do not
improve (Aozora unseen words: 55.0% → 54.5%).

The steps from S12 to v3:
- **More steps alone memorize the training sentences.** In S13 (60k more steps on the same
  data), the hard Aozora training rows rose from 42% to 57%. Items whose words were trained
  but which appear in held-out sentences stayed at 18%.
- **More varied sentences per word generalize** (S14). Each hard word got 6 LLM-written
  sentences (`select_variation_targets.py`), and 2,000 more Aozora works were added. For
  the words with new sentences, base-misread items in held-out sentences were fixed 34% of
  the time, against 20% before.
- **Contrast sentences protect common compounds** (S15). Single-kanji ruby readings (秋《とき》)
  can leak into compounds. Held-out common compounds stayed at about 97%. The contrast
  sentences come from `select_contrast_targets.py` and keep the base model's own output.
- **Checkpoint averaging limits drift on untrained words** (v3). Long continued training
  slowly changed untrained words (pilot dev 82.9% → 80.3%; 前庭 マエニワ → ゼンニワ).
  Averaging the S12 and S15 weights (`interpolate_students.py`) gave pilot dev 82.1% and
  kept about 60% of the Aozora gain (α = 0.7 gave 81.6% / 68.3%).

Details: [`docs/reports/yomi_production.md`](docs/reports/yomi_production.md).

## Repository layout

```
Irodori-TTS/                                 submodule: Aratako/Irodori-TTS (model and inference code)
Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition/  submodule: JKYB-Parakeet and jkyb-eval
scripts/                                     data building, training, evaluation, export
docs/plans/                                  experiment plan (Japanese)
docs/reports/                                reports per stage (Japanese)
data/, outputs/                              local data and results (not tracked)
```

Main scripts:

| Step | Script |
|---|---|
| Lexicon | `build_yomi_lexicon.py` (`--reading-table jkyb` or `kanjidic2`) |
| Targets | `select_yomi_targets.py`, `select_kun_targets.py`, `retrieve_corpus_sentences.py` + `select_corpus_targets.py`, `make_aozora_ruby_train.py`, `select_variation_targets.py`, `select_contrast_targets.py` |
| Sentences | `generate_yomi_sentences.py` (LLM), `verify_homograph_sentences.py` |
| Rows | `prepare_yomi_rows.py` (JKYB-format rows with Sudachi readings) |
| Teacher | `generate_teacher_latents.py` (kanji/kana), `mix_teacher_by_difficulty.py` (hard mining) |
| Training | `train_kana_distill.py`, `student_text.py`, `interpolate_students.py` |
| Evaluation | `generate_jkyb_audio.py` (batched synthesis), `compare_jkyb_runs.py`, `analyze_jkyb_errors.py`, `eval_general_cer.py`, `make_aozora_ruby_eval.py` |
| Export | `run_release.sh`, `generate_captions.py`, `refit_caption_projector.py`, `export_student_checkpoint.py`, `check_exported_checkpoint.py`, `caption_drift_listen.py`, `run_export_eval.sh` |
| Pipelines | `run_yomi_data.sh <name>` (rows → hard mining → kana teacher → mix), `run_var_data.sh`, `run_prod.sh`, `run_prod_s10.sh`, `run_prod_s11.sh`, `run_prod_cont.sh`, `run_prod_s14.sh`, `run_prod_s15.sh`, `run_diag_fit.sh`, `run_student_eval.sh` |

## Setup

```bash
git clone --recursive git@github.com:takuma104/Irodori-TTS-Experiments.git
cd Irodori-TTS-Experiments
uv sync                                   # data tools (Python 3.13)
(cd Irodori-TTS && uv sync)               # model, training and synthesis
(cd Joyo-Kanji-Yomi-Benchmark-Parakeet-Edition && uv sync)   # jkyb-eval, ASR
```

Scripts that load the model run in the Irodori-TTS environment:

```bash
PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/<script>.py ...
```

External data under `data/`, all downloaded separately:
- `data/jmdict/JMdict_e.gz`, `data/jmdict/JmdictFurigana.json`, `data/jmdict/kanjidic2.xml.gz`
- `data/jvs_ver1/`, the [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- `data/jsut_label/basic5000.yaml`, from [jsut-label](https://github.com/sarulab-speech/jsut-label)

Wikipedia (`wikimedia/wikipedia`, 20231101.ja) and the JKYB-Parakeet dataset are
downloaded from the Hugging Face Hub. The Aozora Bunko index and texts are downloaded
from aozora.gr.jp.

The sentence generator expects an OpenAI-compatible LLM server on `localhost:8000`
(`scripts/serve_llm.sh`). One RTX 5090 (32 GB) was used for everything, one GPU job at a
time.

## Exporting a checkpoint

```bash
uv run python scripts/generate_captions.py --output data/yomi/captions.jsonl   # once; needs the LLM server
PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync python scripts/interpolate_students.py \
  outputs/yomi_prod/s12_cont/student outputs/yomi_prod/s15_cont/student --alpha 0.5 \
  --output outputs/yomi_prod/mix_s12_s15_a50/student
bash scripts/run_release.sh outputs/yomi_prod/mix_s12_s15_a50/student release_mix50 ../Irodori-TTS-v4.1-Small-Yomi
```

`run_release.sh` runs the following steps:
1. Refit the caption projector (`refit_caption_projector.py`).
2. Merge the student and the refit caption projector into the base checkpoint
   (`export_student_checkpoint.py --caption`).
3. Check the exported file (`check_exported_checkpoint.py`).
4. Write caption-only samples (`caption_drift_listen.py`).
5. Evaluate the exported file itself (`run_export_eval.sh`).

The intermediate results go to `outputs/release/<name>/`.

`export_student_checkpoint.py` writes the student's weights into the base checkpoint,
with the same keys, dtype and metadata, and copies the tokenizer.
`check_exported_checkpoint.py` checks two things:
- The text path of the exported checkpoint is identical to the evaluated student.
- How far the caption states and emoji tokens drift from the base model.

`caption_drift_listen.py` writes caption-only samples of three systems (the base model,
the base model with the student installed, and the exported checkpoint) together with
their median pitch.

In fp32, the exported checkpoint gives exactly the same text states as the student. The
mixed-precision runtime first casts the whole model to bf16 and then returns the text
modules to fp32. The merged top layers are therefore bf16-rounded there, unlike the
`--student` path. `run_export_eval.sh` re-evaluates the exported file itself in that
setting.

## License

The code in this repository is released under the [MIT License](LICENSE). The data
sources keep their own licenses, for example:
- JMdict, JmdictFurigana and KANJIDIC2: CC BY-SA 4.0
- Wikipedia: CC BY-SA
- JSUT labels: CC BY-SA 4.0
- The JVS corpus has its own terms of use.

The submodules are licensed by their authors.
