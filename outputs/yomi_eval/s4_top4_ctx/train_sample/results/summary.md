# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/train_sample_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/train_sample/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/train_sample/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 1,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,000 / 1,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 80.100% (801 / 1,000) |
| Relaxed Accuracy | 80.100% (801 / 1,000) |
| Target Kana-CER | 15.024% |
| Target Kana-CER@1 | 13.683% |
| Relaxed Target Kana-CER | 15.024% |
| Relaxed Target Kana-CER@1 | 13.683% |
| Sentence Kana-CER | 2.441% |
| Sentence Kana-CER@1 | 2.441% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 441 | 78.912% | 78.912% | 15.327% | 14.080% | 15.327% | 14.080% |
| Kun’yomi | 409 | 87.042% | 87.042% | 9.466% | 9.099% | 9.466% | 9.099% |
| Jōyō appendix readings | 150 | 64.667% | 64.667% | 29.289% | 25.011% | 29.289% | 25.011% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 1,000 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 199 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 333 | Rows with nonzero Sentence Kana-CER |
