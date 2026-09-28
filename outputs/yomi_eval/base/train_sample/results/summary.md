# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/train_sample_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/train_sample/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/train_sample/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 1,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,000 / 1,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 71.200% (712 / 1,000) |
| Relaxed Accuracy | 71.200% (712 / 1,000) |
| Target Kana-CER | 24.778% |
| Target Kana-CER@1 | 21.170% |
| Relaxed Target Kana-CER | 24.778% |
| Relaxed Target Kana-CER@1 | 21.170% |
| Sentence Kana-CER | 3.360% |
| Sentence Kana-CER@1 | 3.360% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 441 | 70.748% | 70.748% | 21.259% | 19.558% | 21.259% | 19.558% |
| Kun’yomi | 409 | 79.707% | 79.707% | 20.207% | 16.213% | 20.207% | 16.213% |
| Jōyō appendix readings | 150 | 49.333% | 49.333% | 47.589% | 39.422% | 47.589% | 39.422% |

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
| [Target review](details/target-review.jsonl) | 288 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 405 | Rows with nonzero Sentence Kana-CER |
