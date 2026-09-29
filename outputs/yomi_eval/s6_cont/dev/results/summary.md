# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 82.851% (744 / 898) |
| Relaxed Accuracy | 82.851% (744 / 898) |
| Target Kana-CER | 13.513% |
| Target Kana-CER@1 | 12.784% |
| Relaxed Target Kana-CER | 13.513% |
| Relaxed Target Kana-CER@1 | 12.784% |
| Sentence Kana-CER | 2.490% |
| Sentence Kana-CER@1 | 2.490% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 82.430% | 82.430% | 13.212% | 12.651% | 13.212% | 12.651% |
| Kun’yomi | 319 | 87.147% | 87.147% | 10.512% | 9.728% | 10.512% | 9.728% |
| Jōyō appendix readings | 44 | 56.818% | 56.818% | 38.939% | 36.553% | 38.939% | 36.553% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 898 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 154 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 291 | Rows with nonzero Sentence Kana-CER |
