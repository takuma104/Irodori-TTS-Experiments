# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s5_top4_ctx/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 81.737% (734 / 898) |
| Relaxed Accuracy | 81.737% (734 / 898) |
| Target Kana-CER | 14.089% |
| Target Kana-CER@1 | 13.415% |
| Relaxed Target Kana-CER | 14.089% |
| Relaxed Target Kana-CER@1 | 13.415% |
| Sentence Kana-CER | 2.537% |
| Sentence Kana-CER@1 | 2.537% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 81.308% | 81.308% | 14.318% | 13.757% | 14.318% | 13.757% |
| Kun’yomi | 319 | 86.520% | 86.520% | 10.199% | 9.572% | 10.199% | 9.572% |
| Jōyō appendix readings | 44 | 52.273% | 52.273% | 39.508% | 37.121% | 39.508% | 37.121% |

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
| [Target review](details/target-review.jsonl) | 164 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 298 | Rows with nonzero Sentence Kana-CER |
