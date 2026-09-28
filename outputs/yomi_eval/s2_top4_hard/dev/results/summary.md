# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s2_top4_hard/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s2_top4_hard/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 79.287% (712 / 898) |
| Relaxed Accuracy | 79.287% (712 / 898) |
| Target Kana-CER | 15.765% |
| Target Kana-CER@1 | 15.135% |
| Relaxed Target Kana-CER | 15.765% |
| Relaxed Target Kana-CER@1 | 15.135% |
| Sentence Kana-CER | 2.777% |
| Sentence Kana-CER@1 | 2.777% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 78.692% | 78.692% | 16.411% | 15.850% | 16.411% | 15.850% |
| Kun’yomi | 319 | 85.580% | 85.580% | 10.982% | 10.355% | 10.982% | 10.355% |
| Jōyō appendix readings | 44 | 40.909% | 40.909% | 42.576% | 41.098% | 42.576% | 41.098% |

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
| [Target review](details/target-review.jsonl) | 186 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 313 | Rows with nonzero Sentence Kana-CER |
