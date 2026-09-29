# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_m3/teacher/dataset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_m3/teacher/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_m3/teacher/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 24,219 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 24,219 / 24,219 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 92.964% (22,515 / 24,219) |
| Relaxed Accuracy | 92.964% (22,515 / 24,219) |
| Target Kana-CER | 3.927% |
| Target Kana-CER@1 | 3.745% |
| Relaxed Target Kana-CER | 3.927% |
| Relaxed Target Kana-CER@1 | 3.745% |
| Sentence Kana-CER | 1.386% |
| Sentence Kana-CER@1 | 1.386% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 14,081 | 92.295% | 92.295% | 4.335% | 4.268% | 4.335% | 4.268% |
| Kun’yomi | 8,132 | 94.466% | 94.466% | 3.339% | 3.022% | 3.339% | 3.022% |
| Jōyō appendix readings | 2,006 | 91.575% | 91.575% | 3.450% | 3.010% | 3.450% | 3.010% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 24,219 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 1,704 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 6,043 | Rows with nonzero Sentence Kana-CER |
