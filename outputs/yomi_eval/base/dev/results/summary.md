# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 80.735% (725 / 898) |
| Relaxed Accuracy | 80.735% (725 / 898) |
| Target Kana-CER | 15.471% |
| Target Kana-CER@1 | 14.519% |
| Relaxed Target Kana-CER | 15.471% |
| Relaxed Target Kana-CER@1 | 14.519% |
| Sentence Kana-CER | 2.583% |
| Sentence Kana-CER@1 | 2.583% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 79.813% | 79.813% | 16.561% | 15.533% | 16.561% | 15.533% |
| Kun’yomi | 319 | 86.834% | 86.834% | 9.963% | 9.336% | 9.963% | 9.336% |
| Jōyō appendix readings | 44 | 47.727% | 47.727% | 42.159% | 39.773% | 42.159% | 39.773% |

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
| [Target review](details/target-review.jsonl) | 173 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 295 | Rows with nonzero Sentence Kana-CER |
