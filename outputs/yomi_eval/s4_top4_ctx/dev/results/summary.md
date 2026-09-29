# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s4_top4_ctx/dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 898 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 898 / 898 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 80.958% (727 / 898) |
| Relaxed Accuracy | 80.958% (727 / 898) |
| Target Kana-CER | 14.946% |
| Target Kana-CER@1 | 14.161% |
| Relaxed Target Kana-CER | 14.946% |
| Relaxed Target Kana-CER@1 | 14.161% |
| Sentence Kana-CER | 2.608% |
| Sentence Kana-CER@1 | 2.608% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 535 | 80.000% | 80.000% | 15.882% | 15.134% | 15.882% | 15.134% |
| Kun’yomi | 319 | 86.834% | 86.834% | 9.728% | 9.101% | 9.728% | 9.101% |
| Jōyō appendix readings | 44 | 50.000% | 50.000% | 41.402% | 39.015% | 41.402% | 39.015% |

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
| [Target review](details/target-review.jsonl) | 171 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 299 | Rows with nonzero Sentence Kana-CER |
