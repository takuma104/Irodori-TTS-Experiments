# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/overfit32/rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/overfit32/base/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/overfit32/base/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 32 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 32 / 32 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 3.125% (1 / 32) |
| Relaxed Accuracy | 3.125% (1 / 32) |
| Target Kana-CER | 88.333% |
| Target Kana-CER@1 | 71.927% |
| Relaxed Target Kana-CER | 88.333% |
| Relaxed Target Kana-CER@1 | 71.927% |
| Sentence Kana-CER | 8.103% |
| Sentence Kana-CER@1 | 8.103% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 16 | 6.250% | 6.250% | 75.729% | 63.229% | 75.729% | 63.229% |
| Kun’yomi | 7 | 0.000% | 0.000% | 85.714% | 71.429% | 85.714% | 71.429% |
| Jōyō appendix readings | 9 | 0.000% | 0.000% | 112.778% | 87.778% | 112.778% | 87.778% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 32 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 31 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 31 | Rows with nonzero Sentence Kana-CER |
