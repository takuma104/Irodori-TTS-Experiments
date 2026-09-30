# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/aozora_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/aozora/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/aozora/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 3,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 3,000 / 3,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 62.500% (1,875 / 3,000) |
| Relaxed Accuracy | 62.500% (1,875 / 3,000) |
| Target Kana-CER | 30.242% |
| Target Kana-CER@1 | 27.769% |
| Relaxed Target Kana-CER | 30.242% |
| Relaxed Target Kana-CER@1 | 27.769% |
| Sentence Kana-CER | 7.718% |
| Sentence Kana-CER@1 | 7.718% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 2,070 | 57.150% | 57.150% | 33.664% | 31.233% | 33.664% | 31.233% |
| Kun’yomi | 684 | 78.509% | 78.509% | 17.780% | 15.880% | 17.780% | 15.880% |
| Jōyō appendix readings | 246 | 63.008% | 63.008% | 36.098% | 31.680% | 36.098% | 31.680% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 4 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 3,000 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 1,125 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 2,234 | Rows with nonzero Sentence Kana-CER |
