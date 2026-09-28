# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/jkyb_seen/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/base/jkyb_seen/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 316 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 316 / 316 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 85.127% (269 / 316) |
| Relaxed Accuracy | 85.443% (270 / 316) |
| Target Kana-CER | 12.975% |
| Target Kana-CER@1 | 10.601% |
| Relaxed Target Kana-CER | 12.658% |
| Relaxed Target Kana-CER@1 | 10.285% |
| Sentence Kana-CER | 1.833% |
| Sentence Kana-CER@1 | 1.833% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 163 | 82.822% | 82.822% | 14.724% | 13.190% | 14.724% | 13.190% |
| Kun’yomi | 33 | 96.970% | 96.970% | 1.515% | 1.515% | 1.515% | 1.515% |
| Jōyō appendix readings | 120 | 85.000% | 85.833% | 13.750% | 9.583% | 12.917% | 8.750% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 316 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 47 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 65 | Rows with nonzero Sentence Kana-CER |
