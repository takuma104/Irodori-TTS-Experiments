# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/kun_dev_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/kun_dev/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/kun_dev/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 3,140 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 3,140 / 3,140 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 91.720% (2,880 / 3,140) |
| Relaxed Accuracy | 91.720% (2,880 / 3,140) |
| Target Kana-CER | 7.895% |
| Target Kana-CER@1 | 6.653% |
| Relaxed Target Kana-CER | 7.895% |
| Relaxed Target Kana-CER@1 | 6.653% |
| Sentence Kana-CER | 1.414% |
| Sentence Kana-CER@1 | 1.414% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 264 | 84.470% | 84.470% | 20.833% | 14.205% | 20.833% | 14.205% |
| Kun’yomi | 2,876 | 92.385% | 92.385% | 6.708% | 5.960% | 6.708% | 5.960% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 3,140 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 260 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 705 | Rows with nonzero Sentence Kana-CER |
