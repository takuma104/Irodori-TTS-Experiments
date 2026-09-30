# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_kun_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/jkyb_seen_kun/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/jkyb_seen_kun/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 2,437 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 2,437 / 2,437 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 97.087% (2,366 / 2,437) |
| Relaxed Accuracy | 97.128% (2,367 / 2,437) |
| Target Kana-CER | 2.499% |
| Target Kana-CER@1 | 2.253% |
| Relaxed Target Kana-CER | 2.417% |
| Relaxed Target Kana-CER@1 | 2.212% |
| Sentence Kana-CER | 0.542% |
| Sentence Kana-CER@1 | 0.542% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 5 | 100.000% | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| Kun’yomi | 2,432 | 97.081% | 97.122% | 2.504% | 2.257% | 2.422% | 2.216% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 2,437 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 71 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 202 | Rows with nonzero Sentence Kana-CER |
