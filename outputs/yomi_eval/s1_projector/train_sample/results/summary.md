# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/train_sample_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s1_projector/train_sample/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s1_projector/train_sample/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 1,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,000 / 1,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 71.000% (710 / 1,000) |
| Relaxed Accuracy | 71.000% (710 / 1,000) |
| Target Kana-CER | 25.104% |
| Target Kana-CER@1 | 21.395% |
| Relaxed Target Kana-CER | 25.104% |
| Relaxed Target Kana-CER@1 | 21.395% |
| Sentence Kana-CER | 3.422% |
| Sentence Kana-CER@1 | 3.422% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 441 | 71.429% | 71.429% | 21.316% | 19.502% | 21.316% | 19.502% |
| Kun’yomi | 409 | 79.218% | 79.218% | 20.081% | 16.577% | 20.081% | 16.577% |
| Jōyō appendix readings | 150 | 47.333% | 47.333% | 49.933% | 40.100% | 49.933% | 40.100% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 1,000 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 290 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 409 | Rows with nonzero Sentence Kana-CER |
