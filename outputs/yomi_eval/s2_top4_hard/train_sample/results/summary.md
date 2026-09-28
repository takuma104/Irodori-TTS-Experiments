# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/train_sample_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s2_top4_hard/train_sample/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s2_top4_hard/train_sample/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 1,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 1,000 / 1,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 72.300% (723 / 1,000) |
| Relaxed Accuracy | 72.300% (723 / 1,000) |
| Target Kana-CER | 23.450% |
| Target Kana-CER@1 | 20.317% |
| Relaxed Target Kana-CER | 23.450% |
| Relaxed Target Kana-CER@1 | 20.317% |
| Sentence Kana-CER | 3.269% |
| Sentence Kana-CER@1 | 3.269% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 441 | 73.923% | 73.923% | 20.053% | 18.353% | 20.053% | 18.353% |
| Kun’yomi | 409 | 80.196% | 80.196% | 18.500% | 15.465% | 18.500% | 15.465% |
| Jōyō appendix readings | 150 | 46.000% | 46.000% | 46.933% | 39.322% | 46.933% | 39.322% |

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
| [Target review](details/target-review.jsonl) | 277 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 393 | Rows with nonzero Sentence Kana-CER |
