# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/jkyb/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/jkyb/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s6_cont/jkyb/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.555% (12,799 / 13,536) |
| Relaxed Accuracy | 94.644% (12,811 / 13,536) |
| Target Kana-CER | 4.810% |
| Target Kana-CER@1 | 4.202% |
| Relaxed Target Kana-CER | 4.713% |
| Relaxed Target Kana-CER@1 | 4.117% |
| Sentence Kana-CER | 0.839% |
| Sentence Kana-CER@1 | 0.839% |
| Text CER | 3.958% |
| Text CER@1 | 3.954% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 93.707% | 93.736% | 5.631% | 4.937% | 5.610% | 4.915% |
| Kun’yomi | 6,108 | 95.989% | 96.103% | 3.607% | 3.143% | 3.476% | 3.029% |
| Jōyō appendix readings | 372 | 87.097% | 87.903% | 8.992% | 7.648% | 8.006% | 6.841% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 13,536 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 737 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,613 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,543 | Rows with nonzero Text CER |
