# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/aozora_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/aozora/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s8_cont/aozora/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 3,000 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 3,000 / 3,000 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 62.733% (1,882 / 3,000) |
| Relaxed Accuracy | 62.733% (1,882 / 3,000) |
| Target Kana-CER | 29.581% |
| Target Kana-CER@1 | 27.171% |
| Relaxed Target Kana-CER | 29.581% |
| Relaxed Target Kana-CER@1 | 27.171% |
| Sentence Kana-CER | 7.688% |
| Sentence Kana-CER@1 | 7.688% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 2,070 | 57.295% | 57.295% | 33.399% | 30.896% | 33.399% | 30.896% |
| Kun’yomi | 684 | 78.947% | 78.947% | 16.757% | 14.851% | 16.757% | 14.851% |
| Jōyō appendix readings | 246 | 63.415% | 63.415% | 33.110% | 30.081% | 33.110% | 30.081% |

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
| [Target review](details/target-review.jsonl) | 1,118 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 2,231 | Rows with nonzero Sentence Kana-CER |
