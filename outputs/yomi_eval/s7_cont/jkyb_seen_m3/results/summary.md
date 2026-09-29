# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/jkyb_seen_m3_rows.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/jkyb_seen_m3/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_eval/s7_cont/jkyb_seen_m3/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 864 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 864 / 864 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 90.972% (786 / 864) |
| Relaxed Accuracy | 91.088% (787 / 864) |
| Target Kana-CER | 7.211% |
| Target Kana-CER@1 | 6.400% |
| Relaxed Target Kana-CER | 7.095% |
| Relaxed Target Kana-CER@1 | 6.285% |
| Sentence Kana-CER | 1.235% |
| Sentence Kana-CER@1 | 1.235% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 488 | 90.369% | 90.369% | 8.060% | 7.343% | 8.060% | 7.343% |
| Kun’yomi | 246 | 93.089% | 93.089% | 5.461% | 4.648% | 5.461% | 4.648% |
| Jōyō appendix readings | 130 | 89.231% | 90.000% | 7.333% | 6.179% | 6.564% | 5.410% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 864 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 78 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 156 | Rows with nonzero Sentence Kana-CER |
