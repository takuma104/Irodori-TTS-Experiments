# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/teacher/dataset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/teacher/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_pilot/teacher/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 9,267 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 9,267 / 9,267 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 96.277% (8,922 / 9,267) |
| Relaxed Accuracy | 96.277% (8,922 / 9,267) |
| Target Kana-CER | 2.527% |
| Target Kana-CER@1 | 2.335% |
| Relaxed Target Kana-CER | 2.527% |
| Relaxed Target Kana-CER@1 | 2.335% |
| Sentence Kana-CER | 1.244% |
| Sentence Kana-CER@1 | 1.244% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 5,151 | 96.175% | 96.175% | 2.764% | 2.506% | 2.764% | 2.506% |
| Kun’yomi | 3,533 | 96.688% | 96.688% | 2.325% | 2.197% | 2.325% | 2.197% |
| Jōyō appendix readings | 583 | 94.683% | 94.683% | 1.661% | 1.661% | 1.661% | 1.661% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 9,267 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 345 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,871 | Rows with nonzero Sentence Kana-CER |
