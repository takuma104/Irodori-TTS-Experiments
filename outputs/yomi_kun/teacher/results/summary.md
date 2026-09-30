# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_kun/teacher/dataset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_kun/teacher/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_kun/teacher/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 2,479 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 2,479 / 2,479 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 86.567% (2,146 / 2,479) |
| Relaxed Accuracy | 86.567% (2,146 / 2,479) |
| Target Kana-CER | 9.763% |
| Target Kana-CER@1 | 8.573% |
| Relaxed Target Kana-CER | 9.763% |
| Relaxed Target Kana-CER@1 | 8.573% |
| Sentence Kana-CER | 1.889% |
| Sentence Kana-CER@1 | 1.889% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 282 | 91.844% | 91.844% | 4.374% | 4.374% | 4.374% | 4.374% |
| Kun’yomi | 2,197 | 85.890% | 85.890% | 10.455% | 9.112% | 10.455% | 9.112% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 2,479 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 333 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 711 | Rows with nonzero Sentence Kana-CER |
