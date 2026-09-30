# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_kun/base_kanji/dataset.jsonl` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_kun/base_kanji/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/yomi_kun/base_kanji/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Rows | 30,952 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 30,952 / 30,952 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 91.374% (28,282 / 30,952) |
| Relaxed Accuracy | 91.374% (28,282 / 30,952) |
| Target Kana-CER | 8.320% |
| Target Kana-CER@1 | 6.951% |
| Relaxed Target Kana-CER | 8.320% |
| Relaxed Target Kana-CER@1 | 6.951% |
| Sentence Kana-CER | 1.660% |
| Sentence Kana-CER@1 | 1.660% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 2,932 | 85.164% | 85.164% | 16.996% | 13.449% | 16.996% | 13.449% |
| Kun’yomi | 28,020 | 92.024% | 92.024% | 7.412% | 6.271% | 7.412% | 6.271% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 0 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 30,952 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 2,670 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 7,364 | Rows with nonzero Sentence Kana-CER |
