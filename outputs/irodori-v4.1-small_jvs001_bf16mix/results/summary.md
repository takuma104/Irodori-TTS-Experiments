# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001_bf16mix/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001_bf16mix/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001_bf16mix/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 94.038% (12,729 / 13,536) |
| Relaxed Accuracy | 94.127% (12,741 / 13,536) |
| Target Kana-CER | 5.512% |
| Target Kana-CER@1 | 4.717% |
| Relaxed Target Kana-CER | 5.415% |
| Relaxed Target Kana-CER@1 | 4.632% |
| Sentence Kana-CER | 0.900% |
| Sentence Kana-CER@1 | 0.900% |
| Text CER | 3.934% |
| Text CER@1 | 3.930% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 92.985% | 93.013% | 6.588% | 5.636% | 6.567% | 5.615% |
| Kun’yomi | 6,108 | 95.874% | 95.989% | 3.912% | 3.355% | 3.781% | 3.241% |
| Jōyō appendix readings | 372 | 83.871% | 84.677% | 11.384% | 9.637% | 10.399% | 8.831% |

## Diagnostics

| Item | Rows |
|---|---:|
| Missing predictions | 0 |
| Extra predictions | 0 |
| Ambiguous target alignments | 1 |

## Details

| File | Rows | Contents |
|---|---:|---|
| [All rows](details/all.jsonl) | 13,536 | Complete per-row results |
| [Missing inputs](details/missing-inputs.jsonl) | 0 | Rows scored as errors because no input was available |
| [Target review](details/target-review.jsonl) | 807 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,647 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,517 | Rows with nonzero Text CER |
