# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 93.913% (12,712 / 13,536) |
| Relaxed Accuracy | 94.001% (12,724 / 13,536) |
| Target Kana-CER | 5.596% |
| Target Kana-CER@1 | 4.819% |
| Relaxed Target Kana-CER | 5.499% |
| Relaxed Target Kana-CER@1 | 4.734% |
| Sentence Kana-CER | 0.908% |
| Sentence Kana-CER@1 | 0.908% |
| Text CER | 3.945% |
| Text CER@1 | 3.943% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 92.914% | 92.942% | 6.644% | 5.707% | 6.623% | 5.685% |
| Kun’yomi | 6,108 | 95.711% | 95.825% | 4.022% | 3.490% | 3.891% | 3.376% |
| Jōyō appendix readings | 372 | 83.333% | 84.140% | 11.541% | 9.794% | 10.556% | 8.987% |

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
| [Target review](details/target-review.jsonl) | 824 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,678 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,550 | Rows with nonzero Text CER |
