# JKYB-Parakeet Results

## Input

| Item | Value |
|---|---|
| Dataset | `Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet` |
| Audio | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001_bf16/audio` |
| Predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001_bf16/results/transcriptions/kana.jsonl` |
| Reading field | `yomi` |
| Text predictions | `/home/takuma/co/Irodori-TTS-Experiments/outputs/irodori-v4.1-small_jvs001_bf16/results/transcriptions/text.jsonl` |
| Text field | `text` |
| Rows | 13,536 |

## Coverage

| Input | Available | Missing |
|---|---:|---:|
| Audio | 13,536 / 13,536 (100.000%) | 0 |

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | 93.484% (12,654 / 13,536) |
| Relaxed Accuracy | 93.580% (12,667 / 13,536) |
| Target Kana-CER | 5.922% |
| Target Kana-CER@1 | 5.119% |
| Relaxed Target Kana-CER | 5.813% |
| Relaxed Target Kana-CER@1 | 5.023% |
| Sentence Kana-CER | 1.005% |
| Sentence Kana-CER@1 | 1.005% |
| Text CER | 4.028% |
| Text CER@1 | 4.023% |

Raw CER can exceed 100%; CER@1 clips each example's CER to 100% before averaging.

## Metrics by Reading Category

| Category | Rows | Accuracy | Relaxed Accuracy | Target Kana-CER | Target Kana-CER@1 | Relaxed Target Kana-CER | Relaxed Target Kana-CER@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| On’yomi | 7,056 | 92.404% | 92.446% | 7.020% | 6.052% | 6.985% | 6.016% |
| Kun’yomi | 6,108 | 95.318% | 95.432% | 4.292% | 3.768% | 4.161% | 3.653% |
| Jōyō appendix readings | 372 | 83.871% | 84.677% | 11.846% | 9.606% | 10.726% | 8.665% |

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
| [Target review](details/target-review.jsonl) | 882 | Incorrect or marginal target readings and alignment warnings |
| [Sentence mismatches](details/sentence-mismatches.jsonl) | 1,852 | Rows with nonzero Sentence Kana-CER |
| [Text mismatches](details/text-mismatches.jsonl) | 4,563 | Rows with nonzero Text CER |
