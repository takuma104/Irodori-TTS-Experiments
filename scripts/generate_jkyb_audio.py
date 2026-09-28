#!/usr/bin/env python3
"""Synthesize JKYB-Parakeet sentences with Irodori-TTS.

Run inside the Irodori-TTS environment, e.g.:

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync \
        python scripts/generate_jkyb_audio.py \
        --output-dir outputs/jkyb/irodori-v4.1-small/audio \
        --ref-wav data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav

Each dataset row's ``text`` is synthesized and saved as ``<key>.wav``. Texts are
sampled in length-sorted batches (see ``batch_synth.py``). Existing files are
skipped so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from batch_synth import (
    DEFAULT_HF_CHECKPOINT,
    BatchSynthesizer,
    SamplingSettings,
    SynthItem,
    load_runtime,
    set_sdpa_backend,
)
from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import save_wav
from student_text import install_student

DATASET_REPO = "Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet"
DATASET_FILENAME = "data/common_kanji_source.jsonl"


def load_rows(dataset: Path | None, text_field: str = "text") -> list[dict[str, str]]:
    path = (
        dataset
        if dataset is not None
        else Path(
            hf_hub_download(
                repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILENAME
            )
        )
    )
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw = json.loads(line)
                rows.append({"key": raw["key"], "text": raw[text_field]})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", default=DEFAULT_HF_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument(
        "--text-field",
        default="text",
        help="Dataset field to synthesize (e.g. text_hira from jkyb_kana_oracle.py).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--sdpa-backend",
        choices=["efficient", "cudnn"],
        default="efficient",
        help="cudnn (Irodori's default) is slow in bf16 with varying lengths.",
    )
    parser.add_argument(
        "--student",
        type=Path,
        default=None,
        help="Trained student text encoder directory (train_kana_distill.py output).",
    )
    parser.add_argument("--ref-wav", default=None, help="Omit to run with --no-ref.")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-frames", type=int, default=8192)
    parser.add_argument("--pad-multiple", type=int, default=1)
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Also save untrimmed latents as <output-dir>/../latents/<key>.pt.",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.dataset, args.text_field)[args.shard :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = out_dir.parent / "latents"
    if args.save_latents:
        latent_dir.mkdir(parents=True, exist_ok=True)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir.parent / f"generation_config_shard{args.shard}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    set_sdpa_backend(args.sdpa_backend)
    runtime = load_runtime(
        args.hf_checkpoint, device=args.device, precision=args.precision
    )
    if args.student is not None:
        install_student(runtime.model, args.student)
    synth = BatchSynthesizer(
        runtime,
        ref_wav=args.ref_wav,
        settings=SamplingSettings(num_steps=args.num_steps),
    )

    todo = [
        SynthItem(key=row["key"], text=row["text"], seed=args.seed)
        for row in rows
        if not (out_dir / f"{row['key']}.wav").is_file()
    ]
    print(f"rows={len(rows)} todo={len(todo)}", flush=True)
    start = time.perf_counter()
    last_report = 0

    def report(done: int, total: int) -> None:
        nonlocal last_report
        if done - last_report < 200 and done != total:
            return
        last_report = done
        elapsed = time.perf_counter() - start
        eta = elapsed / done * (total - done)
        print(
            f"progress {done}/{total} elapsed={elapsed:.0f}s eta={eta:.0f}s "
            f"rate={done / elapsed:.2f}/s",
            flush=True,
        )

    for output in synth.synthesize(
        todo,
        max_batch_size=args.batch_size,
        max_batch_frames=args.max_batch_frames,
        pad_multiple=args.pad_multiple,
        progress=report,
    ):
        save_wav(out_dir / f"{output.key}.wav", output.audio, output.sample_rate)
        if args.save_latents:
            torch.save(output.latent.float().clone(), latent_dir / f"{output.key}.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
