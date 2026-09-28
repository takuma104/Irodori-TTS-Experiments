#!/usr/bin/env python3
"""Check that batched synthesis matches ``InferenceRuntime.synthesize``.

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync \
        python scripts/check_batch_synth.py --precision fp32

For a few JKYB sentences of different lengths, compares the predicted length and
the final (trimmed, watermarked) audio of one-at-a-time runtime synthesis with
``BatchSynthesizer`` run as a single padded batch, and reports the speed of both.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import torch
from batch_synth import BatchSynthesizer, SynthItem, load_runtime, set_sdpa_backend
from generate_jkyb_audio import load_rows
from irodori_tts.inference_runtime import SamplingRequest

REF_WAV = "data/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"
FRAMES_PATTERN = re.compile(r"using_frames=(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--sdpa-backend", choices=["efficient", "cudnn"], default="efficient"
    )
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--max-rel-diff", type=float, default=0.05)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for BatchSynthesizer (default: all sentences in one batch).",
    )
    args = parser.parse_args()

    set_sdpa_backend(args.sdpa_backend)
    rows = load_rows(None)[:: 13536 // args.num][: args.num]
    runtime = load_runtime(precision=args.precision)
    synth = BatchSynthesizer(runtime, ref_wav=REF_WAV)

    # Warm up both paths so the timings exclude one-off initialization.
    runtime.synthesize(SamplingRequest(text=rows[0]["text"], ref_wav=REF_WAV, seed=0))
    list(synth.synthesize([SynthItem(key="warmup", text=rows[0]["text"], seed=0)]))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    single: dict[str, tuple[int, torch.Tensor]] = {}
    for row in rows:
        result = runtime.synthesize(
            SamplingRequest(text=row["text"], ref_wav=REF_WAV, seed=0)
        )
        frames = next(
            int(m.group(1))
            for msg in result.messages
            if (m := FRAMES_PATTERN.search(msg))
        )
        single[row["key"]] = (frames, result.audio)
    torch.cuda.synchronize()
    single_sec = time.perf_counter() - t0

    items = [SynthItem(key=row["key"], text=row["text"], seed=0) for row in rows]
    t0 = time.perf_counter()
    batch_size = len(items) if args.batch_size is None else args.batch_size
    batched = {
        out.key: out
        for out in synth.synthesize(
            items, max_batch_size=batch_size, duration_batch_size=batch_size
        )
    }
    torch.cuda.synchronize()
    batch_sec = time.perf_counter() - t0

    failures = 0
    for row in rows:
        key = row["key"]
        frames, audio_ref = single[key]
        audio = batched[key].audio
        steps = int(batched[key].latent.shape[0])
        length_ok = frames == steps and audio.shape == audio_ref.shape
        if length_ok:
            rel = float((audio - audio_ref).norm() / audio_ref.norm().clamp_min(1e-8))
        else:
            rel = float("nan")
        ok = length_ok and rel <= args.max_rel_diff
        failures += not ok
        print(
            f"{'OK ' if ok else 'NG '} {key}: frames {frames} vs {steps}, "
            f"samples {audio_ref.shape[-1]} vs {audio.shape[-1]}, rel_l2={rel:.4f}"
        )
    print(
        f"single: {len(rows) / single_sec:.2f} sent/s, batched: {len(rows) / batch_sec:.2f} sent/s"
    )
    print(f"failures={failures}/{len(rows)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
