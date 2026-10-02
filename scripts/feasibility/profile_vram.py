#!/usr/bin/env python3
"""Inference VRAM and latency of Irodori-TTS checkpoints, in total and per stage.

For each checkpoint (RF / MF) x model precision x codec precision, reports the
model weight size, the resident allocation after loading (model + codec), the
peak allocation while synthesizing a short (auto-duration) and a 30-second
utterance, the runtime's stage timings, and the activation peak of each stage
above the allocation at its start.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import torch
from common import MIB, OUTPUT_DIR, load_runtime, reference_wav
from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest

SHORT_TEXT = "木曜日、停戦会談は、何の進展もないまま終了しました。今後の見通しは立っていません。"
LONG_TEXT = "長文のテスト。" * 30


class StagePeaks:
    """Wraps callables to record their activation peak (MiB above the allocation at entry)."""

    def __init__(self) -> None:
        self.peaks: dict[str, float] = {}

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - before) / MIB
            self.peaks[name] = max(self.peaks.get(name, 0.0), peak)
            return out

        return inner


def instrument(runtime: InferenceRuntime, stages: StagePeaks) -> None:
    runtime.codec.decode_latent = stages.wrap("decode_latent", runtime.codec.decode_latent)
    runtime.codec.encode_waveform = stages.wrap("encode_reference", runtime.codec.encode_waveform)
    runtime.model.encode_conditions = stages.wrap(
        "encode_conditions", runtime.model.encode_conditions
    )


def measure(name: str, model_precision: str, codec_precision: str, repeats: int) -> dict[str, Any]:
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    runtime = load_runtime(name, model_precision, codec_precision)
    resident = (torch.cuda.memory_allocated() - base) / MIB
    weights = sum(p.numel() * p.element_size() for p in runtime.model.parameters()) / MIB
    stages = StagePeaks()
    instrument(runtime, stages)
    utterances: dict[str, Any] = {}
    for label, text, seconds in (("short", SHORT_TEXT, None), ("30s", LONG_TEXT, 30.0)):
        request = SamplingRequest(text=text, ref_wav=reference_wav(1), seed=0, seconds=seconds)
        for _ in range(repeats):  # the last repeat is reported (earlier ones warm up kernels)
            stages.peaks.clear()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            result = runtime.synthesize(request)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        utterances[label] = {
            "audio_seconds": result.audio.shape[-1] / result.sample_rate,
            "peak_mib": (torch.cuda.max_memory_allocated() - base) / MIB,
            "total_ms": elapsed * 1000.0,
            "stage_ms": {stage: sec * 1000.0 for stage, sec in result.stage_timings},
            "stage_activation_peak_mib": dict(stages.peaks),
        }
    runtime.unload()
    del runtime
    torch.cuda.empty_cache()
    return {
        "checkpoint": name,
        "model_precision": model_precision,
        "codec_precision": codec_precision,
        "weights_mib": weights,
        "resident_mib": resident,
        "utterances": utterances,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=["RF", "MF"])
    parser.add_argument("--model-precisions", nargs="+", default=["fp32", "bf16"])
    parser.add_argument("--codec-precisions", nargs="+", default=["fp32", "bf16"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "profile_vram.json"))
    args = parser.parse_args()

    results = []
    for name in args.checkpoints:
        for model_precision in args.model_precisions:
            for codec_precision in args.codec_precisions:
                info = measure(name, model_precision, codec_precision, args.repeats)
                results.append(info)
                short, long = info["utterances"]["short"], info["utterances"]["30s"]
                print(
                    f"{name} model={model_precision} codec={codec_precision} "
                    f"weights={info['weights_mib']:.0f}MiB resident={info['resident_mib']:.0f}MiB "
                    f"peak(short {short['audio_seconds']:.1f}s)={short['peak_mib']:.0f}MiB "
                    f"peak(30s)={long['peak_mib']:.0f}MiB total(short)={short['total_ms']:.0f}ms",
                    flush=True,
                )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
