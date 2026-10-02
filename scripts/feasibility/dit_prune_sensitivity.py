#!/usr/bin/env python3
"""Zero-shot pruning sensitivity of the DiT (RF and MeanFlow checkpoints).

The checkpoint synthesizes JSUT sentences; the sampled latents x0 give noised
inputs x_t = (1 - t) x0 + t noise at fixed evaluation points (MF: the four
default inference steps t in {1, .75, .5, .25} with interval 0.25; RF: five
timesteps). For each pruning pattern, reports the mean relative L2 error of the
conditional model output against the unpruned model:

- leave-one-block-out, and greedy cumulative removal of up to six blocks
  (all samples);
- MLP-neuron pruning that keeps the top fraction of hidden units per block,
  ranked by mean |h| x ||w2 column|| on half of the samples and evaluated on
  the other half, next to block removal at a similar parameter saving.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from common import OUTPUT_DIR, jsut_sentences, load_runtime, synthesize_captured
from irodori_tts.model import TextToLatentRFDiT

EVAL_POINTS: dict[str, list[tuple[float, float | None]]] = {
    "MF": [(1.0, 0.25), (0.75, 0.25), (0.5, 0.25), (0.25, 0.25)],
    "RF": [(0.95, None), (0.75, None), (0.5, None), (0.25, None), (0.05, None)],
}
Sample = dict[str, Any]


@torch.inference_mode()
def model_output(
    model: TextToLatentRFDiT, encoded: tuple, x_t: torch.Tensor, t: float, delta: float | None
) -> torch.Tensor:
    t_vec = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
    delta_vec = None if delta is None else torch.full_like(t_vec, delta)
    text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask = encoded
    return model.forward_with_encoded_conditions(
        x_t=x_t,
        t=t_vec,
        text_state=text_state,
        text_mask=text_mask,
        speaker_state=speaker_state,
        speaker_mask=speaker_mask,
        caption_state=caption_state,
        caption_mask=caption_mask,
        delta_t=delta_vec,
    )


@contextmanager
def skipped_blocks(model: TextToLatentRFDiT, skip: set[int]) -> Iterator[None]:
    originals = {index: model.blocks[index].forward for index in skip}
    for index in skip:
        model.blocks[index].forward = lambda x, *_, **__: x
    try:
        yield
    finally:
        for index, forward in originals.items():
            model.blocks[index].forward = forward


def mean_error(
    model: TextToLatentRFDiT,
    samples: list[Sample],
    points: list[tuple[float, float | None]],
    skip: set[int] = frozenset(),
) -> float:
    errors = []
    with skipped_blocks(model, set(skip)):
        for sample in samples:
            for (t, delta), x_t, ref in zip(points, sample["x_t"], sample["ref"], strict=True):
                out = model_output(model, sample["encoded"], x_t, t, delta)
                errors.append(float((out - ref).norm() / ref.norm()))
    return sum(errors) / len(errors)


@torch.inference_mode()
def build_samples(
    model: TextToLatentRFDiT, calls: list[dict[str, Any]], points: list[tuple[float, float | None]]
) -> list[Sample]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    samples = []
    for call in calls:
        encoded = model.encode_conditions(
            text_input_ids=call["text_input_ids"],
            text_mask=call["text_mask"],
            ref_latent=call["ref_latent"],
            ref_mask=call["ref_mask"],
            caption_input_ids=call["caption_input_ids"],
            caption_mask=call["caption_mask"],
        )
        x0 = call["z"]
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        x_ts = [(1.0 - t) * x0 + t * noise for t, _ in points]
        refs = [model_output(model, encoded, x, t, d) for x, (t, d) in zip(x_ts, points, strict=True)]
        samples.append({"encoded": encoded, "x_t": x_ts, "ref": refs})
    return samples


@torch.inference_mode()
def mlp_importance(
    model: TextToLatentRFDiT, samples: list[Sample], points: list[tuple[float, float | None]]
) -> list[torch.Tensor]:
    totals = [torch.zeros(block.mlp.w2.in_features, device="cuda") for block in model.blocks]
    hooks = []
    for index, block in enumerate(model.blocks):

        def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...], index: int = index) -> None:
            totals[index] += inputs[0].abs().flatten(0, -2).float().sum(0)

        hooks.append(block.mlp.w2.register_forward_pre_hook(hook))
    for sample in samples:
        for (t, delta), x_t in zip(points, sample["x_t"], strict=True):
            model_output(model, sample["encoded"], x_t, t, delta)
    for handle in hooks:
        handle.remove()
    return [total * block.mlp.w2.weight.norm(dim=0) for total, block in zip(totals, model.blocks, strict=True)]


def analyze(name: str, num_sentences: int) -> dict[str, Any]:
    runtime = load_runtime(name, model_precision="fp32")
    calls = synthesize_captured(runtime, jsut_sentences(seed=1)[:num_sentences])
    model = runtime.model
    points = EVAL_POINTS[name]
    samples = build_samples(model, calls, points)
    num_blocks = len(model.blocks)

    single = {index: mean_error(model, samples, points, {index}) for index in range(num_blocks)}
    greedy = []
    removed: set[int] = set()
    for k in range(1, 7):
        best = min(
            (index for index in range(num_blocks) if index not in removed),
            key=lambda index: mean_error(model, samples, points, removed | {index}),
        )
        removed.add(best)
        greedy.append({"k": k, "removed": sorted(removed), "rel_err": mean_error(model, samples, points, removed)})

    calibration, held_out = samples[::2], samples[1::2]
    importance = mlp_importance(model, calibration, points)
    original_w2 = [block.mlp.w2.weight.data.clone() for block in model.blocks]
    mlp: dict[str, float] = {}
    for keep in (0.75, 0.5, 0.3):
        for block, score, weight in zip(model.blocks, importance, original_w2, strict=True):
            mask = torch.zeros_like(score)
            mask[score.topk(int(score.numel() * keep)).indices] = 1.0
            block.mlp.w2.weight.data = weight * mask[None, :]
        mlp[f"keep_{keep}"] = mean_error(model, held_out, points)
    for block, weight in zip(model.blocks, original_w2, strict=True):
        block.mlp.w2.weight.data = weight
    mlp["drop_blocks_1_4_6"] = mean_error(model, held_out, points, {1, 4, 6})
    runtime.unload()
    return {"single_block": single, "greedy": greedy, "mlp_pruning_held_out": mlp}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=["MF", "RF"])
    parser.add_argument("--num-sentences", type=int, default=16)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "dit_prune_sensitivity.json"))
    args = parser.parse_args()

    results = {}
    for name in args.checkpoints:
        results[name] = analyze(name, args.num_sentences)
        print(name, json.dumps(results[name]), flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
