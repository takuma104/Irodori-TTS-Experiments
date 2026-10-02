#!/usr/bin/env python3
"""Throughput of one DiT distillation step on a single GPU.

A frozen MeanFlow teacher DiT runs forward under bf16 autocast, and a
randomly initialized student DiT of a smaller shape runs forward/backward with
AdamW against the teacher output. Conditions are random tensors with typical
lengths (batch 32, 250 latent frames ~ 10 s, 64 text and 64 speaker tokens);
only the DiT part is timed.
"""

from __future__ import annotations

import argparse
import copy
import time

import torch
from common import load_model
from irodori_tts.model import TextToLatentRFDiT

DIT_PREFIXES = ("blocks.", "cond_module.", "delta_cond_module.", "in_proj.", "out_norm.", "out_proj.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-frames", type=int, default=250)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    teacher, _ = load_model("MF")
    teacher.requires_grad_(False)
    cfg = teacher.cfg
    batch, frames, dev = args.batch_size, args.latent_frames, "cuda"
    text_state = torch.randn(batch, 64, cfg.text_dim, device=dev)
    speaker_state = torch.randn(batch, 64, cfg.speaker_dim, device=dev)
    caption_state = torch.randn(batch, 1, cfg.caption_dim_resolved, device=dev)
    conditions = {
        "text_state": text_state,
        "text_mask": torch.ones(batch, 64, dtype=torch.bool, device=dev),
        "speaker_state": speaker_state,
        "speaker_mask": torch.ones(batch, 64, dtype=torch.bool, device=dev),
        "caption_state": caption_state,
        "caption_mask": torch.zeros(batch, 1, dtype=torch.bool, device=dev),
    }
    x_t = torch.randn(batch, frames, cfg.patched_latent_dim, device=dev)
    t = torch.rand(batch, device=dev)
    delta = torch.full((batch,), 0.25, device=dev)

    for num_layers, mlp_ratio in ((12, 2.875), (6, 2.875), (9, 1.5)):
        student_cfg = copy.deepcopy(cfg)
        student_cfg.num_layers, student_cfg.mlp_ratio = num_layers, mlp_ratio
        student = TextToLatentRFDiT(
            student_cfg,
            pretrained_backbone_config=teacher.pretrained_text_backbone.config_dict,
            load_pretrained_backbone_weights=False,
        )
        for unused in ("pretrained_text_backbone", "speaker_encoder", "duration_predictor"):
            setattr(student, unused, None)  # only the DiT is benchmarked
        params = [p for n, p in student.named_parameters() if n.startswith(DIT_PREFIXES)]
        student = student.to(dev).train()
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        def step(model: TextToLatentRFDiT = student, opt: torch.optim.Optimizer = optimizer) -> None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    target = teacher.forward_with_encoded_conditions(x_t=x_t, t=t, delta_t=delta, **conditions)
                out = model.forward_with_encoded_conditions(x_t=x_t, t=t, delta_t=delta, **conditions)
            loss = ((out.float() - target.float()) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        ms = 1000.0 * (time.perf_counter() - t0) / args.steps
        print(
            f"student layers={num_layers} mlp_ratio={mlp_ratio} "
            f"DiT params={sum(p.numel() for p in params) / 1e6:.0f}M step={ms:.0f}ms "
            f"(B={batch}, frames={frames}) peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
            flush=True,
        )
        del student, optimizer
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
