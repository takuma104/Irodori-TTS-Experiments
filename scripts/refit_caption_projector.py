#!/usr/bin/env python3
"""Refit the caption projector to the student's backbone (state matching only).

    PYTHONPATH=Irodori-TTS:scripts uv run --project Irodori-TTS --no-sync python \
        scripts/refit_caption_projector.py outputs/yomi_prod/s10_cont/part2/student \
        --output-dir outputs/yomi_prod/s10_cont/caption_refit

Irodori-TTS v4.1 runs the caption through the same ModernBERT as the reading
text. Merging a student whose top layers were trained (``top<N>`` scopes) into
a stock checkpoint therefore shifts the caption states too. This trains a copy
of the caption projector and ``caption_norm`` so that, on the merged backbone,
they reproduce the base model's caption states, on voice-design captions and
general sentences. The text path is untouched. No audio is involved.

Writes ``caption.safetensors`` (checkpoint key names, for
``export_student_checkpoint.py --caption``) and ``refit_log.jsonl``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from batch_synth import DEFAULT_HF_CHECKPOINT, load_runtime
from safetensors.torch import load_file, save_file
from torch import nn


class Precomputed(nn.Module):
    """Stands in for the backbone and returns states computed beforehand."""

    def __init__(self, state: torch.Tensor) -> None:
        super().__init__()
        self.state = state

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.state


def read_texts(path: Path, split: str | None = None) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [
        r["text"] for r in rows if split is None or r.get("split", "train") == split
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--captions", type=Path, default=Path("data/yomi/captions.jsonl")
    )
    parser.add_argument(
        "--general-text", type=Path, default=Path("data/yomi/general_text.jsonl")
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    runtime = load_runtime(DEFAULT_HF_CHECKPOINT, precision="fp32")
    model = runtime.model.requires_grad_(False).eval()
    tokenizer = runtime.caption_tokenizer
    max_len = runtime.default_caption_max_len
    device = model.caption_norm.weight.device

    original = model.pretrained_text_backbone
    merged = copy.deepcopy(original)
    student = load_file(str(args.student / "student.safetensors"))
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in student.items()
        if k.startswith("backbone.")
    }
    missing = set(backbone_state) - set(merged.state_dict())
    if missing:
        raise KeyError(f"unknown backbone tensors: {sorted(missing)[:3]}")
    merged.load_state_dict(backbone_state, strict=False)
    merged.to(device)
    encoder = copy.deepcopy(model.caption_encoder).requires_grad_(True).train()
    norm = copy.deepcopy(model.caption_norm).requires_grad_(True).train()
    parameters = list(encoder.parameters()) + list(norm.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)

    captions = read_texts(args.captions, "train")
    general = read_texts(args.general_text)
    dev = read_texts(args.captions, "dev")[:512] + rng.sample(general, 512)
    print(f"captions={len(captions)} general={len(general)} dev={len(dev)}", flush=True)

    def batch_loss(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = tokenizer.batch_encode(texts, max_length=max_len)
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            target = model.caption_norm(model.caption_encoder(original, ids, mask))
            state = merged(ids, mask)
        pred = norm(encoder(Precomputed(state), ids, mask))
        valid = mask.unsqueeze(-1).float()
        diff = ((pred - target) ** 2 * valid).sum() / (valid.sum() * pred.shape[-1])
        rel = ((pred - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-8))[mask]
        return diff, rel.mean()

    def evaluate() -> dict[str, float]:
        encoder.eval()
        norm.eval()
        losses, rels = [], []
        with torch.no_grad():
            for start in range(0, len(dev), args.batch_size):
                loss, rel = batch_loss(dev[start : start + args.batch_size])
                losses.append(float(loss))
                rels.append(float(rel))
        encoder.train()
        norm.train()
        return {"mse": sum(losses) / len(losses), "rel_l2": sum(rels) / len(rels)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = (args.output_dir / "refit_log.jsonl").open("w", encoding="utf-8")
    metrics = evaluate()
    print(f"step 0 dev {metrics}", flush=True)
    log.write(json.dumps({"step": 0, "dev": metrics}) + "\n")
    half = args.batch_size // 2
    for step in range(1, args.steps + 1):
        scale = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * step / args.steps))).item()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1.0, step / 100) * (0.1 + 0.9 * scale)
        texts = rng.sample(captions, half) + rng.sample(general, args.batch_size - half)
        loss, _ = batch_loss(texts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate()
            print(f"step {step} train {float(loss):.6f} dev {metrics}", flush=True)
            log.write(json.dumps({"step": step, "dev": metrics}) + "\n")
            log.flush()

    tensors = {
        f"caption_encoder.{k}": v.detach().float().cpu().contiguous()
        for k, v in encoder.state_dict().items()
    }
    tensors |= {
        f"caption_norm.{k}": v.detach().float().cpu().contiguous()
        for k, v in norm.state_dict().items()
    }
    save_file(tensors, str(args.output_dir / "caption.safetensors"))
    print(f"saved {args.output_dir / 'caption.safetensors'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
