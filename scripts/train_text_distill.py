#!/usr/bin/env python3
"""Text-only feature distillation of the text encoder (plan Phase 1a).

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/train_text_distill.py \
        --student-init outputs/students/text_mbert130m/init \
        --output-dir outputs/students/text_mbert130m/p1a

The student (``make_text_student.py``) learns to reproduce the frozen teacher's
condition states for the same sentence:

- the text state ``text_norm(text_encoder(backbone(x)))`` and the caption state
  ``caption_norm(caption_encoder(backbone(x)))`` (both 512-d, one shared
  backbone pass), with a relative squared error over valid tokens plus a
  per-token cosine term;
- the frozen duration predictor's log frame count on the student's text state
  (no speaker, no caption), so that length prediction is preserved.

Only the text path is trained; the rest of the checkpoint stays a copy of the
teacher, so every saved ``model.safetensors`` is directly usable for synthesis
(teacher DiT + student text encoder).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from distill_lib import (
    TEXT_PATH_PREFIXES,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from irodori_tts.duration import build_duration_features
from irodori_tts.model import PretrainedConditionProjector, TextToLatentRFDiT
from irodori_tts.text_normalization import normalize_text
from irodori_tts.tokenizer import PretrainedTextTokenizer
from voice_captions import voice_captions

MAX_TEXT_LEN = 256


def project(projector: PretrainedConditionProjector, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``PretrainedConditionProjector.forward`` on a precomputed backbone state."""
    projected = projector.projector(state)
    if projector.projector_type == "residual_mlp":
        residual = F.silu(projector.residual_up(projector.residual_norm(state)))
        projected = projected + projector.residual_down(projector.residual_dropout(residual))
    return projected * mask.unsqueeze(-1).to(dtype=projected.dtype)


def condition_states(
    model: TextToLatentRFDiT, ids: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = model.pretrained_text_backbone(ids, mask)
    text_state = model.text_norm(project(model.text_encoder, state, mask))
    caption_state = model.caption_norm(project(model.caption_encoder, state, mask))
    return text_state, caption_state


def duration_log_frames(
    model: TextToLatentRFDiT, text_state: torch.Tensor, mask: torch.Tensor, texts: list[str]
) -> torch.Tensor:
    batch = text_state.shape[0]
    no_condition = torch.zeros(batch, dtype=torch.bool, device=text_state.device)
    features = build_duration_features(
        texts, token_counts=mask.sum(dim=1), max_text_len=MAX_TEXT_LEN, has_speaker=no_condition
    ).to(text_state.device)
    return model.predict_duration_log_frames(
        text_state=text_state.float(),
        text_mask=mask,
        speaker_state=None,
        speaker_mask=None,
        duration_features=features,
        has_speaker=no_condition,
        caption_state=None,
        caption_mask=None,
        has_caption=no_condition,
        detach_condition=False,
    )


def relative_sq_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).float()
    return (((pred.float() - target) ** 2) * weight).sum() / ((target**2) * weight).sum()


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(pred.float(), target, dim=-1)
    weight = mask.float()
    return ((1.0 - cos) * weight).sum() / weight.sum()


def read_lines(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        lines.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [normalize_text(line).strip() for line in lines if line]


def batches(texts: list[str], batch_size: int, seed: int) -> Iterator[list[str]]:
    rng = random.Random(seed)
    order = list(range(len(texts)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            yield [texts[i] for i in order[start : start + batch_size]]


def encode(
    tokenizer: PretrainedTextTokenizer, texts: list[str], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(ids) for ids in tokenizer.tokenizer(texts, add_special_tokens=False)["input_ids"]]
    max_length = min(MAX_TEXT_LEN, max(lengths) + 1)  # + BOS
    ids, mask = tokenizer.batch_encode(texts, max_length=max_length)
    return ids.to(device), mask.to(device)


def lr_scale(step: int, warmup: int, total: int, min_scale: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_train_mode(student: TextToLatentRFDiT) -> None:
    """Train mode for the text path only; the frozen duration predictor keeps dropout off."""
    student.train()
    student.duration_predictor.eval()


@torch.no_grad()
def evaluate(
    teacher: TextToLatentRFDiT,
    student: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    texts: list[str],
    batch_size: int,
    captions: list[str],
) -> dict[str, float]:
    """Relative errors on corpus sentences (both projectors) and on voice captions (caption path)."""
    student.eval()
    sums = {"text_num": 0.0, "text_den": 0.0, "caption_num": 0.0, "caption_den": 0.0, "dur_abs": 0.0,
            "vc_num": 0.0, "vc_den": 0.0}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        ids, mask = encode(tokenizer, chunk, "cuda")
        t_text, t_caption = condition_states(teacher, ids, mask)
        s_text, s_caption = condition_states(student, ids, mask)
        weight = mask.unsqueeze(-1).float()
        sums["text_num"] += float((((s_text - t_text) ** 2) * weight).sum())
        sums["text_den"] += float(((t_text**2) * weight).sum())
        sums["caption_num"] += float((((s_caption - t_caption) ** 2) * weight).sum())
        sums["caption_den"] += float(((t_caption**2) * weight).sum())
        t_dur = duration_log_frames(teacher, t_text, mask, chunk)
        s_dur = duration_log_frames(teacher, s_text, mask, chunk)
        sums["dur_abs"] += float((s_dur - t_dur).abs().sum())
    for start in range(0, len(captions), batch_size):
        ids, mask = encode(tokenizer, captions[start : start + batch_size], "cuda")
        _, t_caption = condition_states(teacher, ids, mask)
        _, s_caption = condition_states(student, ids, mask)
        weight = mask.unsqueeze(-1).float()
        sums["vc_num"] += float((((s_caption - t_caption) ** 2) * weight).sum())
        sums["vc_den"] += float(((t_caption**2) * weight).sum())
    set_train_mode(student)
    result = {
        "text_rel_err": math.sqrt(sums["text_num"] / sums["text_den"]),
        "caption_rel_err": math.sqrt(sums["caption_num"] / sums["caption_den"]),
        "duration_abs_log_err": sums["dur_abs"] / len(texts),
    }
    if captions:
        result["voice_caption_rel_err"] = math.sqrt(sums["vc_num"] / sums["vc_den"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="MF")
    parser.add_argument("--student-init", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, nargs="+", default=[Path("data/corpus/wiki_train.txt")])
    parser.add_argument("--val-corpus", type=Path, default=Path("data/corpus/wiki_val.txt"))
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Projectors and norms.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--min-lr-scale", type=float, default=0.05)
    parser.add_argument("--caption-weight", type=float, default=1.0)
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=0,
        help="Voice captions per step for the caption path (voice_captions.py); 0 disables.",
    )
    parser.add_argument("--num-captions", type=int, default=20000)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--duration-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    teacher_ckpt = load_checkpoint(args.teacher)
    student_ckpt = load_checkpoint(args.student_init)
    teacher = build_model(teacher_ckpt.flat_config, teacher_ckpt.text_encoder_config, teacher_ckpt.state)
    teacher = teacher.cuda().eval().requires_grad_(False)
    student = build_model(student_ckpt.flat_config, student_ckpt.text_encoder_config, student_ckpt.state)
    student = student.cuda()
    set_train_mode(student)
    for name, param in student.named_parameters():
        param.requires_grad_(name.startswith(TEXT_PATH_PREFIXES))
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(teacher_ckpt.tokenizer_dir), add_bos=True, local_files_only=True
    )

    backbone_params = [p for n, p in student.named_parameters() if n.startswith("pretrained_text_backbone.")]
    head_params = [
        p for n, p in student.named_parameters() if p.requires_grad and not n.startswith("pretrained_text_backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    train_texts = read_lines(args.train_corpus)
    val_texts = read_lines([args.val_corpus])[: args.num_val]
    all_captions = voice_captions(args.num_captions, seed=12345)
    val_captions = all_captions[:500]  # the same held-out captions for every run
    train_captions = all_captions[500:]
    caption_batches = batches(train_captions, args.caption_batch_size, args.seed) if args.caption_batch_size else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    (args.output_dir / "train_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"train={len(train_texts):,} val={len(val_texts):,}", flush=True)

    def log(record: dict[str, float | int]) -> None:
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    log({"step": 0, **evaluate(teacher, student, tokenizer, val_texts, 256, val_captions)})
    start = time.perf_counter()
    running: dict[str, float] = {}
    for step, texts in enumerate(batches(train_texts, args.batch_size, args.seed)):
        if step >= args.steps:
            break
        scale = lr_scale(step, args.warmup, args.steps, args.min_lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        ids, mask = encode(tokenizer, texts, "cuda")
        with torch.no_grad():
            t_text, t_caption = condition_states(teacher, ids, mask)
            t_dur = duration_log_frames(teacher, t_text, mask, texts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_text, s_caption = condition_states(student, ids, mask)
        losses = {
            "text": relative_sq_error(s_text, t_text, mask),
            "caption": relative_sq_error(s_caption, t_caption, mask),
            "cosine": cosine_loss(s_text, t_text, mask) + cosine_loss(s_caption, t_caption, mask),
            "duration": F.smooth_l1_loss(duration_log_frames(student, s_text, mask, texts), t_dur, beta=0.1),
        }
        loss = (
            losses["text"]
            + args.caption_weight * losses["caption"]
            + args.cosine_weight * losses["cosine"]
            + args.duration_weight * losses["duration"]
        )
        if caption_batches is not None:
            caption_texts = next(caption_batches)
            c_ids, c_mask = encode(tokenizer, caption_texts, "cuda")
            with torch.no_grad():
                _, t_voice = condition_states(teacher, c_ids, c_mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, s_voice = condition_states(student, c_ids, c_mask)
            losses["voice_caption"] = relative_sq_error(s_voice, t_voice, c_mask)
            loss = loss + args.caption_weight * (
                losses["voice_caption"] + args.cosine_weight * cosine_loss(s_voice, t_voice, c_mask)
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach())
        if (step + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start
            log(
                {
                    "step": step + 1,
                    **{f"loss_{k}": v / args.log_every for k, v in running.items()},
                    "grad_norm": float(grad_norm),
                    "lr_scale": scale,
                    "steps_per_s": (step + 1) / elapsed,
                }
            )
            running = {}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            log({"step": step + 1, **evaluate(teacher, student, tokenizer, val_texts, 256, val_captions)})

    save_checkpoint(
        student,
        flat_config=student_ckpt.flat_config,
        text_encoder_config=student_ckpt.text_encoder_config,
        output_dir=args.output_dir,
        tokenizer_dir=student_ckpt.tokenizer_dir,
    )
    print(f"saved {args.output_dir / 'model.safetensors'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
