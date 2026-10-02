#!/usr/bin/env python3
"""MeanFlow -> MeanFlow output distillation of a smaller DiT (plan Phase 2).

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/train_dit_distill.py \
        --student-init outputs/students/dit_d6/init --output-dir outputs/students/dit_d6/p2

Each step samples condition triples (text, reference voice or none, caption or
none), encodes them with the frozen teacher's encoders, predicts their lengths,
and runs the teacher's 4-step MeanFlow sampler. The student DiT regresses the
teacher's output u(x_t, t, delta) at

- the four inference states of the teacher trajectory (t = 1, .75, .5, .25;
  delta = .25), or, with ``--onpolicy-prob``, at the states of the student's own
  trajectory (teacher targets recomputed there);
- ``--random-points`` extra states x_t = (1 - t) x_0 + t x_1 on the teacher's
  path with random t and delta <= t, so other step counts keep working.

The loss is the per-utterance mean squared error over valid frames. Only the
DiT (blocks, timestep/interval embeddings, in/out projections) is trained; the
condition encoders and duration predictor are the teacher's, so every saved
``model.safetensors`` is a complete drop-in checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from batch_synth import set_sdpa_backend
from distill_data import (
    ConditionSampler,
    EncodedBatch,
    encode_batch,
    initial_noise,
    load_refs,
    load_texts,
    masked_relative_error,
    masked_utterance_mse,
    meanflow_rollout,
)
from distill_lib import build_model, load_checkpoint, save_checkpoint
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from voice_captions import voice_captions

DIT_PREFIXES = ("blocks.", "cond_module.", "delta_cond_module.", "in_proj.", "out_norm.", "out_proj.")


def model_output(
    model: TextToLatentRFDiT, encoded: EncodedBatch, x_t: torch.Tensor, t: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    return model.forward_with_encoded_conditions(
        x_t=x_t, t=t, delta_t=delta, latent_mask=encoded.latent_mask, **encoded.conditions()
    )


def lr_scale(step: int, warmup: int, total: int, min_scale: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def training_batch(
    teacher: TextToLatentRFDiT,
    student: TextToLatentRFDiT,
    encoded: EncodedBatch,
    x_1: torch.Tensor,
    *,
    on_policy: bool,
    random_points: int,
    rng: random.Random,
) -> tuple[EncodedBatch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (conditions, x_t, t, delta, teacher target) over inference and random points."""
    batch = x_1.shape[0]
    device = x_1.device
    with torch.no_grad():
        teacher_path = meanflow_rollout(teacher, encoded, x_1)
        if on_policy:
            student.eval()
            states = meanflow_rollout(student, encoded, x_1).states
            student.train()
            targets = [
                model_output(teacher, encoded, x, torch.full((batch,), t, device=device), torch.full((batch,), d, device=device))
                for x, t, d in zip(states, teacher_path.times, teacher_path.deltas, strict=True)
            ]
        else:
            states, targets = teacher_path.states, teacher_path.outputs
        times = [torch.full((batch,), t, device=device) for t in teacher_path.times]
        deltas = [torch.full((batch,), d, device=device) for d in teacher_path.deltas]
        repeats = len(states)
        xs, ts, ds, us = list(states), times, deltas, list(targets)
        if random_points > 0:
            index = torch.tensor(rng.sample(range(batch), min(random_points, batch)), device=device)
            sub = EncodedBatch(
                text_state=encoded.text_state[index],
                text_mask=encoded.text_mask[index],
                speaker_state=None if encoded.speaker_state is None else encoded.speaker_state[index],
                speaker_mask=None if encoded.speaker_mask is None else encoded.speaker_mask[index],
                caption_state=None if encoded.caption_state is None else encoded.caption_state[index],
                caption_mask=None if encoded.caption_mask is None else encoded.caption_mask[index],
                frames=[encoded.frames[i] for i in index.tolist()],
                latent_mask=encoded.latent_mask[index],
            )
            t = torch.rand(len(index), device=device).clamp_min(0.02)
            d = torch.rand(len(index), device=device) * t
            x0, x1 = teacher_path.final[index], x_1[index]
            x_t = ((1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1).to(x_1.dtype)
            u = model_output(teacher, sub, x_t, t, d)
    stacked = encoded.repeat(repeats)
    if random_points > 0:
        stacked = concat_encoded(stacked, sub)
        xs.append(x_t), ts.append(t), ds.append(d), us.append(u)
    return stacked, torch.cat(xs), torch.cat(ts), torch.cat(ds), torch.cat(us)


def concat_encoded(a: EncodedBatch, b: EncodedBatch) -> EncodedBatch:
    def cat(x: torch.Tensor | None, y: torch.Tensor | None) -> torch.Tensor | None:
        if x is None or y is None:
            return None
        if x.ndim >= 2 and x.shape[1] != y.shape[1]:  # pad the sequence axis to a common length
            length = max(x.shape[1], y.shape[1])
            x = torch.nn.functional.pad(x, (0, 0) * (x.ndim - 2) + (0, length - x.shape[1]))
            y = torch.nn.functional.pad(y, (0, 0) * (y.ndim - 2) + (0, length - y.shape[1]))
        return torch.cat([x, y])

    return EncodedBatch(
        text_state=cat(a.text_state, b.text_state),
        text_mask=cat(a.text_mask, b.text_mask),
        speaker_state=cat(a.speaker_state, b.speaker_state),
        speaker_mask=cat(a.speaker_mask, b.speaker_mask),
        caption_state=cat(a.caption_state, b.caption_state),
        caption_mask=cat(a.caption_mask, b.caption_mask),
        frames=a.frames + b.frames,
        latent_mask=cat(a.latent_mask, b.latent_mask),
    )


@torch.no_grad()
def evaluate(
    teacher: TextToLatentRFDiT,
    student: TextToLatentRFDiT,
    val_batches: list[tuple[EncodedBatch, torch.Tensor]],
) -> dict[str, float]:
    student.eval()
    final_errors, step_errors = [], [[] for _ in range(4)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for encoded, x_1 in val_batches:
            teacher_path = meanflow_rollout(teacher, encoded, x_1)
            student_path = meanflow_rollout(student, encoded, x_1)
            final_errors.append(masked_relative_error(student_path.final, teacher_path.final, encoded.latent_mask))
            for k, (x, t, d, u) in enumerate(
                zip(teacher_path.states, teacher_path.times, teacher_path.deltas, teacher_path.outputs, strict=True)
            ):
                batch = x.shape[0]
                pred = model_output(
                    student, encoded, x, torch.full((batch,), t, device=x.device), torch.full((batch,), d, device=x.device)
                )
                step_errors[k].append(masked_relative_error(pred, u, encoded.latent_mask))
    student.train()
    result = {"val_final_rel_err": sum(final_errors) / len(final_errors)}
    for k, errors in enumerate(step_errors):
        result[f"val_u_rel_err_step{k}"] = sum(errors) / len(errors)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="MF")
    parser.add_argument("--student-init", type=Path, required=True)
    parser.add_argument("--texts", type=Path, nargs="+", default=[Path("data/corpus/wiki_train.txt")])
    parser.add_argument("--val-texts", type=Path, default=Path("data/corpus/wiki_val.txt"))
    parser.add_argument("--refs", type=Path, nargs="+", default=[Path("data/refs/real_pool.pt")])
    parser.add_argument("--num-captions", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8, help="Condition triples per step.")
    parser.add_argument("--random-points", type=int, default=8)
    parser.add_argument("--onpolicy-prob", type=float, default=0.0)
    parser.add_argument("--onpolicy-start", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--num-val", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sdpa-backend",
        choices=["efficient", "cudnn"],
        default="efficient",
        help="cudnn rebuilds its plan for every new sequence length in bf16 (slow here).",
    )
    args = parser.parse_args()

    set_sdpa_backend(args.sdpa_backend)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher_ckpt = load_checkpoint(args.teacher)
    student_ckpt = load_checkpoint(args.student_init)
    teacher = build_model(teacher_ckpt.flat_config, teacher_ckpt.text_encoder_config, teacher_ckpt.state)
    teacher = teacher.cuda().eval().requires_grad_(False)
    student = build_model(student_ckpt.flat_config, student_ckpt.text_encoder_config, student_ckpt.state).cuda()
    student.train()
    params = []
    for name, param in student.named_parameters():
        param.requires_grad_(name.startswith(DIT_PREFIXES))
        if param.requires_grad:
            params.append(param)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(teacher_ckpt.tokenizer_dir), add_bos=True, local_files_only=True
    )
    captions = voice_captions(args.num_captions, seed=args.seed)
    refs = load_refs(args.refs)
    sampler = ConditionSampler(load_texts(args.texts), refs, captions, seed=args.seed)
    val_sampler = ConditionSampler(load_texts([args.val_texts]), refs, captions, seed=12345)
    device = torch.device("cuda")
    val_generator = torch.Generator(device="cuda").manual_seed(12345)
    val_batches = []
    for _ in range(math.ceil(args.num_val / 16)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = encode_batch(teacher, tokenizer, val_sampler.sample(16), device=device)
        val_batches.append((encoded, initial_noise(encoded, teacher.cfg.patched_latent_dim, val_generator, torch.float32)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    (args.output_dir / "train_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def log(record: dict[str, float | int]) -> None:
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def save() -> None:
        save_checkpoint(
            student,
            flat_config=student_ckpt.flat_config,
            text_encoder_config=student_ckpt.text_encoder_config,
            output_dir=args.output_dir,
            tokenizer_dir=student_ckpt.tokenizer_dir,
        )

    log({"step": 0, **evaluate(teacher, student, val_batches)})
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    start = time.perf_counter()
    running_loss, running_onpolicy = 0.0, 0
    for step in range(args.steps):
        scale = lr_scale(step, args.warmup, args.steps, args.min_lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        on_policy = step >= args.onpolicy_start and rng.random() < args.onpolicy_prob
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                encoded = encode_batch(teacher, tokenizer, sampler.sample(args.batch_size), device=device)
            x_1 = initial_noise(encoded, teacher.cfg.patched_latent_dim, generator, torch.float32)
            stacked, x_t, t, delta, target = training_batch(
                teacher, student, encoded, x_1, on_policy=on_policy, random_points=args.random_points, rng=rng
            )
            pred = model_output(student, stacked, x_t, t, delta)
        loss = masked_utterance_mse(pred, target, stacked.latent_mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach())
        running_onpolicy += int(on_policy)
        if (step + 1) % args.log_every == 0:
            log(
                {
                    "step": step + 1,
                    "loss": running_loss / args.log_every,
                    "onpolicy_frac": running_onpolicy / args.log_every,
                    "grad_norm": float(grad_norm),
                    "lr_scale": scale,
                    "steps_per_s": (step + 1) / (time.perf_counter() - start),
                    "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            running_loss, running_onpolicy = 0.0, 0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            log({"step": step + 1, **evaluate(teacher, student, val_batches)})
        if (step + 1) % args.save_every == 0:
            save()
    save()
    print(f"saved {args.output_dir / 'model.safetensors'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
