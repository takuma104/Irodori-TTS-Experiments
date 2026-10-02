#!/usr/bin/env python3
"""MeanFlow output distillation of a smaller student (plan Phases 1b, 2 and 3a).

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/train_dit_distill.py \
        --student-init outputs/students/dit_d6/init --output-dir outputs/students/dit_d6/p2

Each step samples condition triples (text, reference voice or none, caption or
none), encodes them with the frozen teacher's encoders, predicts their lengths,
and runs the teacher's 4-step MeanFlow sampler. The student regresses the
teacher's output u(x_t, t, delta) at

- the four inference states of the teacher trajectory (t = 1, .75, .5, .25;
  delta = .25), or, with ``--onpolicy-prob``, at the states of the student's own
  trajectory (teacher targets recomputed there);
- ``--random-points`` extra states x_t = (1 - t) x_0 + t x_1 on the teacher's
  path with random t and delta <= t, so other step counts keep working.

The loss is the per-utterance mean squared error over valid frames.

What is trained:

- default (Phase 2): the student DiT (blocks, timestep/interval embeddings,
  in/out projections) on the teacher's encoded conditions;
- ``--train-text`` (Phase 1b with ``--no-train-dit``, Phase 3a with both): the
  student's text path as well. The student then encodes the items itself (with
  the teacher's latent lengths), and the loss adds the relative error of its
  text/caption states and its duration prediction against the teacher's.

Every saved ``model.safetensors`` is a complete drop-in checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from batch_synth import set_sdpa_backend
from distill_data import (
    ConditionItem,
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
from distill_lib import (
    TEXT_PATH_PREFIXES,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from irodori_tts.config import TrainConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.optim import build_optimizer
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


def set_train_mode(student: TextToLatentRFDiT) -> None:
    """Train mode, except the frozen duration predictor (its dropout stays off)."""
    student.train()
    student.duration_predictor.eval()


def detached(encoded: EncodedBatch) -> EncodedBatch:
    def det(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach()

    return EncodedBatch(
        text_state=det(encoded.text_state),
        text_mask=encoded.text_mask,
        speaker_state=det(encoded.speaker_state),
        speaker_mask=encoded.speaker_mask,
        caption_state=det(encoded.caption_state),
        caption_mask=encoded.caption_mask,
        frames=encoded.frames,
        latent_mask=encoded.latent_mask,
        log_frames=det(encoded.log_frames),
    )


@dataclass
class Targets:
    x_t: torch.Tensor
    t: torch.Tensor
    delta: torch.Tensor
    u: torch.Tensor
    repeats: int  # number of trajectory points per item
    random_index: torch.Tensor | None  # items that also got a random point

    def item_index(self, batch: int) -> torch.Tensor:
        """Which condition item each target row belongs to."""
        device = self.x_t.device
        parts = [torch.arange(batch, device=device)] * self.repeats
        if self.random_index is not None:
            parts.append(self.random_index)
        return torch.cat(parts)


@torch.no_grad()
def teacher_targets(
    teacher: TextToLatentRFDiT,
    student: TextToLatentRFDiT,
    teacher_enc: EncodedBatch,
    student_enc: EncodedBatch,
    x_1: torch.Tensor,
    *,
    on_policy: bool,
    random_points: int,
    rng: random.Random,
) -> Targets:
    batch, device = x_1.shape[0], x_1.device

    def full(value: float, n: int = batch) -> torch.Tensor:
        return torch.full((n,), value, device=device)

    teacher_path = meanflow_rollout(teacher, teacher_enc, x_1)
    if on_policy:
        student.eval()
        states = meanflow_rollout(student, student_enc, x_1).states
        set_train_mode(student)
        outputs = [
            model_output(teacher, teacher_enc, x, full(t), full(d))
            for x, t, d in zip(states, teacher_path.times, teacher_path.deltas, strict=True)
        ]
    else:
        states, outputs = teacher_path.states, teacher_path.outputs
    xs, us = list(states), list(outputs)
    ts = [full(t) for t in teacher_path.times]
    ds = [full(d) for d in teacher_path.deltas]
    index = None
    if random_points > 0:
        index = torch.tensor(rng.sample(range(batch), min(random_points, batch)), device=device)
        t = torch.rand(len(index), device=device).clamp_min(0.02)
        d = torch.rand(len(index), device=device) * t
        x_t = ((1.0 - t)[:, None, None] * teacher_path.final[index] + t[:, None, None] * x_1[index]).to(x_1.dtype)
        xs.append(x_t), ts.append(t), ds.append(d)
        us.append(model_output(teacher, teacher_enc.select(index), x_t, t, d))
    return Targets(
        x_t=torch.cat(xs),
        t=torch.cat(ts),
        delta=torch.cat(ds),
        u=torch.cat(us),
        repeats=len(states),
        random_index=index,
    )


class BlockOutputs:
    """Records the outputs of selected DiT blocks during a forward pass."""

    def __init__(self, model: TextToLatentRFDiT, indices: list[int]) -> None:
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = [model.blocks[i].register_forward_hook(self._hook(i)) for i in indices]

    def _hook(self, index: int) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        def hook(_: torch.nn.Module, __: tuple, output: torch.Tensor) -> None:
            self.outputs[index] = output

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def relative_sq_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).float()
    den = ((target.float() ** 2) * weight).sum()
    return (((pred.float() - target.float()) ** 2) * weight).sum() / den.clamp_min(1e-6)


def text_path_losses(student_enc: EncodedBatch, teacher_enc: EncodedBatch) -> dict[str, torch.Tensor]:
    losses = {
        "text_state": relative_sq_error(student_enc.text_state, teacher_enc.text_state, teacher_enc.text_mask),
        "duration": F.smooth_l1_loss(student_enc.log_frames.float(), teacher_enc.log_frames.float(), beta=0.1),
    }
    if teacher_enc.caption_state is not None and bool(teacher_enc.caption_mask.any()):
        losses["caption_state"] = relative_sq_error(
            student_enc.caption_state, teacher_enc.caption_state, teacher_enc.caption_mask
        )
    return losses


@torch.no_grad()
def evaluate(
    teacher: TextToLatentRFDiT,
    student: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    val_batches: list[tuple[list[ConditionItem], EncodedBatch, torch.Tensor]],
    student_encodes: bool,
) -> dict[str, float]:
    student.eval()
    final_errors, step_errors = [], [[] for _ in range(4)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for items, teacher_enc, x_1 in val_batches:
            student_enc = teacher_enc
            if student_encodes:
                student_enc = encode_batch(student, tokenizer, items, device=x_1.device, frames=teacher_enc.frames)
            teacher_path = meanflow_rollout(teacher, teacher_enc, x_1)
            student_path = meanflow_rollout(student, student_enc, x_1)
            mask = teacher_enc.latent_mask
            final_errors.append(masked_relative_error(student_path.final, teacher_path.final, mask))
            for k, (x, t, d, u) in enumerate(
                zip(teacher_path.states, teacher_path.times, teacher_path.deltas, teacher_path.outputs, strict=True)
            ):
                batch = x.shape[0]
                t_vec = torch.full((batch,), t, device=x.device)
                pred = model_output(student, student_enc, x, t_vec, torch.full((batch,), d, device=x.device))
                step_errors[k].append(masked_relative_error(pred, u, mask))
    set_train_mode(student)
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
    parser.add_argument("--train-dit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-text", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4, help="DiT learning rate.")
    parser.add_argument("--text-lr", type=float, default=3e-5)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument(
        "--hidden-weight",
        type=float,
        default=0.0,
        help="Weight of the per-block hidden-state loss (student block i vs teacher block --layer-map[i]).",
    )
    parser.add_argument("--layer-map", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--num-val", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--micro-batches",
        type=int,
        default=1,
        help="Split each step's student forward/backward into length-sorted chunks to save memory.",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=["efficient", "cudnn"],
        default="efficient",
        help="cudnn rebuilds its plan for every new sequence length in bf16 (slow here).",
    )
    args = parser.parse_args()
    if not (args.train_dit or args.train_text):
        parser.error("nothing to train: pass --train-text or keep --train-dit")

    set_sdpa_backend(args.sdpa_backend)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher_ckpt = load_checkpoint(args.teacher)
    student_ckpt = load_checkpoint(args.student_init)
    teacher = build_model(teacher_ckpt.flat_config, teacher_ckpt.text_encoder_config, teacher_ckpt.state)
    teacher = teacher.cuda().eval().requires_grad_(False)
    student = build_model(student_ckpt.flat_config, student_ckpt.text_encoder_config, student_ckpt.state).cuda()
    set_train_mode(student)
    groups: dict[str, list[torch.nn.Parameter]] = {"dit": [], "text": []}
    for name, param in student.named_parameters():
        group = None
        if args.train_dit and name.startswith(DIT_PREFIXES):
            group = "dit"
        elif args.train_text and name.startswith(TEXT_PATH_PREFIXES):
            group = "text"
        param.requires_grad_(group is not None)
        if group is not None:
            groups[group].append(param)
    if args.optimizer == "muon":
        optimizer = build_optimizer(
            student,
            TrainConfig(
                optimizer="muon",
                learning_rate=args.lr,
                pretrained_text_encoder_learning_rate=args.text_lr,
                weight_decay=args.weight_decay,
            ),
        )
    else:
        base = {"dit": args.lr, "text": args.text_lr}
        optimizer = torch.optim.AdamW(
            [{"params": params, "lr": base[name]} for name, params in groups.items() if params],
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    params = [p for params in groups.values() for p in params]
    if args.hidden_weight > 0 and (args.layer_map is None or len(args.layer_map) != len(student.blocks)):
        parser.error("--hidden-weight needs --layer-map with one teacher block per student block")
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
        items = val_sampler.sample(16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = encode_batch(teacher, tokenizer, items, device=device)
        noise = initial_noise(encoded, teacher.cfg.patched_latent_dim, val_generator, torch.float32)
        val_batches.append((items, encoded, noise))

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

    def run_eval() -> dict[str, float]:
        return evaluate(teacher, student, tokenizer, val_batches, args.train_text)

    log({"step": 0, **run_eval()})
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    start = time.perf_counter()
    running: dict[str, float] = {}
    running_onpolicy = 0
    for step in range(args.steps):
        scale = lr_scale(step, args.warmup, args.steps, args.min_lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        on_policy = step >= args.onpolicy_start and rng.random() < args.onpolicy_prob
        items = sampler.sample(args.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_enc = encode_batch(teacher, tokenizer, items, device=device)
            student_enc = teacher_enc
            if args.train_text:
                student_enc = encode_batch(
                    student, tokenizer, items, device=device, frames=teacher_enc.frames, grad=True
                )
            x_1 = initial_noise(teacher_enc, teacher.cfg.patched_latent_dim, generator, torch.float32)
            targets = teacher_targets(
                teacher,
                student,
                teacher_enc,
                detached(student_enc),
                x_1,
                on_policy=on_policy,
                random_points=args.random_points,
                rng=rng,
            )
        # Gradient caching: the DiT chunks backpropagate into detached leaf copies of the
        # student's text/caption states; one final backward carries the summed leaf
        # gradients (plus the text-path losses) through the student's encoders.
        leaves: dict[str, torch.Tensor] = {}
        cond_enc = student_enc
        if args.train_text:
            cond_enc = detached(student_enc)
            for name in ("text_state", "caption_state"):
                leaf = getattr(cond_enc, name)
                if leaf is not None:
                    leaves[name] = leaf.requires_grad_(True)
        item_index = targets.item_index(len(items))
        losses: dict[str, torch.Tensor] = {}
        total = targets.x_t.shape[0]
        frames = [cond_enc.frames[i] for i in item_index.tolist()]
        order = torch.tensor(sorted(range(total), key=lambda i: frames[i]), device=device)
        u_loss = torch.zeros((), device=device)
        hidden_loss = torch.zeros((), device=device)
        for index in order.chunk(args.micro_batches):
            sub = cond_enc.select(item_index[index])  # a fresh graph per chunk from the leaves
            length = max(sub.frames)
            sub.latent_mask = sub.latent_mask[:, :length]
            x_t, t, delta = targets.x_t[index, :length], targets.t[index], targets.delta[index]
            weight = len(index) / total
            if args.hidden_weight > 0:
                teacher_sub = teacher_enc.select(item_index[index])
                teacher_sub.latent_mask = sub.latent_mask
                teacher_hidden = BlockOutputs(teacher, sorted(set(args.layer_map)))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model_output(teacher, teacher_sub, x_t, t, delta)
                teacher_hidden.remove()
                student_hidden = BlockOutputs(student, list(range(len(student.blocks))))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model_output(student, sub, x_t, t, delta)
            chunk_loss = masked_utterance_mse(pred, targets.u[index, :length], sub.latent_mask) * weight
            u_loss += chunk_loss.detach()
            if args.hidden_weight > 0:
                student_hidden.remove()
                chunk_hidden = sum(
                    relative_sq_error(student_hidden.outputs[i], teacher_hidden.outputs[j], sub.latent_mask)
                    for i, j in enumerate(args.layer_map)
                ) / len(args.layer_map)
                hidden_loss += chunk_hidden.detach() * weight
                chunk_loss = chunk_loss + args.hidden_weight * chunk_hidden * weight
            chunk_loss.backward()
        if args.hidden_weight > 0:
            losses["hidden"] = hidden_loss
        if args.train_text:
            text_loss = torch.zeros((), device=device)
            for name, value in text_path_losses(student_enc, teacher_enc).items():
                losses[name] = value.detach()
                weight = args.duration_weight if name == "duration" else args.feature_weight
                text_loss = text_loss + weight * value
            for name, leaf in leaves.items():
                if leaf.grad is not None:
                    text_loss = text_loss + (getattr(student_enc, name).float() * leaf.grad.float()).sum()
            text_loss.backward()
        losses["u"] = u_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + float(value.detach())
        running_onpolicy += int(on_policy)
        if (step + 1) % args.log_every == 0:
            log(
                {
                    "step": step + 1,
                    **{("loss" if k == "u" else f"loss_{k}"): v / args.log_every for k, v in running.items()},
                    "onpolicy_frac": running_onpolicy / args.log_every,
                    "grad_norm": float(grad_norm),
                    "lr_scale": scale,
                    "steps_per_s": (step + 1) / (time.perf_counter() - start),
                    "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            running, running_onpolicy = {}, 0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            log({"step": step + 1, **run_eval()})
        if (step + 1) % args.save_every == 0:
            save()
    save()
    print(f"saved {args.output_dir / 'model.safetensors'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
