#!/usr/bin/env python3
"""Kana-substitution self-distillation of the text encoder (plan §5.1).

    PYTHONPATH=Irodori-TTS uv run --project Irodori-TTS --no-sync \
        python scripts/train_kana_distill.py \
        --teacher-dir outputs/yomi_pilot/teacher --output-dir outputs/yomi_pilot/s1 \
        --scope projector

The frozen model is the teacher: for target rows it sees ``kana_text`` (the
word written in hiragana), for contrast/general rows the kanji text. The student
text encoder (``student_text.py``) sees the kanji text, and the frozen DiT runs
on both conditions with the same x_t = (1 - t) x0 + t noise, where x0 is the
teacher latent. The loss matches the conditional velocities (utterance-mean
MSE over valid frames) plus the predicted log duration. Speaker conditioning is
dropped for a fraction of samples on both sides, because independent CFG also
evaluates the text-conditioned, speaker-dropped branch.

The frozen DiT runs in bf16; the text path (teacher and student encoders) runs in
fp32, matching inference, where a bf16 text encoder costs ~0.5pt on JKYB.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from batch_synth import load_runtime, set_sdpa_backend
from irodori_tts.duration import build_duration_features
from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest
from irodori_tts.rf import sample_stratified_logit_normal_t
from irodori_tts.text_normalization import normalize_text
from student_text import StudentTextEncoder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class Sample:
    key: str
    student_text: str
    teacher_text: str
    ref_wav: str | None
    role: str
    bucket: str
    latent_path: Path


def misread_keys(results_dirs: list[Path]) -> set[str]:
    """Keys whose teacher audio was scored wrong by ``jkyb-eval tts``."""
    keys: set[str] = set()
    for results_dir in results_dirs:
        for row in read_jsonl(results_dir / "details" / "all.jsonl"):
            if not row["target_exact"]:
                keys.add(row["key"])
    return keys


def load_samples(
    teacher_dirs: list[Path], split: str, exclude: set[str] | None = None
) -> list[Sample]:
    samples: list[Sample] = []
    for teacher_dir in teacher_dirs:
        for row in read_jsonl(teacher_dir / "manifest.jsonl"):
            if row.get("split", "train") != split:
                continue
            if exclude and row["key"] in exclude:
                continue
            samples.append(
                Sample(
                    key=row["key"],
                    student_text=row["kanji_text"],
                    teacher_text=row["text"],
                    ref_wav=row["ref_wav"],
                    role=row.get("role") or "general",
                    bucket=row.get("bucket") or "H",
                    latent_path=Path(row["latent_path"])
                    if "latent_path" in row
                    else teacher_dir / "latents" / f"{row['key']}.pt",
                )
            )
    return samples


class ReferenceCache:
    """Patched reference latents per reference wav (None = no reference)."""

    def __init__(self, runtime: InferenceRuntime) -> None:
        self.runtime = runtime
        self.cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def get(self, ref_wav: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        if ref_wav not in self.cache:
            latent, mask = self.runtime._load_reference_latent(
                req=SamplingRequest(text="", ref_wav=ref_wav, no_ref=ref_wav is None),
                batch_size=1,
                messages=[],
            )
            self.cache[ref_wav] = (latent[0], mask[0])
        return self.cache[ref_wav]


def pad_stack(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(t.shape[0] for t in tensors)
    out = tensors[0].new_zeros((len(tensors), length, *tensors[0].shape[1:]))
    mask = torch.zeros(
        (len(tensors), length), dtype=torch.bool, device=tensors[0].device
    )
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return out, mask


class Distiller:
    def __init__(
        self,
        runtime: InferenceRuntime,
        student: StudentTextEncoder,
        *,
        speaker_dropout: float,
        duration_weight: float,
        keep_weight: float,
    ) -> None:
        self.runtime = runtime
        self.model = runtime.model
        self.student = student
        self.device = runtime.model_device
        self.refs = ReferenceCache(runtime)
        self.speaker_dropout = speaker_dropout
        self.duration_weight = duration_weight
        self.keep_weight = keep_weight
        self.text_max_len = int(runtime.default_text_max_len)

    def _tokens(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.runtime.tokenizer.batch_encode(
            texts, max_length=self.text_max_len
        )
        return ids.to(self.device), mask.to(self.device)

    def _duration(
        self,
        texts: list[str],
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        has_speaker: torch.Tensor,
    ) -> torch.Tensor:
        features = build_duration_features(
            texts,
            token_counts=text_mask.sum(dim=1),
            max_text_len=self.text_max_len,
            has_speaker=has_speaker,
        ).to(self.device)
        has_caption = None
        if self.runtime.model_cfg.use_caption_condition:
            has_caption = torch.zeros_like(has_speaker)
        return self.model.predict_duration_log_frames(
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
            duration_features=features,
            has_speaker=has_speaker,
            has_caption=has_caption,
            detach_condition=False,
        ).reshape(-1)

    def loss(self, batch: list[Sample]) -> tuple[torch.Tensor, dict[str, float]]:
        model = self.model
        bsz = len(batch)
        student_texts = [normalize_text(s.student_text).strip() for s in batch]
        teacher_texts = [normalize_text(s.teacher_text).strip() for s in batch]
        x0, latent_mask = pad_stack(
            [
                torch.load(s.latent_path, weights_only=True).to(self.device)
                for s in batch
            ]
        )
        refs = [self.refs.get(s.ref_wav) for s in batch]
        ref_latent, _ = pad_stack([r[0] for r in refs])
        ref_mask = pad_stack([r[1].float() for r in refs])[0].bool()
        drop = torch.rand(bsz, device=self.device) < self.speaker_dropout
        ref_mask = ref_mask & ~drop[:, None]
        has_speaker = ref_mask.any(dim=1)
        caption_ids = caption_mask = None
        if self.runtime.model_cfg.use_caption_condition:
            caption_ids, caption_mask = self.runtime.caption_tokenizer.batch_encode(
                [""] * bsz, max_length=int(self.runtime.default_caption_max_len)
            )
            caption_mask.zero_()
            caption_ids, caption_mask = (
                caption_ids.to(self.device),
                caption_mask.to(self.device),
            )

        t = sample_stratified_logit_normal_t(batch_size=bsz, device=self.device)
        noise = torch.randn_like(x0)
        x_t = ((1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise).to(
            torch.bfloat16
        )
        t = t.to(torch.bfloat16)

        # The text path runs in fp32 (as in inference); the frozen DiT in bf16.
        dit_dtype = model.in_proj.weight.dtype
        with torch.no_grad():
            teacher_ids, teacher_mask = self._tokens(teacher_texts)
            (
                teacher_state,
                teacher_mask,
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            ) = model.encode_conditions(
                text_input_ids=teacher_ids,
                text_mask=teacher_mask,
                ref_latent=ref_latent.to(dit_dtype),
                ref_mask=ref_mask,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
            )
            teacher_state = teacher_state.to(dit_dtype)
            if caption_state is not None:
                caption_state = caption_state.to(dit_dtype)
            v_teacher = model.forward_with_encoded_conditions(
                x_t=x_t,
                t=t,
                text_state=teacher_state,
                text_mask=teacher_mask,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                caption_state=caption_state,
                caption_mask=caption_mask,
                latent_mask=latent_mask,
            ).float()
            dur_teacher = self._duration(
                teacher_texts,
                teacher_state,
                teacher_mask,
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
                has_speaker,
            ).float()
        student_ids, student_mask = self._tokens(student_texts)
        student_state = self.student.encode(
            model.pretrained_text_backbone, student_ids, student_mask
        ).to(dit_dtype)
        v_student = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=student_state,
            text_mask=student_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
            latent_mask=latent_mask,
        ).float()
        dur_student = self._duration(
            student_texts,
            student_state,
            student_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            has_speaker,
        ).float()

        valid = latent_mask.float()
        per_sample_v = ((v_student - v_teacher) ** 2).mean(dim=-1)
        per_sample_v = (per_sample_v * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(
            1.0
        )
        per_sample_dur = torch.nn.functional.smooth_l1_loss(
            dur_student, dur_teacher, reduction="none", beta=0.1
        )
        weight = torch.tensor(
            [1.0 if s.role == "target" else self.keep_weight for s in batch],
            device=self.device,
        )
        per_sample = per_sample_v + self.duration_weight * per_sample_dur
        loss = (weight * per_sample).sum() / weight.sum()
        is_target = torch.tensor(
            [s.role == "target" for s in batch], device=self.device
        )
        per_sample_v = per_sample_v.detach()
        per_sample_dur = per_sample_dur.detach()
        metrics = {
            "loss": float(loss.detach()),
            "v_target": float(per_sample_v[is_target].mean())
            if is_target.any()
            else math.nan,
            "v_keep": float(per_sample_v[~is_target].mean())
            if (~is_target).any()
            else math.nan,
            "dur": float(per_sample_dur.mean()),
        }
        return loss, metrics


def batches(
    samples: list[Sample], batch_size: int, rng: random.Random
) -> list[list[Sample]]:
    """Length-bucketed shuffled batches (by latent file size as a length proxy)."""
    order = sorted(
        samples, key=lambda s: s.latent_path.stat().st_size + rng.random() * 2048
    )
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)
    return chunks


@torch.no_grad()
def evaluate(
    distiller: Distiller, samples: list[Sample], batch_size: int
) -> dict[str, float]:
    distiller.student.eval()
    torch.manual_seed(1234)
    sums: dict[str, list[float]] = defaultdict(list)
    for chunk in batches(samples, batch_size, random.Random(0)):
        _, metrics = distiller.loss(chunk)
        for key, value in metrics.items():
            if not math.isnan(value):
                sums[key].append(value)
    distiller.student.train()
    return {key: sum(values) / len(values) for key, values in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--teacher-results",
        type=Path,
        action="append",
        default=[],
        help="jkyb-eval results of the teacher audio; misread rows are dropped.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope", default="projector", help="projector, top<N>, or all"
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Projector / text_norm.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--speaker-dropout", type=float, default=0.2)
    parser.add_argument("--duration-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-repeat",
        type=int,
        default=1,
        help="Repeat target (kana-teacher) rows this many times per epoch.",
    )
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    set_sdpa_backend("efficient")
    runtime = load_runtime(
        precision="bf16", text_precision="fp32", codec_precision="fp32"
    )
    runtime.model.requires_grad_(False)
    runtime.model.eval()
    student = StudentTextEncoder(runtime.model, args.scope).to(runtime.model_device)
    student.train()
    distiller = Distiller(
        runtime,
        student,
        speaker_dropout=args.speaker_dropout,
        duration_weight=args.duration_weight,
        keep_weight=args.keep_weight,
    )

    exclude = misread_keys(args.teacher_results)
    train = load_samples(args.teacher_dir, "train", exclude)
    # Oversample the rows that carry the kana teacher.
    train += [s for s in train if s.role == "target"] * (args.target_repeat - 1)
    dev = load_samples(args.teacher_dir, "dev", exclude)
    print(f"excluded {len(exclude)} rows whose teacher audio was misread", flush=True)
    print(f"train={len(train)} dev={len(dev)} scope={args.scope}", flush=True)
    backbone_params = [
        p
        for n, p in student.named_parameters()
        if p.requires_grad and n.startswith("backbone.")
    ]
    head_params = [
        p
        for n, p in student.named_parameters()
        if p.requires_grad and not n.startswith("backbone.")
    ]
    groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(
        groups, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    print(
        f"trainable={sum(p.numel() for p in head_params + backbone_params):,}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    log = (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8")
    if dev:
        metrics = evaluate(distiller, dev, args.batch_size)
        print(f"step 0 dev {metrics}", flush=True)
        log.write(json.dumps({"step": 0, "dev": metrics}) + "\n")

    step = 0
    start = time.perf_counter()
    running: dict[str, list[float]] = defaultdict(list)
    while step < args.steps:
        for batch in batches(train, args.batch_size, rng):
            if step >= args.steps:
                break
            scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            progress = step / max(args.steps, 1)
            scale *= 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.9 + 0.1
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr * scale
            loss, metrics = distiller.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.trainable_parameters(), 1.0
            )
            optimizer.step()
            step += 1
            metrics["grad_norm"] = float(grad_norm)
            for key, value in metrics.items():
                if not math.isnan(value):
                    running[key].append(value)
            if step % 25 == 0:
                summary = {k: round(sum(v) / len(v), 5) for k, v in running.items()}
                elapsed = time.perf_counter() - start
                print(f"step {step} {summary} {step / elapsed:.2f} it/s", flush=True)
                log.write(json.dumps({"step": step, "train": summary}) + "\n")
                log.flush()
                running.clear()
            if dev and (step % args.eval_every == 0 or step == args.steps):
                metrics = evaluate(distiller, dev, args.batch_size)
                print(f"step {step} dev {metrics}", flush=True)
                log.write(json.dumps({"step": step, "dev": metrics}) + "\n")
                log.flush()
    student.save(args.output_dir / "student")
    print(f"saved {args.output_dir / 'student'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
