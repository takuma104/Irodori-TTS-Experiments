#!/usr/bin/env python3
"""Initialize a smaller-DiT student from the teacher as a drop-in checkpoint (plan Phase 2).

    PYTHONPATH=third-party/Irodori-TTS:scripts third-party/Irodori-TTS/.venv/bin/python \
        scripts/make_dit_student.py --keep-blocks 0 7 8 9 10 11 --output-dir outputs/students/dit_d6/init

The student keeps the teacher's DiT blocks listed in ``--keep-blocks`` (in
order) and everything else (text path, speaker encoder, duration predictor,
timestep/interval embeddings, in/out projections) unchanged. With
``--mlp-hidden``, each kept block's SwiGLU is cut to that many hidden units,
keeping the units with the largest mean |h| x ||w2 column|| over teacher
MeanFlow rollouts on sampled conditions (calibration data). Defaults follow
the zero-shot sensitivity analysis (``scripts/feasibility/dit_prune_sensitivity.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from distill_data import (
    ConditionSampler,
    encode_batch,
    initial_noise,
    load_refs,
    load_texts,
    meanflow_rollout,
)
from distill_lib import build_model, count_parameters, load_checkpoint, save_checkpoint
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from voice_captions import voice_captions

DIT_PREFIXES = ("blocks.", "cond_module.", "delta_cond_module.", "in_proj.", "out_norm.", "out_proj.")


@torch.no_grad()
def mlp_importance(
    teacher: TextToLatentRFDiT,
    tokenizer: PretrainedTextTokenizer,
    sampler: ConditionSampler,
    batches: int,
    batch_size: int,
) -> list[torch.Tensor]:
    totals = [torch.zeros(block.mlp.w2.in_features, device="cuda") for block in teacher.blocks]
    hooks = []
    for index, block in enumerate(teacher.blocks):

        def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...], index: int = index) -> None:
            totals[index] += inputs[0].abs().flatten(0, -2).float().sum(0)

        hooks.append(block.mlp.w2.register_forward_pre_hook(hook))
    generator = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(batches):
        encoded = encode_batch(teacher, tokenizer, sampler.sample(batch_size), device=torch.device("cuda"))
        noise = initial_noise(encoded, teacher.cfg.patched_latent_dim, generator, torch.float32)
        meanflow_rollout(teacher, encoded, noise)
    for handle in hooks:
        handle.remove()
    return [total * block.mlp.w2.weight.norm(dim=0) for total, block in zip(totals, teacher.blocks, strict=True)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="MF")
    parser.add_argument("--keep-blocks", type=int, nargs="+", default=[0, 7, 8, 9, 10, 11])
    parser.add_argument("--mlp-hidden", type=int, default=None)
    parser.add_argument("--texts", type=Path, default=Path("data/corpus/wiki_val.txt"))
    parser.add_argument("--refs", type=Path, nargs="+", default=[Path("data/refs/real_pool.pt")])
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--calib-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    teacher_ckpt = load_checkpoint(args.teacher)
    model_dim = int(teacher_ckpt.flat_config["model_dim"])
    flat_config = dict(teacher_ckpt.flat_config, num_layers=len(args.keep_blocks))
    if args.mlp_hidden is not None:
        ratio = args.mlp_hidden / model_dim
        if int(model_dim * ratio) != args.mlp_hidden:
            raise ValueError(f"mlp_hidden={args.mlp_hidden} is not representable as int({model_dim} * ratio).")
        flat_config["mlp_ratio"] = ratio

    keep_mlp: dict[int, torch.Tensor] = {}
    if args.mlp_hidden is not None:
        teacher = build_model(teacher_ckpt.flat_config, teacher_ckpt.text_encoder_config, teacher_ckpt.state)
        teacher = teacher.cuda().eval()
        tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=str(teacher_ckpt.tokenizer_dir), add_bos=True, local_files_only=True
        )
        sampler = ConditionSampler(load_texts([args.texts]), load_refs(args.refs), voice_captions(2000))
        scores = mlp_importance(teacher, tokenizer, sampler, args.calib_batches, args.calib_batch_size)
        for old in args.keep_blocks:
            keep_mlp[old] = scores[old].topk(args.mlp_hidden).indices.sort().values.cpu()
        del teacher
        torch.cuda.empty_cache()

    state: dict[str, torch.Tensor] = {}
    for key, tensor in teacher_ckpt.state.items():
        if not key.startswith("blocks."):
            state[key] = tensor
            continue
        old, rest = key[len("blocks.") :].split(".", 1)
        if int(old) not in args.keep_blocks:
            continue
        new = args.keep_blocks.index(int(old))
        if int(old) in keep_mlp:
            units = keep_mlp[int(old)]
            if rest in ("mlp.w1.weight", "mlp.w3.weight"):
                tensor = tensor[units]
            elif rest == "mlp.w2.weight":
                tensor = tensor[:, units]
        state[f"blocks.{new}.{rest}"] = tensor.contiguous()

    model = build_model(flat_config, teacher_ckpt.text_encoder_config, state)
    path = save_checkpoint(
        model,
        flat_config=flat_config,
        text_encoder_config=teacher_ckpt.text_encoder_config,
        output_dir=args.output_dir,
        tokenizer_dir=teacher_ckpt.tokenizer_dir,
    )
    print(
        f"wrote {path}: total={count_parameters(model.state_dict()) / 1e6:.1f}M "
        f"DiT={count_parameters(model.state_dict(), DIT_PREFIXES) / 1e6:.1f}M "
        f"(teacher DiT={count_parameters(teacher_ckpt.state, DIT_PREFIXES) / 1e6:.1f}M)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
