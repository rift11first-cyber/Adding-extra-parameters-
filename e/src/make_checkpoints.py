from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expanded_model import (
    ExpandedLittleKedi,
    build_expanded,
    init_random_pre_head,
    save_expanded,
    set_pre_head_weights,
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_generator(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("llm_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate_weights"):
        raise AttributeError(f"{path} must define generate_weights(cfg, seed=0)")
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT.parent))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["random", "llm"], required=True)
    ap.add_argument("--hidden", type=int, default=3890)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--llm-prompt-sha256", default="")
    args = ap.parse_args()

    print(f"[{args.mode}] building expanded model", flush=True)
    model, cfg_dict, tok_json, tok = build_expanded(args.model_dir, hidden=args.hidden)
    base_total = sum(p.numel() for p in model.parameters()) - model.pre_head.param_count()

    generator_info = {}
    if args.mode == "random":
        init_random_pre_head(model, cfg_dict["init_std"], args.seed)
    else:
        if not args.generator:
            raise SystemExit("--generator is required for --mode llm")
        gen = load_generator(args.generator)
        print(f"[llm] calling generator {args.generator}", flush=True)
        weights = gen.generate_weights(cfg_dict, seed=args.seed)
        set_pre_head_weights(model, weights)
        generator_info = {
            "generator_file": str(Path(args.generator).resolve()),
            "generator_sha256": sha256_file(args.generator),
            "llm_model": args.llm_model,
            "llm_prompt_sha256": args.llm_prompt_sha256,
        }

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    counts = model.param_counts()
    if counts["pre_head"] != 999858:
        raise RuntimeError(f"unexpected new-layer parameter count: {counts['pre_head']}")
    if counts["total"] != 2990770:
        raise RuntimeError(f"unexpected expanded parameter count: {counts['total']}")

    state = model.state_dict()
    manifest = {
        "mode": args.mode,
        "insertion": "after_final_rmsnorm_before_tied_head",
        "activation": "gelu",
        "hidden": args.hidden,
        "seed": args.seed,
        "base_checkpoint": str(Path(args.model_dir).resolve() / "model.safetensors"),
        "base_checkpoint_sha256": sha256_file(Path(args.model_dir) / "model.safetensors"),
        "base_total": base_total,
        "new_layer_params": counts["pre_head"],
        "expanded_total": counts["total"],
        "state_sha256": sha256_state(state),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "dtype": "float32",
        "bias_policy": "zero" if args.mode == "random" else "generator",
        **generator_info,
    }

    out = save_expanded(args.out, model, cfg_dict, tok_json, manifest)
    manifest["model_file_sha256"] = sha256_file(out / "model.safetensors")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(out), **counts, "manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
