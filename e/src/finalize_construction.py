from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expanded_model import (
    NEW_LAYER_KEYS,
    build_expanded,
    init_random_pre_head,
    load_expanded,
    load_original,
)

DEFAULT_BASE_DIR = ROOT.parent
DEFAULT_B_DIR = ROOT / "checkpoints" / "random_seed0"
DEFAULT_C_DIR = ROOT / "checkpoints" / "llm_generated"
DEFAULT_VERIFY_PATH = ROOT / "results" / "verification.json"
DEFAULT_OUT_PATH = ROOT / "results" / "construction_summary.json"
DEFAULT_GEN_PATH = ROOT / "src" / "llm_generator.py"
DEFAULT_PROMPT_PATH = ROOT / "src" / "llm_generator_prompt.txt"

BASE_SHA = "e97c3a4c4cf0a1b00119cb350297fa5ae6e81fdead6cf56381a43a6ac9f7e9ce"
BASE_TOTAL = 1_990_912
EXPANDED_TOTAL = 2_990_770
NEW_PARAMS = 999_858
EXPECTED_SHAPES = {
    "pre_head.w1": (3890, 128),
    "pre_head.b1": (3890,),
    "pre_head.w2": (128, 3890),
    "pre_head.b2": (128,),
}
EXPECTED_FILES = ("model.safetensors", "config.json", "tokenizer.json")


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


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("llm_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate_weights"):
        raise AttributeError(f"{path} must define generate_weights")
    return module


def embedded_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as fh:
        return dict(fh.metadata() or {})


def file_hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in EXPECTED_FILES}


def check_checkpoint(
    name: str,
    ckpt_dir: Path,
    base_state: dict[str, torch.Tensor],
    base_cfg: dict,
    base_hashes: dict[str, str],
    model_a,
    x: torch.Tensor,
    a_logits: torch.Tensor,
) -> tuple[dict, dict]:
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    state = load_file(str(ckpt_dir / "model.safetensors"))
    hashes = file_hashes(ckpt_dir)
    meta = embedded_metadata(ckpt_dir / "model.safetensors")

    base_keys = set(base_state)
    ckpt_keys = set(state)
    extra = sorted(ckpt_keys - base_keys)
    missing = sorted(base_keys - ckpt_keys)
    base_equal = all(torch.equal(base_state[k], state[k]) for k in base_keys)
    shape_ok = all(tuple(state[k].shape) == EXPECTED_SHAPES[k] for k in NEW_LAYER_KEYS)
    dtype_ok = all(state[k].dtype == torch.float32 for k in NEW_LAYER_KEYS)
    finite_ok = all(torch.isfinite(state[k]).all().item() for k in NEW_LAYER_KEYS)
    config_equal = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8")) == base_cfg

    model, cfg, _, tok = load_expanded(ckpt_dir)
    counts = model.param_counts()
    total = int(counts["total"])
    pre_head = int(counts["pre_head"])
    base_total = int(counts["base_total"])
    no_grad = all(not p.requires_grad for p in model.parameters())
    with torch.inference_mode():
        logits = model(x).logits
    forward_ok = bool(torch.isfinite(logits).all())
    logits_diff = float((logits - a_logits).abs().max())

    checks = {
        f"{name}_manifest_base_sha": manifest.get("base_checkpoint_sha256") == BASE_SHA,
        f"{name}_manifest_base_total": manifest.get("base_total") == BASE_TOTAL,
        f"{name}_manifest_new_params": manifest.get("new_layer_params") == NEW_PARAMS,
        f"{name}_manifest_expanded_total": manifest.get("expanded_total") == EXPANDED_TOTAL,
        f"{name}_manifest_insertion": manifest.get("insertion") == "after_final_rmsnorm_before_tied_head",
        f"{name}_manifest_activation": manifest.get("activation") == "gelu",
        f"{name}_manifest_hidden": manifest.get("hidden") == 3890,
        f"{name}_manifest_dtype": manifest.get("dtype") == "float32",
        f"{name}_manifest_seed": manifest.get("seed") == 0,
        f"{name}_manifest_state_sha": manifest.get("state_sha256") == sha256_state(state),
        f"{name}_manifest_file_sha": manifest.get("model_file_sha256") == hashes["model.safetensors"],
        f"{name}_embedded_state_sha": meta.get("state_sha256") == manifest.get("state_sha256"),
        f"{name}_embedded_mode": meta.get("mode") == manifest.get("mode"),
        f"{name}_base_keys_missing": not missing,
        f"{name}_extra_keys": extra == sorted(NEW_LAYER_KEYS),
        f"{name}_base_tensors_equal": base_equal,
        f"{name}_shapes": shape_ok,
        f"{name}_dtype": dtype_ok,
        f"{name}_finite": finite_ok,
        f"{name}_base_total": base_total == BASE_TOTAL,
        f"{name}_total": total == EXPANDED_TOTAL,
        f"{name}_pre_head": pre_head == NEW_PARAMS,
        f"{name}_param_counts": counts["total"] == EXPANDED_TOTAL and counts["pre_head"] == NEW_PARAMS,
        f"{name}_no_grad": no_grad,
        f"{name}_forward_finite": forward_ok,
        f"{name}_differs_from_A": logits_diff > 0,
        f"{name}_config_matches_base": config_equal,
        f"{name}_tokenizer_matches_base": hashes["tokenizer.json"] == base_hashes["tokenizer.json"],
    }

    stats = {}
    for key in NEW_LAYER_KEYS:
        t = state[key]
        stats[key] = {
            "shape": list(t.shape),
            "rms": float(t.square().mean().sqrt()),
            "max_abs": float(t.abs().max()),
        }

    result = {
        "checkpoint": str(ckpt_dir.resolve()),
        "manifest": manifest,
        "file_sha256": hashes,
        "state_sha256": sha256_state(state),
        "param_counts": counts,
        "new_layer_shapes": {k: list(state[k].shape) for k in NEW_LAYER_KEYS},
        "new_layer_stats": stats,
        "base_tensors_identical": base_equal,
        "missing_keys": missing,
        "extra_keys": extra,
        "requires_grad_any": any(p.requires_grad for p in model.parameters()),
        "logits_finite": forward_ok,
        "logits_max_abs_diff_vs_A": logits_diff,
        "embedded_metadata": meta,
        "checks": checks,
    }
    return result, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_BASE_DIR))
    ap.add_argument("--random-dir", default=str(DEFAULT_B_DIR))
    ap.add_argument("--llm-dir", default=str(DEFAULT_C_DIR))
    ap.add_argument("--verify", default=str(DEFAULT_VERIFY_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--generator", default=str(DEFAULT_GEN_PATH))
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH))
    args = ap.parse_args()

    base_dir = Path(args.model_dir).resolve()
    b_dir = Path(args.random_dir).resolve()
    c_dir = Path(args.llm_dir).resolve()
    verify_path = Path(args.verify).resolve()
    out_path = Path(args.out).resolve()
    gen_path = Path(args.generator).resolve()
    prompt_path = Path(args.prompt).resolve()

    base_path = base_dir / "model.safetensors"
    base_hash = sha256_file(base_path)
    base_state = load_file(str(base_path))
    base_hashes = {
        "model.safetensors": base_hash,
        "config.json": sha256_file(base_dir / "config.json"),
        "tokenizer.json": sha256_file(base_dir / "tokenizer.json"),
    }
    base_cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    model_a, cfg_a, _, tok = load_original(base_dir)
    base_param_count = sum(p.numel() for p in model_a.parameters())

    text = "The Earth revolves around the Sun. Water is made of hydrogen and oxygen."
    ids = [tok.eot_id] + tok.encode(text)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        a_logits = model_a(x).logits

    b_result, b_checks = check_checkpoint("B", b_dir, base_state, base_cfg, base_hashes, model_a, x, a_logits)
    c_result, c_checks = check_checkpoint("C", c_dir, base_state, base_cfg, base_hashes, model_a, x, a_logits)

    gen = load_generator(gen_path)
    gen_weights = gen.generate_weights(base_cfg, seed=0)
    c_state = load_file(str(c_dir / "model.safetensors"))
    gen_equal = all(torch.equal(gen_weights[k], c_state[k]) for k in NEW_LAYER_KEYS)
    gen_keys_ok = set(gen_weights) == set(NEW_LAYER_KEYS)

    b_tmp, b_cfg, _, _ = build_expanded(base_dir, hidden=3890)
    init_random_pre_head(b_tmp, b_cfg["init_std"], 0)
    b_state = load_file(str(b_dir / "model.safetensors"))
    b_repro = all(torch.equal(b_tmp.state_dict()[k], b_state[k]) for k in NEW_LAYER_KEYS)

    verification = json.loads(verify_path.read_text(encoding="utf-8"))
    gen_sha = sha256_file(gen_path)
    prompt_sha = sha256_file(prompt_path)

    b_bias_zero = bool((b_state["pre_head.b1"] == 0).all() and (b_state["pre_head.b2"] == 0).all())
    c_bias_zero = bool((c_state["pre_head.b1"] == 0).all() and (c_state["pre_head.b2"] == 0).all())

    checks = {
        "original_hash_matches_known": base_hash == BASE_SHA,
        "original_params_1990912": base_param_count == BASE_TOTAL,
        "verification_all_passed": bool(verification.get("all_passed")),
        "verification_base_hash": verification.get("base_checkpoint_sha256") == BASE_SHA,
        "B_bias_zero": b_bias_zero,
        "B_reproducible_seed0": b_repro,
        "C_bias_zero": c_bias_zero,
        "C_generator_keys": gen_keys_ok,
        "C_generator_reproduces_weights": gen_equal,
        "C_generator_sha_matches_manifest": c_result["manifest"].get("generator_sha256") == gen_sha,
        "C_prompt_sha_matches_manifest": c_result["manifest"].get("llm_prompt_sha256") == prompt_sha,
        "C_generator_file_exists": gen_path.exists(),
        "C_prompt_file_exists": prompt_path.exists(),
        "upstream_commit_pinned": (ROOT / "vendor" / "LITTEKEDI_COMMIT").read_text(encoding="utf-8").strip()
        == "a282a8fdc945d783e05b25b8dcd42fc493cf8f23",
    }
    checks.update(b_checks)
    checks.update(c_checks)

    result = {
        "construction_complete": all(checks.values()),
        "original": {
            "path": str(base_dir),
            "model_sha256": base_hash,
            "config_sha256": base_hashes["config.json"],
            "tokenizer_sha256": base_hashes["tokenizer.json"],
            "total_params": base_param_count,
            "state_tensors": len(base_state),
            "state_scalars_including_buffers": sum(t.numel() for t in base_state.values()),
        },
        "upstream": {
            "path": str((ROOT / "vendor" / "littlekedi").resolve()),
            "commit": (ROOT / "vendor" / "LITTEKEDI_COMMIT").read_text(encoding="utf-8").strip(),
        },
        "new_layer": {
            "insertion": "after_final_rmsnorm_before_tied_head",
            "activation": "gelu",
            "hidden": 3890,
            "shapes": {k: list(v) for k, v in EXPECTED_SHAPES.items()},
            "params": NEW_PARAMS,
            "expanded_total": EXPANDED_TOTAL,
        },
        "random_control": b_result,
        "llm_generated": c_result,
        "verification": verification,
        "checks": checks,
        "all_passed": all(checks.values()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    failed = [k for k, v in checks.items() if not v]
    print(json.dumps({
        "construction_complete": result["construction_complete"],
        "all_passed": result["all_passed"],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "failed": failed,
        "output": str(out_path),
        "B_state_sha256": b_result["state_sha256"],
        "C_state_sha256": c_result["state_sha256"],
        "C_generator_sha256": gen_sha,
        "C_prompt_sha256": prompt_sha,
    }, indent=2), flush=True)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
