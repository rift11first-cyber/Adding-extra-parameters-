from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expanded_model import NEW_LAYER_KEYS, load_expanded, load_original


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_state(base_state: dict[str, torch.Tensor], expanded_state: dict[str, torch.Tensor]) -> dict:
    bad = []
    for key, value in base_state.items():
        if key not in expanded_state:
            bad.append({"key": key, "problem": "missing"})
        elif not torch.equal(value, expanded_state[key]):
            bad.append({"key": key, "problem": "not_equal"})
    extra = sorted(set(expanded_state) - set(base_state))
    return {"identical": not bad and extra == sorted(NEW_LAYER_KEYS), "bad": bad, "extra": extra}


def check_expanded(name: str, ckpt_dir: str, base_state: dict[str, torch.Tensor], model_a, x, a_logits) -> dict:
    model, cfg, _, tok = load_expanded(ckpt_dir)
    state = load_file(str(Path(ckpt_dir) / "model.safetensors"))
    comparison = compare_state(base_state, state)
    total = sum(p.numel() for p in model.parameters())
    pre_head = model.pre_head.param_count()
    requires_grad = {k: p.requires_grad for k, p in model.named_parameters() if k.startswith("pre_head.")}
    calls = []

    def hook(module, inputs, output):
        calls.append({"input": list(inputs[0].shape), "output": list(output.shape)})

    handle = model.pre_head.register_forward_hook(hook)
    with torch.inference_mode():
        logits = model(x).logits
        forward_calls = len(calls)
        caches = [{} for _ in model.layers]
        cached = model.forward_cached(x, caches, 0)
        cached_calls = len(calls) - forward_calls
    handle.remove()

    with torch.inference_mode():
        old_w2 = model.pre_head.w2.detach().clone()
        old_b2 = model.pre_head.b2.detach().clone()
        model.pre_head.w2.zero_()
        model.pre_head.b2.zero_()
        zero_logits = model(x).logits
        model.pre_head.w2.copy_(old_w2)
        model.pre_head.b2.copy_(old_b2)

    return {
        "name": name,
        "checkpoint": str(Path(ckpt_dir).resolve()),
        "total_params": total,
        "pre_head_params": pre_head,
        "base_params": total - pre_head,
        "base_state": comparison,
        "requires_grad_pre_head": requires_grad,
        "forward_hook_calls": forward_calls,
        "cached_hook_calls": cached_calls,
        "hook_shapes": calls[:2],
        "logits_finite": bool(torch.isfinite(logits).all()),
        "cached_logits_finite": bool(torch.isfinite(cached).all()),
        "logits_differs_from_A": float((logits - a_logits).abs().max()),
        "zero_branch_max_abs_diff_vs_A": float((zero_logits - a_logits).abs().max()),
        "zero_branch_allclose": bool(torch.allclose(zero_logits, a_logits, atol=1e-6, rtol=1e-5)),
        "cached_logits_shape": list(cached.shape),
        "state_keys": sorted(state.keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT.parent))
    ap.add_argument("--random-dir", required=True)
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "verification.json"))
    args = ap.parse_args()

    base_path = Path(args.model_dir) / "model.safetensors"
    base_hash = sha256_file(base_path)
    base_state = load_file(str(base_path))
    model_a, cfg, _, tok = load_original(args.model_dir)

    text = "The Earth revolves around the Sun. Water is made of hydrogen and oxygen."
    ids = [tok.eot_id] + tok.encode(text)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        a_logits = model_a(x).logits

    result = {
        "base_checkpoint": str(base_path.resolve()),
        "base_checkpoint_sha256": base_hash,
        "base_total_params": sum(p.numel() for p in model_a.parameters()),
        "base_state_tensors": len(base_state),
        "input_shape": list(x.shape),
        "A_logits_shape": list(a_logits.shape),
        "A_logits_finite": bool(torch.isfinite(a_logits).all()),
        "random": check_expanded("B", args.random_dir, base_state, model_a, x, a_logits),
        "llm": check_expanded("C", args.llm_dir, base_state, model_a, x, a_logits),
    }

    checks = {
        "base_hash_matches_known": base_hash == "e97c3a4c4cf0a1b00119cb350297fa5ae6e81fdead6cf56381a43a6ac9f7e9ce",
        "base_params_1990912": result["base_total_params"] == 1990912,
        "random_total_2990770": result["random"]["total_params"] == 2990770,
        "llm_total_2990770": result["llm"]["total_params"] == 2990770,
        "random_pre_head_999858": result["random"]["pre_head_params"] == 999858,
        "llm_pre_head_999858": result["llm"]["pre_head_params"] == 999858,
        "random_base_identical": result["random"]["base_state"]["identical"],
        "llm_base_identical": result["llm"]["base_state"]["identical"],
        "random_connected": result["random"]["forward_hook_calls"] == 1 and result["random"]["cached_hook_calls"] == 1,
        "llm_connected": result["llm"]["forward_hook_calls"] == 1 and result["llm"]["cached_hook_calls"] == 1,
        "random_zero_branch_matches_A": result["random"]["zero_branch_allclose"],
        "llm_zero_branch_matches_A": result["llm"]["zero_branch_allclose"],
        "random_logits_differ": result["random"]["logits_differs_from_A"] > 0,
        "llm_logits_differ": result["llm"]["logits_differs_from_A"] > 0,
        "random_finite": result["random"]["logits_finite"] and result["random"]["cached_logits_finite"],
        "llm_finite": result["llm"]["logits_finite"] and result["llm"]["cached_logits_finite"],
        "random_no_grad": not any(result["random"]["requires_grad_pre_head"].values()),
        "llm_no_grad": not any(result["llm"]["requires_grad_pre_head"].values()),
    }
    result["checks"] = checks
    result["all_passed"] = all(checks.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "checks": checks}, indent=2), flush=True)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
