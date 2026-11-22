from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.tokenizer import BPETokenizer

NEW_LAYER_KEYS = ("pre_head.w1", "pre_head.b1", "pre_head.w2", "pre_head.b2")


class PreHeadMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(hidden, dim))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(dim, hidden))
        self.b2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.gelu(F.linear(x, self.w1, self.b1)), self.w2, self.b2)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ExpandedLittleKedi(LittleKediModel):
    def __init__(self, cfg: ModelConfig, hidden: int = 3890):
        super().__init__(cfg)
        self.pre_head = PreHeadMLP(cfg.dim, hidden)
        self.pre_head_hidden = hidden
        for p in self.parameters():
            p.requires_grad_(False)

    def output(self, h: torch.Tensor) -> torch.Tensor:
        hn = self.norm(h)
        y = hn + self.pre_head(hn)
        return F.linear(y, self.embed.weight) if self.head is None else self.head(y)

    def param_counts(self) -> dict[str, int | float]:
        counts = super().param_counts()
        counts["pre_head"] = self.pre_head.param_count()
        counts["base_total"] = int(counts["total"]) - counts["pre_head"]
        return counts


def load_base_files(model_dir: str | Path) -> tuple[dict, dict[str, torch.Tensor], str]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    state = load_file(str(model_dir / "model.safetensors"))
    tok_json = (model_dir / "tokenizer.json").read_text(encoding="utf-8")
    return cfg, state, tok_json


def load_original(model_dir: str | Path, device: str = "cpu"):
    cfg_dict, state, tok_json = load_base_files(model_dir)
    model = LittleKediModel(ModelConfig.from_dict(cfg_dict))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = BPETokenizer.from_str(tok_json)
    return model, cfg_dict, tok_json, tok


def build_expanded(model_dir: str | Path, hidden: int = 3890, device: str = "cpu"):
    cfg_dict, state, tok_json = load_base_files(model_dir)
    model = ExpandedLittleKedi(ModelConfig.from_dict(cfg_dict), hidden=hidden)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("pre_head.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"base load mismatch: missing={bad_missing} unexpected={unexpected}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = BPETokenizer.from_str(tok_json)
    return model, cfg_dict, tok_json, tok


def load_expanded(ckpt_dir: str | Path, device: str = "cpu"):
    ckpt_dir = Path(ckpt_dir)
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    state = load_file(str(ckpt_dir / "model.safetensors"))
    hidden = int(state["pre_head.w1"].shape[0])
    model = ExpandedLittleKedi(ModelConfig.from_dict(cfg_dict), hidden=hidden)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok_json = (ckpt_dir / "tokenizer.json").read_text(encoding="utf-8")
    tok = BPETokenizer.from_str(tok_json)
    return model, cfg_dict, tok_json, tok


def save_expanded(
    ckpt_dir: str | Path,
    model: ExpandedLittleKedi,
    cfg_dict: dict,
    tok_json: str,
    manifest: dict,
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    metadata = {str(k): str(v) for k, v in manifest.items()}
    save_file(state, str(ckpt_dir / "model.safetensors"), metadata=metadata)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    (ckpt_dir / "tokenizer.json").write_text(tok_json, encoding="utf-8")
    (ckpt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ckpt_dir


def init_random_pre_head(model: ExpandedLittleKedi, std: float, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        model.pre_head.w1.normal_(0.0, std, generator=g)
        model.pre_head.b1.zero_()
        model.pre_head.w2.normal_(0.0, std, generator=g)
        model.pre_head.b2.zero_()


def set_pre_head_weights(model: ExpandedLittleKedi, weights: dict[str, torch.Tensor]) -> None:
    expected = {
        "pre_head.w1": tuple(model.pre_head.w1.shape),
        "pre_head.b1": tuple(model.pre_head.b1.shape),
        "pre_head.w2": tuple(model.pre_head.w2.shape),
        "pre_head.b2": tuple(model.pre_head.b2.shape),
    }
    for key, shape in expected.items():
        if key not in weights:
            raise KeyError(f"generator did not return {key}")
        tensor = weights[key].detach().to(dtype=torch.float32, device="cpu")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{key}: expected {shape}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key}: non-finite values")
        with torch.no_grad():
            param = dict(model.named_parameters())[key]
            param.copy_(tensor)
