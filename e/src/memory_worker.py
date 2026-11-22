from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expanded_model import load_expanded, load_original


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["A", "B", "C"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.kind == "A":
        model, cfg, _, tok = load_original(args.path, device=args.device)
    else:
        model, cfg, _, tok = load_expanded(args.path, device=args.device)
    model.eval()

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    x = torch.randint(0, model.cfg.vocab_size, (1, args.seq_len), dtype=torch.long, device=args.device)

    with torch.inference_mode():
        model(x)
    gc.collect()

    baseline = rss_mib()
    peak = baseline
    stop = threading.Event()

    def sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, rss_mib())
            time.sleep(0.005)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.iters):
            model(x)
    elapsed = time.perf_counter() - t0
    stop.set()
    thread.join(timeout=1.0)
    peak = max(peak, rss_mib())

    result = {
        "kind": args.kind,
        "path": str(Path(args.path).resolve()),
        "device": args.device,
        "threads": args.threads,
        "seq_len": args.seq_len,
        "iters": args.iters,
        "param_bytes": param_bytes,
        "param_mib": param_bytes / (1024 * 1024),
        "state_bytes": state_bytes,
        "state_mib": state_bytes / (1024 * 1024),
        "baseline_rss_mib": baseline,
        "peak_rss_mib": peak,
        "delta_rss_mib": peak - baseline,
        "forward_ms": 1000 * elapsed / args.iters,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
