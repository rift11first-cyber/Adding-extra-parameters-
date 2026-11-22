from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expanded_model import load_expanded, load_original
from progress import bar

TASK_FILES = {
    "sciq": "sciq_test.jsonl",
    "arc_easy": "arc_easy_test.jsonl",
    "hellaswag": "hellaswag_val_1000.jsonl",
    "piqa": "piqa_val_500.jsonl",
    "arithmark3": "arithmark3_500.jsonl",
}


def load_items(path: Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    return items[:limit] if limit else items


def score_task(model, tok, items: list[dict], max_len: int, batch_size: int, device: str, desc: str) -> dict:
    bos = [tok.eot_id]
    pad_id = tok.eot_id
    correct = correct_norm = correct_tok = 0
    n_done = 0
    t0 = time.time()
    pbar = bar(len(items), desc, unit="item", leave=False)
    chunk = 256

    for start in range(0, len(items), chunk):
        batch = items[start : start + chunk]
        reqs, owners = [], []
        for k, item in enumerate(batch):
            ctx = bos + tok.encode(item["prompt"])
            for choice in item["choices"]:
                cont = tok.encode(" " + str(choice).strip())[: max_len - 1]
                keep = max_len + 1 - len(cont)
                reqs.append((ctx[-keep:], cont))
                owners.append(k)

        order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i][0]) + len(reqs[i][1])))
        out = [0.0] * len(reqs)
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            seqs = [reqs[i][0] + reqs[i][1] for i in idx]
            T = max(len(s) for s in seqs) - 1
            x = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
            for j, s in enumerate(seqs):
                x[j, : len(s) - 1] = torch.tensor(s[:-1], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(x).logits
            logp = logits.float().log_softmax(-1)
            for j, i in enumerate(idx):
                ctx, cont = reqs[i]
                pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cont), device=device)
                out[i] = logp[j, pos, torch.tensor(cont, dtype=torch.long, device=device)].sum().item()

        per_item: dict[int, list[float]] = {}
        per_tok: dict[int, list[float]] = {}
        for k, lp, (_, cont) in zip(owners, out, reqs):
            per_item.setdefault(k, []).append(lp)
            per_tok.setdefault(k, []).append(lp / max(1, len(cont)))

        for k, item in enumerate(batch):
            lp = per_item[k]
            tok_lp = per_tok[k]
            norm = [l / max(1, len(" " + str(c).strip())) for l, c in zip(lp, item["choices"])]
            correct += max(range(len(lp)), key=lp.__getitem__) == item["answer"]
            correct_norm += max(range(len(norm)), key=norm.__getitem__) == item["answer"]
            correct_tok += max(range(len(tok_lp)), key=tok_lp.__getitem__) == item["answer"]

        n_done += len(batch)
        pbar.update(len(batch))
        pbar.set_postfix(acc=f"{correct / n_done:.1%}", acc_norm=f"{correct_norm / n_done:.1%}")

    pbar.close()
    n = len(items)
    acc_norm = correct_norm / n
    elapsed = time.time() - t0
    return {
        "acc": correct / n,
        "acc_norm": acc_norm,
        "acc_norm_tok": correct_tok / n,
        "n": n,
        "stderr": math.sqrt(max(acc_norm * (1 - acc_norm), 1e-9) / n),
        "elapsed_s": elapsed,
        "items_per_s": n / elapsed if elapsed > 0 else None,
    }


def evaluate_accuracy(model, tok, model_name: str, tasks: list[str], data_dir: Path, max_len: int, batch_size: int, device: str, limit: int | None) -> dict:
    results = {}
    for task in tasks:
        path = data_dir / TASK_FILES.get(task, f"{task}.jsonl")
        if not path.exists():
            print(f"skipping {task}: {path} not found", flush=True)
            continue
        items = load_items(path, limit)
        print(f"[{model_name}] {task}: {len(items):,} items", flush=True)
        results[task] = score_task(model, tok, items, max_len, batch_size, device, f"{model_name} {task}")
        r = results[task]
        print(
            f"[{model_name}] {task}: acc {r['acc']:.2%}  acc_norm {r['acc_norm']:.2%} "
            f"acc_norm_tok {r['acc_norm_tok']:.2%}  n={r['n']}  {r['elapsed_s']:.1f}s",
            flush=True,
        )
    return results


def evaluate_perplexity(model, model_name: str, data_dir: Path, device: str, seq_len: int, batch_size: int) -> dict:
    ids = np.load(data_dir / "wikitext2_test_ids.npy").astype(np.int64)
    token_bytes = np.load(data_dir / "wikitext2_test_token_bytes.npy").astype(np.int64)
    n_windows = len(ids) // seq_len
    windows = ids[: n_windows * seq_len].reshape(n_windows, seq_len)
    byte_windows = token_bytes[: n_windows * seq_len].reshape(n_windows, seq_len)
    nll_sum = 0.0
    token_sum = 0
    byte_sum = 0
    t0 = time.time()
    pbar = bar(n_windows, f"{model_name} wikitext2", unit="win", leave=False)
    with torch.inference_mode():
        for start in range(0, n_windows, batch_size):
            stop = min(start + batch_size, n_windows)
            x = torch.tensor(windows[start:stop, :-1], dtype=torch.long, device=device)
            y = torch.tensor(windows[start:stop, 1:], dtype=torch.long, device=device)
            logits = model(x).logits
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), reduction="sum")
            nll_sum += float(nll.item())
            token_sum += y.numel()
            byte_sum += int(byte_windows[start:stop, 1:].sum())
            pbar.update(stop - start)
            pbar.set_postfix(ppl=f"{math.exp(nll_sum / token_sum):.2f}")
    pbar.close()
    elapsed = time.time() - t0
    ppl = math.exp(nll_sum / token_sum)
    bpb = nll_sum / math.log(2) / byte_sum if byte_sum else None
    result = {
        "nll": nll_sum,
        "tokens": token_sum,
        "bytes": byte_sum,
        "ppl": ppl,
        "bpb": bpb,
        "windows": n_windows,
        "seq_len": seq_len,
        "elapsed_s": elapsed,
        "tokens_per_s": token_sum / elapsed if elapsed > 0 else None,
    }
    print(
        f"[{model_name}] wikitext2: ppl {ppl:.3f}  bpb {bpb:.4f}  tokens {token_sum:,}  {elapsed:.1f}s",
        flush=True,
    )
    return result


def _stats(times_ms: list[float]) -> dict:
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def evaluate_latency(model, model_name: str, device: str, threads: int, prefill_lengths: list[int], decode_steps: int, reps: int) -> dict:
    torch.set_num_threads(threads)
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    results = {"threads": threads, "prefill": {}, "decode": {}}

    for T in prefill_lengths:
        x = torch.randint(0, model.cfg.vocab_size, (1, T), generator=g, dtype=torch.long).to(device)
        with torch.inference_mode():
            for _ in range(3):
                model(x)
            times = []
            pbar = bar(reps, f"{model_name} prefill {T}", unit="run", leave=False)
            for _ in range(reps):
                t0 = time.perf_counter()
                model(x)
                times.append(1000 * (time.perf_counter() - t0))
                pbar.update(1)
            pbar.close()
        stats = _stats(times)
        stats["tokens_per_s"] = T / (stats["median_ms"] / 1000)
        results["prefill"][str(T)] = stats
        print(f"[{model_name}] prefill T={T}: {stats['median_ms']:.2f} ms  {stats['tokens_per_s']:.1f} tok/s", flush=True)

    prefill_T = 128
    x = torch.randint(0, model.cfg.vocab_size, (1, prefill_T), generator=g, dtype=torch.long).to(device)
    caches = [{} for _ in model.layers]
    with torch.inference_mode():
        logits = model.forward_cached(x, caches, 0)
        next_id = logits.argmax(-1, keepdim=True)
        times = []
        pbar = bar(decode_steps, f"{model_name} decode", unit="tok", leave=False)
        for step in range(decode_steps):
            t0 = time.perf_counter()
            logits = model.forward_cached(next_id, caches, prefill_T + step)
            times.append(1000 * (time.perf_counter() - t0))
            next_id = logits.argmax(-1, keepdim=True)
            pbar.update(1)
        pbar.close()
    stats = _stats(times)
    stats["tokens_per_s"] = 1000 / stats["median_ms"] if stats["median_ms"] > 0 else None
    results["decode"] = stats
    print(f"[{model_name}] decode: {stats['median_ms']:.2f} ms/token  {stats['tokens_per_s']:.1f} tok/s", flush=True)
    return results


def run_memory_worker(kind: str, path: str, out_path: Path, device: str, threads: int) -> dict:
    print(f"[{kind}] measuring memory", flush=True)
    cmd = [
        sys.executable,
        str(ROOT / "src" / "memory_worker.py"),
        "--kind", kind,
        "--path", path,
        "--out", str(out_path),
        "--device", device,
        "--threads", str(threads),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def write_summary(out_dir: Path, model_names: list[str], params: dict, accuracy: dict, ppl: dict, latency: dict, memory: dict) -> dict:
    summary = {}
    for name in model_names:
        summary[name] = {
            "params": params.get(name, {}),
            "accuracy": accuracy.get(name, {}),
            "perplexity": ppl.get(name, {}),
            "latency": latency.get(name, {}),
            "memory": memory.get(name, {}),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tasks = ["sciq", "arc_easy", "hellaswag", "piqa", "arithmark3"]
    header = "| model | total params | new params | " + " | ".join(f"{t} acc_norm" for t in tasks) + " | wikitext2 ppl | prefill 128 ms | decode ms/tok | peak RSS MiB |"
    sep = "|---|---:|---:|" + "---:|" * len(tasks) + "---:|---:|---:|---:|"
    lines = [header, sep]
    for name in model_names:
        s = summary[name]
        p = s["params"]
        acc = s["accuracy"]
        pplv = s["perplexity"].get("ppl")
        lat = s["latency"]
        mem = s["memory"]
        prefill = lat.get("prefill", {}).get("128", {}).get("median_ms")
        decode = lat.get("decode", {}).get("median_ms")
        peak = mem.get("peak_rss_mib")
        cells = [
            name,
            f"{p.get('total', 0):,}",
            f"{p.get('pre_head', 0):,}",
        ]
        for task in tasks:
            v = acc.get(task, {}).get("acc_norm")
            cells.append(f"{v:.4f}" if v is not None else "n/a")
        cells.extend([
            f"{pplv:.3f}" if pplv is not None else "n/a",
            f"{prefill:.2f}" if prefill is not None else "n/a",
            f"{decode:.2f}" if decode is not None else "n/a",
            f"{peak:.1f}" if peak is not None else "n/a",
        ])
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT.parent))
    ap.add_argument("--random-dir", default=str(ROOT / "checkpoints" / "random_seed0"))
    ap.add_argument("--llm-dir", default=str(ROOT / "checkpoints" / "llm_generated"))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--models", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--tasks", nargs="+", default=["sciq", "arc_easy", "hellaswag", "piqa", "arithmark3"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ppl-batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--latency-reps", type=int, default=20)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-memory", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    paths = {
        "A": str(Path(args.model_dir)),
        "B": str(Path(args.random_dir)),
        "C": str(Path(args.llm_dir)),
    }

    stages = []
    for name in args.models:
        if not args.skip_accuracy:
            stages.append(f"{name}:accuracy")
        if not args.skip_ppl:
            stages.append(f"{name}:ppl")
        if not args.skip_latency:
            stages.append(f"{name}:latency")
        if not args.skip_memory:
            stages.append(f"{name}:memory")
    overall = bar(len(stages), "overall", unit="stage", leave=True)

    params: dict = {}
    accuracy: dict = {}
    ppl: dict = {}
    latency: dict = {}
    memory: dict = {}

    for name in args.models:
        print(f"\n=== model {name}: {paths[name]}", flush=True)
        if name == "A":
            model, cfg, _, tok = load_original(paths[name], device=args.device)
        else:
            model, cfg, _, tok = load_expanded(paths[name], device=args.device)
        params[name] = model.param_counts()
        print(json.dumps(params[name], indent=2), flush=True)

        if not args.skip_accuracy:
            accuracy[name] = evaluate_accuracy(
                model, tok, name, args.tasks, data_dir, model.cfg.max_seq_len, args.batch_size, args.device, args.limit
            )
            (out_dir / f"accuracy_{name}.json").write_text(json.dumps(accuracy[name], indent=2), encoding="utf-8")
            overall.update(1)

        if not args.skip_ppl:
            ppl[name] = evaluate_perplexity(model, name, data_dir, args.device, args.seq_len, args.ppl_batch_size)
            (out_dir / f"perplexity_{name}.json").write_text(json.dumps(ppl[name], indent=2), encoding="utf-8")
            overall.update(1)

        if not args.skip_latency:
            latency[name] = evaluate_latency(
                model, name, args.device, args.threads, [128, 512, 1024], args.decode_steps, args.latency_reps
            )
            (out_dir / f"latency_{name}.json").write_text(json.dumps(latency[name], indent=2), encoding="utf-8")
            overall.update(1)

        if not args.skip_memory:
            memory[name] = run_memory_worker(name, paths[name], out_dir / f"memory_{name}.json", args.device, args.threads)
            overall.update(1)

        del model
        gc.collect()

    overall.close()
    summary = write_summary(out_dir, args.models, params, accuracy, ppl, latency, memory)
    print(f"\nwrote {out_dir / 'summary.json'} and {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
