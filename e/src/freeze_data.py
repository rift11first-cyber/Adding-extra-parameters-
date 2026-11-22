from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from littlekedi.tokenizer import BPETokenizer

DATASETS = {
    "sciq": {
        "repo": "allenai/sciq",
        "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
        "path": "data/test-00000-of-00001.parquet",
    },
    "arc_easy": {
        "repo": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "path": "ARC-Easy/test-00000-of-00001.parquet",
    },
    "hellaswag": {
        "repo": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "path": "data/validation-00000-of-00001.parquet",
    },
    "piqa": {
        "repo": "baber/piqa",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "path": "piqa_validation.parquet",
    },
    "arithmark3": {
        "repo": "AxiomicLabs/Arithmark-3.0",
        "revision": "6f6e59dd9b7e2c63455f7af7f838f9ecc3d0a746",
        "path": "arithmark-3.jsonl",
    },
    "wikitext2": {
        "repo": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "path": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    },
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, desc: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"cached {path.name} ({path.stat().st_size:,} bytes)", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total,
            desc=desc,
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as bar:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(path)


def source_url(name: str) -> str:
    info = DATASETS[name]
    return f"https://huggingface.co/datasets/{info['repo']}/resolve/{info['revision']}/{info['path']}"


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256_file(path)


def build_sciq(raw: Path) -> list[dict]:
    rows = pq.read_table(raw).to_pylist()
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": f"sciq_test_{i}",
            "prompt": f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".lstrip(),
            "choices": [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]],
            "answer": 3,
        })
    return out


def build_arc_easy(raw: Path) -> list[dict]:
    rows = pq.read_table(raw).to_pylist()
    out = []
    for i, r in enumerate(rows):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        out.append({
            "id": f"arc_easy_test_{i}",
            "prompt": f"Question: {r['question']}\nAnswer:",
            "choices": texts,
            "answer": labels.index(r["answerKey"]),
        })
    return out


def build_hellaswag(raw: Path, n: int, seed: int) -> tuple[list[dict], list[int]]:
    rows = pq.read_table(raw).to_pylist()
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    out = []
    for i in idx:
        r = rows[i]
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        endings = r["endings"] if isinstance(r["endings"], list) else ast.literal_eval(r["endings"])
        out.append({
            "id": f"hellaswag_val_{i}",
            "prompt": _hellaswag_clean(r["activity_label"] + ": " + ctx),
            "choices": [_hellaswag_clean(e) for e in endings],
            "answer": int(r["label"]),
        })
    return out, idx


def build_piqa(raw: Path, n: int, seed: int) -> tuple[list[dict], list[int]]:
    rows = pq.read_table(raw).to_pylist()
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    out = []
    for i in idx:
        r = rows[i]
        out.append({
            "id": f"piqa_val_{i}",
            "prompt": f"Question: {r['goal']}\nAnswer:",
            "choices": [r["sol1"], r["sol2"]],
            "answer": int(r["label"]),
        })
    return out, idx


def build_arithmark3(raw: Path, n: int, seed: int) -> tuple[list[dict], list[int]]:
    with open(raw, encoding="utf-8-sig") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    out = []
    for i in idx:
        r = rows[i]
        out.append({
            "id": f"arithmark3_{i}",
            "prompt": r["ctx"],
            "choices": r["endings"],
            "answer": int(r["label"]),
        })
    return out, idx


def gpt2_byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_byte_lengths(tok: BPETokenizer) -> np.ndarray:
    decoder = gpt2_byte_decoder()
    lengths = np.zeros(tok.vocab_size, dtype=np.uint16)
    for i in range(tok.vocab_size):
        s = tok.tok.id_to_token(i)
        if s is None or s.startswith("<|"):
            continue
        try:
            lengths[i] = len(bytes(decoder[ch] for ch in s))
        except KeyError:
            lengths[i] = 0
    return lengths


def build_wikitext2(raw: Path, tok: BPETokenizer, out_dir: Path) -> dict:
    rows = pq.read_table(raw).to_pylist()
    text = "\n".join(r["text"] for r in rows)
    (out_dir / "wikitext2_test_text.txt").write_text(text, encoding="utf-8")
    ids = np.array(tok.encode(text), dtype=np.uint16)
    np.save(out_dir / "wikitext2_test_ids.npy", ids)
    lengths = token_byte_lengths(tok)
    token_bytes = lengths[ids.astype(np.int64)]
    np.save(out_dir / "wikitext2_test_token_bytes.npy", token_bytes)
    return {
        "rows": len(rows),
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "tokens": int(len(ids)),
        "token_byte_lengths_max": int(lengths.max()),
        "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT.parent))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in DATASETS:
        download(source_url(name), raw_dir / f"{name}.bin", f"download {name}")

    manifest = {
        "datasets": {},
        "seed": args.seed,
        "files": {},
    }

    specs = [
        ("sciq", "sciq_test.jsonl", build_sciq),
        ("arc_easy", "arc_easy_test.jsonl", build_arc_easy),
    ]
    for name, filename, builder in specs:
        print(f"building {name}", flush=True)
        rows = builder(raw_dir / f"{name}.bin")
        path = out_dir / filename
        digest = write_jsonl(path, rows)
        manifest["files"][filename] = {"n": len(rows), "sha256": digest}
        manifest["datasets"][name] = DATASETS[name]

    for name, filename, builder, n in [
        ("hellaswag", "hellaswag_val_1000.jsonl", build_hellaswag, 1000),
        ("piqa", "piqa_val_500.jsonl", build_piqa, 500),
        ("arithmark3", "arithmark3_500.jsonl", build_arithmark3, 500),
    ]:
        print(f"building {name} (sample {n})", flush=True)
        rows, idx = builder(raw_dir / f"{name}.bin", n, args.seed)
        path = out_dir / filename
        digest = write_jsonl(path, rows)
        manifest["files"][filename] = {"n": len(rows), "sha256": digest, "indices": idx}
        manifest["datasets"][name] = DATASETS[name]

    tok = BPETokenizer.from_file(Path(args.model_dir) / "tokenizer.json")
    print("tokenizing wikitext2", flush=True)
    manifest["wikitext2"] = build_wikitext2(raw_dir / "wikitext2.bin", tok, out_dir)
    manifest["datasets"]["wikitext2"] = DATASETS["wikitext2"]
    manifest["files"]["wikitext2_test_ids.npy"] = {
        "n": manifest["wikitext2"]["tokens"],
        "sha256": sha256_file(out_dir / "wikitext2_test_ids.npy"),
    }
    manifest["files"]["wikitext2_test_text.txt"] = {
        "n": manifest["wikitext2"]["bytes"],
        "sha256": sha256_file(out_dir / "wikitext2_test_text.txt"),
    }

    for name in DATASETS:
        raw = raw_dir / f"{name}.bin"
        manifest["datasets"][name]["raw_sha256"] = sha256_file(raw)
        manifest["datasets"][name]["raw_bytes"] = raw.stat().st_size

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
