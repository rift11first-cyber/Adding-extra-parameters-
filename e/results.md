# LittleKedi Expansion Results

Generated: 2026-11-22 02:51 UTC  
Status: complete  
Scope: inference-only evaluation of the original model and two expanded variants. No fine-tuning, gradients, or optimizer were used.

## Summary

- **A (original):** 1,990,912 parameters.
- **B (random control):** 2,990,770 parameters, including 999,858 new parameters.
- **C (LLM-designed):** 2,990,770 parameters, including 999,858 new parameters.
- Both expanded models keep the original base weights exactly and add a residual MLP after the final RMSNorm, before the tied LM head.
- **C preserves the original model better than B:** WikiText-2 perplexity is 73.33 for C vs 76.14 for B and 71.96 for A. C also has lower peak RSS than B in this run.
- **Neither expansion improves accuracy.** All A/B/C accuracy differences are small and mostly within one standard error.
- **The extra layer costs latency and memory** without a measurable downstream accuracy gain on this benchmark suite.
- C vs B is mixed on `acc_norm`: C is better on PIQA, while B is better on SciQ, ARC-Easy, HellaSwag, and ArithMark. There is no clear accuracy winner between B and C.

## Setup

- Hardware: CPU-only Intel i5-1135G7, 8 logical CPUs, 7.5 GB RAM.
- Runtime: PyTorch 2.14.1+cpu, 4 CPU threads.
- Base checkpoint SHA256: `e97c3a4c4cf0a1b00119cb350297fa5ae6e81fdead6cf56381a43a6ac9f7e9ce`.
- Upstream code commit: `a282a8fdc945d783e05b25b8dcd42fc493cf8f23`.
- Insertion: after final RMSNorm, before the tied LM head:
  `hn = norm(h); y = hn + pre_head(hn); logits = F.linear(y, embed.weight)`.
- New layer: `128 -> 3890 -> 128`, GELU activation, zero biases.
- New parameters: `w1` 497,920 + `b1` 3,890 + `w2` 497,920 + `b2` 128 = **999,858**.
- B initialization: independent iid Gaussian, `std=0.02`, seed 0.
- C initialization: external-LLM-authored deterministic generator; QR orthonormal frame `Q`, `w1=Q`, `w2=0.15*Q^T`, zero biases.
- All parameters have `requires_grad=False`; evaluation uses `torch.inference_mode()`.

## Parameter Counts

| Model | Total | Active | Active non-embedding | New params | Active fraction |
|---|---:|---:|---:|---:|---:|
| A | 1,990,912 | 1,548,544 | 499,968 | 0 | 77.8% |
| B | 2,990,770 | 2,548,402 | 1,499,826 | 999,858 | 85.2% |
| C | 2,990,770 | 2,548,402 | 1,499,826 | 999,858 | 85.2% |

## Accuracy

Frozen evaluation sets. SciQ and ARC-Easy use the full test sets; HellaSwag, PIQA, and ArithMark-3 use fixed seeded subsets.

### Raw accuracy

| Task | n | A | B | C |
|---|---:|---:|---:|---:|
| SciQ | 1,000 | 0.5980 | 0.5860 | 0.5930 |
| ARC-Easy | 2,376 | 0.3232 | 0.3136 | 0.3215 |
| HellaSwag | 1,000 | 0.2570 | 0.2570 | 0.2620 |
| PIQA | 500 | 0.5520 | 0.5540 | 0.5500 |
| ArithMark-3 | 500 | 0.3060 | 0.3160 | 0.3040 |

### Length-normalized accuracy (`acc_norm`)

| Task | n | A | B | C | A stderr |
|---|---:|---:|---:|---:|---:|
| SciQ | 1,000 | **0.5630** | 0.5600 | 0.5530 | 0.0157 |
| ARC-Easy | 2,376 | **0.3274** | 0.3232 | 0.3224 | 0.0096 |
| HellaSwag | 1,000 | 0.2730 | **0.2750** | 0.2680 | 0.0141 |
| PIQA | 500 | **0.5440** | 0.5280 | 0.5420 | 0.0223 |
| ArithMark-3 | 500 | 0.3040 | **0.3160** | 0.3020 | 0.0206 |

### Token-normalized accuracy (`acc_norm_tok`)

| Task | A | B | C |
|---|---:|---:|---:|
| SciQ | 0.5260 | 0.5350 | 0.5240 |
| ARC-Easy | 0.2681 | 0.2706 | 0.2681 |
| HellaSwag | 0.2590 | 0.2640 | 0.2640 |
| PIQA | 0.5520 | 0.5580 | 0.5580 |
| ArithMark-3 | 0.2700 | 0.2780 | 0.2680 |

### Accuracy deltas vs A

| Task | B - A | C - A | C - B |
|---|---:|---:|---:|
| SciQ | -0.30 pp | -1.00 pp | -0.70 pp |
| ARC-Easy | -0.42 pp | -0.51 pp | -0.08 pp |
| HellaSwag | +0.20 pp | -0.50 pp | -0.70 pp |
| PIQA | -1.60 pp | -0.20 pp | +1.40 pp |
| ArithMark-3 | +1.20 pp | -0.20 pp | -1.40 pp |

Most differences are within roughly one standard error. The largest single difference, B's PIQA result, is 1.6 pp with a standard error of about 2.2 pp.

## Perplexity

WikiText-2 raw test, non-overlapping 1024-token windows. There are 349 windows and 357,027 scored tokens.

| Model | NLL | PPL | BPB | PPL vs A | Tokens/s |
|---|---:|---:|---:|---:|---:|
| A | 1,526,673.56 | **71.957** | 1.7092 | baseline | 11,724.5 |
| B | 1,546,847.40 | 76.140 | 1.7318 | +5.81% | 9,389.1 |
| C | 1,533,407.39 | 73.327 | 1.7167 | +1.90% | 9,688.3 |

C is substantially closer to A than B on perplexity. B's degradation is about three times larger than C's in relative terms.

## Latency

Median wall-clock time on CPU, batch size 1, 4 threads, 20 repetitions after 3 warmups. Decode is 64 cached steps after a 128-token prefill.

| Model | Prefill 128 ms | tok/s | Prefill 512 ms | tok/s | Prefill 1024 ms | tok/s | Decode ms/tok | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | **21.66** | 5,908.9 | **43.00** | 11,908.1 | **71.11** | 14,400.6 | **12.72** | 78.6 |
| B | 25.04 | 5,111.1 | 56.75 | 9,022.2 | 94.34 | 10,853.9 | 15.84 | 63.1 |
| C | 24.72 | 5,177.2 | 53.16 | 9,631.8 | 103.30 | 9,913.1 | 15.73 | 63.6 |

Relative to A:

- B: prefill 128 +15.6%, prefill 512 +32.0%, prefill 1024 +32.7%, decode +24.5%.
- C: prefill 128 +14.1%, prefill 512 +23.6%, prefill 1024 +45.3%, decode +23.7%.
- C is not faster than B overall; C is faster at 512-token prefill and roughly tied on decode, but slower at 1024-token prefill.

## Memory

Separate process per model. Peak RSS is process-level; `delta RSS` is the inference working set above the post-load baseline. Forward timing is for 10 iterations at 512 tokens.

| Model | Param MiB | State MiB | Baseline RSS MiB | Peak RSS MiB | Delta RSS MiB | Forward 512 ms |
|---|---:|---:|---:|---:|---:|---:|
| A | 7.59 | 7.59 | 261.26 | **275.43** | 14.17 | **47.97** |
| B | 11.41 | 11.41 | 273.93 | 305.38 | 31.44 | 60.52 |
| C | 11.41 | 11.41 | 273.71 | 287.90 | 14.18 | 56.93 |

The expanded checkpoints add about 3.81 MiB of parameter storage. C's peak RSS is 17.5 MiB lower than B's in this run, despite identical parameter shapes. Because B and C have identical activation shapes, this difference is likely allocator/measurement behavior rather than a structural memory advantage; treat it as a single-run observation.

## Construction Verification

- **70/70 construction checks passed.**
- Original model hash is unchanged.
- B and C have exactly 2,990,770 parameters and exactly 999,858 new parameters.
- All original base tensors are bit-identical in B and C.
- Extra state keys are exactly `pre_head.w1`, `pre_head.b1`, `pre_head.w2`, `pre_head.b2`.
- Zeroing `w2` and `b2` restores the original logits to within `1e-6` (`torch.allclose`), confirming the residual insertion.
- The new layer is called once in both normal and cached forward paths.
- No parameter requires gradients.
- C reproduces exactly from the frozen generator.

| Item | B | C |
|---|---|---|
| Role | random control | LLM-designed |
| Initialization | iid Gaussian, std 0.02 | QR frame, `w1=Q`, `w2=0.15*Q^T` |
| Biases | zero | zero |
| State SHA256 | `5929edb42544999a898b6bf5eb395cb9336f68fea7ae5b2e3939d808d0aedb51` | `80ea1c0680de14fc018589005216e24f261dab631dbe653a2fc1385bafcd613a` |
| Model file SHA256 | `0cc2d0edead1c31825d7727f4607a54dd8fc31aafee1d57c4cd4936b6af53136` | `5032e44ac9b4d3837fb09aecc50190c67033f15c52cbdf0a9f13f9676642dda6` |
| Generator SHA256 | n/a | `aa2f3ba071871d558427dd83c30651283a16f090d9ce26b666e78efea06fc14f` |
| Prompt record SHA256 | n/a | `fe9441783e629bff572bb8e3d0a807e78a02ddfe002f21004dcea82d6a9388a9` |

## Interpretation

1. **C is the safer expansion of the two.** It loses only 1.9% perplexity relative to A, compared with 5.8% for B. It also stays closer to A's peak memory in this run.
2. **The LLM-designed initialization does not create a capability gain.** C does not beat A on `acc_norm` for any of the five tasks. B beats A only slightly on HellaSwag and ArithMark (`acc_norm`), and loses more on PIQA.
3. **B vs C is not a clean win for either.** C is better on PIQA; B is better on SciQ, ARC-Easy, HellaSwag, and ArithMark (`acc_norm`). The differences are small relative to the measured standard errors.
4. **The cost is real.** The expanded models add about 1M active parameters, 14-45% prefill latency, about 24% decode latency, and 1.9-5.8% perplexity. On this suite, that cost is not offset by an accuracy improvement.
5. **The main positive result is constructional:** a deterministic, externally authored, non-random initialization can add capacity while staying much closer to the original model than a random control.

## Caveats

- CPU-only, single run, 4 threads; latency and RSS measurements are noisy.
- HellaSwag, PIQA, and ArithMark-3 use fixed subsets, not full validation sets.
- Accuracy differences are mostly within one standard error; do not overinterpret small deltas.
- Peak RSS is process-level and allocator-dependent; the B/C memory difference should be re-measured if it matters.
- The tokenizer is the released byte-level BPE tokenizer as-is. It does not split digits, despite upstream code/card hints.
- No training or adaptation was performed. This evaluates initialization-only expansion.

## Reproducibility

Checkpoints:

- `/home/nikki/Desktop/model/expansion/checkpoints/random_seed0/`
- `/home/nikki/Desktop/model/expansion/checkpoints/llm_generated/`

Result artifacts:

- `/home/nikki/Desktop/model/expansion/results/summary.json`
- `/home/nikki/Desktop/model/expansion/results/summary.md`
- `/home/nikki/Desktop/model/expansion/results/accuracy_A.json`
- `/home/nikki/Desktop/model/expansion/results/accuracy_B.json`
- `/home/nikki/Desktop/model/expansion/results/accuracy_C.json`
- `/home/nikki/Desktop/model/expansion/results/perplexity_A.json`
- `/home/nikki/Desktop/model/expansion/results/perplexity_B.json`
- `/home/nikki/Desktop/model/expansion/results/perplexity_C.json`
- `/home/nikki/Desktop/model/expansion/results/latency_A.json`
- `/home/nikki/Desktop/model/expansion/results/latency_B.json`
- `/home/nikki/Desktop/model/expansion/results/latency_C.json`
- `/home/nikki/Desktop/model/expansion/results/memory_A.json`
- `/home/nikki/Desktop/model/expansion/results/memory_B.json`
- `/home/nikki/Desktop/model/expansion/results/memory_C.json`
- `/home/nikki/Desktop/model/expansion/results/construction_summary.json`
- `/home/nikki/Desktop/model/expansion/results/verification.json`
- `/home/nikki/Desktop/model/expansion/results/logs/full_eval.log`

Main commands:

```bash
python3 src/freeze_data.py --model-dir /home/nikki/Desktop/model --out-dir data --seed 0
python3 src/make_checkpoints.py --mode random --out checkpoints/random_seed0 --seed 0
python3 src/make_checkpoints.py --mode llm --generator src/llm_generator.py --out checkpoints/llm_generated --seed 0
python3 src/verify.py --random-dir checkpoints/random_seed0 --llm-dir checkpoints/llm_generated --out results/verification.json
python3 src/finalize_construction.py
python3 src/evaluate.py --models A B C --batch-size 16 --out-dir results
```

## Bottom Line

The expansion is mechanically correct and fully reproducible. C is a better-behaved expansion than B, but neither variant improves accuracy. The LLM-designed layer mainly acts as a low-impact residual: it preserves perplexity better than random initialization, while still adding the full inference cost of the extra parameters.
