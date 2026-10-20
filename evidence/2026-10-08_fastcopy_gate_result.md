# FASTCOPY-GATE result: PASS (2026-10-08)

Chris's Claude on our workstation (2 × Radeon AI PRO R9700). Prereg `evidence/preregs/2026-10-08_partner_fastcopy_gate.md` (b3eff53, pushed before any output).
Question: may conv3d_fast `fp32_taps` in bf16 replace stock bf16 for the copy model? Reference = fast fp32.
Strip n81 (native s022814, rows 0-177, cols 81-102), 78 tool crops, A0 setting (`--no-tta`, overlap 0.5). One card (GPU 1).
Data: `fastcopy_summary.json` (a copy is in this folder), `g3_tta.json`, `copies/fc_<arm>.json` (our working files, not in this repository).

## Verdict: PASS (G1 and G2 pass; all copy checks passed, identical crop lists)

| Gate | stock bf16 | fp32_taps bf16 | ratio | bar | result |
|---|---|---|---|---|---|
| G1 pooled p99 of \|y - y_ref\| (78 crops) | 1.242 | 1.004 | 0.81 | <= 1.10 | pass |
| G1 pooled mean | 0.213 | 0.190 | 0.89 | <= 1.10 | pass |
| G2 D (step change >= 75 µm or real sign flip, 7,618 pairs) | 0 | 0 | | <= 1.10 x 0 + 0.002 | pass |

- G1 per crop: p99 ratio median 0.94 (range 0.44-1.13), mean ratio median 0.95 (0.54-1.15); 3.8 % of crops (3 of 78) are over 1.10.
  Pooled over many crops, `fp32_taps` sits a little below stock's bf16 noise, as the per-layer check predicted.
- **G2 is uninformative here:** neither bf16 copy changes any step decision against the fp32 copy (D = 0 for both). The PASS
  therefore rests on G1. A strip where bf16 noise moves decisions would be a sharper test.

## G3 record (not a gate)

| | stock bf16 | fp32_taps bf16 | fast fp32 (ref) |
|---|---|---|---|
| forward per crop (median) | 76.0 s | 0.86 s | 3.04 s |
| copy wall time, 78 crops | 6,528 s | 354 s | 496 s |
| peak reserved VRAM, copy | 29.7 GB | 20.3 GB | 31.1 GB |

- The fast fp32 copy peaks at 31.1 GB reserved: **over our workstation's new 30.5 GB limit** (set by Chris tonight).
- One 8-variant FIX TTA call with `fp32_taps` (first crop): 8.9 s, 22.1 GB allocated, **28.86 GB reserved**. That fits the new
  30.5 GB limit (the earlier 28 GB margin no longer applies).
- Copy wall times include S3 reads, so `fp32_taps` is 18x faster than stock per copy here (88x per forward).

## Scope

- Copy model only, crop 128 x 384 x 384, bf16 vs stock bf16, one strip of PHerc0841. No TTA at the metric level; TTA was tested
  for memory and time only. Other models need their own gates (v8-in has its own).

## Open item

- COPYFIX Stage 0 run 2 measured `acc32` at 1.41x / 1.36x on its sub-crop; a116c24 got 1.18x / 1.05x on the same sub-crop with the
  same code, while the full-crop outputs repeat to 5 digits. Cause not found. Not part of this gate.

GPU: one card, 20:19-22:23 (about 2.1 h). Claim released.
