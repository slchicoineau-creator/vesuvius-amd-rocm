# Pre-registration: FASTCOPY-GATE: is conv3d_fast `fp32_taps` in bf16 equivalent to stock bf16 for the copy model? (partner, [internal reference removed])

- **Date:** 2026-10-08 (evening CDT). Pushed with the code, before any gate output exists (the commit time is the proof).
  **Owner:** partner (lane B machine). **Decision owner** for team use: Sebastien.
- **Why:** copyfix Stage 0 failed the fast path in bf16 on one sub-crop (2.1x stock's rounding noise). The instrument check
  `[internal reference removed]` (a116c24) showed that a one-crop ratio is mostly luck: on 8 sub-crops of
  one crop, every mode spans about 0.3-1.9x, including a mode whose per-layer noise equals stock's. It also added the
  opt-in mode `fp32_taps` (per-layer noise at or below stock in all 71 layers; 0.87 s per full-crop forward vs 75.6 s stock).
  This gate asks, with many crops and at the copy's own decision metric, whether `fp32_taps` bf16 may replace stock bf16.
- **This is not COPYFIX.** COPYFIX files stay unchanged and no COPYFIX arm runs. cf.py, ct9.py and tf9.py are imported
  read-only (cf._d style metric, ct9 step / normals / volume / checkpoint, tf9 FIX TTA wrapper).
- **Truth rule:** no w00, no humans, no labels. Every number compares a copy or an output with the fp32 reference on the same input.

## Data

- Strip **n81**: the native public surface s022814 (`[internal reference removed]`, 188 x 245 grid),
  rows 0-177 (S0c's rows), cols 81-102. S0c is cols 103-236 of the same grid, and its 22-column strips are all used
  (COPYFIX c00-c110, TTASIGN 66-87; only 2 columns remain), so the fresh strip is the 22 columns just left of S0c.
  3,809 vertices, z 13,089-16,122. No copy has used these columns. Normals are computed on the full native grid (ct9.normals).
- The tool's own crop list at `--bbox-overlap 0.5`: **78 crops** (`[internal reference removed]`, CPU, made before this push).
- Scan PHerc0841 9.366 µm level 0, checkpoint `copy_displacement_latest.pth`, crop 128 x 384 x 384, bf16 (the
  checkpoint's own amp) except the reference.

## Arms (script `[internal reference removed]`, run by `job.sh`; one card, GPU 1)

All three are the tool's A0 copy as in COPYFIX: `--no-tta`, `--bbox-overlap 0.5`, batch 1, ct9's argv on strip n81.
- **REF:** conv3d_fast default in fp32 (amp off), `ROCBLAS_USE_HIPBLASLT=0`. Exact: p99 0.0002 vs stock fp32 on a full real crop.
- **F32T:** conv3d_fast `mode="fp32_taps"`, bf16, `ROCBLAS_USE_HIPBLASLT=1`.
- **STOCK:** stock conv3d, bf16, `ROCBLAS_USE_HIPBLASLT=1`. At every crop the same input also goes through F32T and REF in
  the same process (for G1); the tool keeps the stock output. (Fast fp32 under hipBLASLt 1 matched stock fp32 to 4
  decimals on the Stage 0 sub-crop, a116c24.)
- Per-copy checks (else exit 3, verdict VOID): output grid = strip grid; valid count 90-102 % of the input; `bbox_count`
  = 78 = the plan; the same crop z starts as the plan; overlap 0.5; crop 128 x 384 x 384; no TTA call; the run's amp as
  the arm says; the conv path asserted at every crop. All three copies must have identical crop lists.

## Gates

- **G1, output level, pooled.** For every crop: |y - y_REF| over all 6 x 128 x 384 x 384 output values, for stock bf16
  and for F32T bf16. Pooled over the 78 crops: p99 (from a 0.5 %-wide log histogram, identical bins for both) and mean.
  **Pass:** pooled p99 ratio F32T/stock <= 1.10 AND pooled mean ratio <= 1.10. Reported: per-crop ratios (median,
  range, share of crops over 1.10).
- **G2, decision metric.** Step s = (K - S) . n x 9.366 µm per vertex and side (ct9.step). For copy X (STOCK, F32T),
  over (vertex, side) pairs valid in X and REF: a pair **disagrees** if |s_X - s_REF| >= 75 µm (ct9's sheet-scale
  break), or if sign(s_X) != sign(s_REF) where |s_REF| >= 37.5 µm (a flip of a real step; flips of near-zero steps are
  rounding, not a decision). D(X) = disagreeing share.
  **Pass:** D(F32T) <= 1.10 x D(STOCK) + 0.002.
  The allowance: D may be near 0 here. With ~7,600 pairs (3,809 vertices x 2 sides), 0.002 is about 15 pairs; below
  that a ratio would turn a handful of pairs into a verdict. It is fixed now and not changed.
- **Verdict:** **PASS** = G1 and G2 pass. **FAIL** = either fails. **VOID** = any copy check fails or the crop lists differ.
- **G3 (record, not a gate):** forward time per mode (median over crops in the STOCK run); each copy's wall time and peak
  reserved VRAM; one 8-variant FIX TTA call (copyfix's settings and `tf9.make_rmt(..., "FIX")`) on the strip's first
  crop with F32T: time and peak reserved. Stated plainly: fits the 28 GB headroom or not. If not, the options are listed
  (not tried): allocator settings (e.g. a smaller split size; expandable segments are suspect on this box), freeing
  cached blocks between TTA variants, or a TTA batch that holds fewer variant outputs at once.

## What each outcome changes

- **PASS:** `fp32_taps` bf16 may be used for copy-model runs in place of stock bf16 (record `conv_path =
  conv3d_fast+fp32_taps`). TTA runs still need the G3 memory answer. Sebastien decides any team use.
- **FAIL / VOID:** no change: copy runs on [internal reference removed] stay stock bf16 or fast fp32.
- Not claimed: anything about other models (v8-in has its own gates), other scrolls, or which copy is on the right sheet.

## Open item (recorded, not part of this gate)

- COPYFIX Stage 0 run 2 measured `acc32` at 1.41x / 1.36x on its sub-crop; a116c24 measured 1.18x / 1.05x on the same
  sub-crop with the same code, while the full-crop outputs repeat to 5 digits and run 1 (`taps`) repeats exactly. Cause not
  found (a per-process difference such as MIOpen's kernel choice is one guess). Not investigated here.

## Compute

- REF and F32T copies: minutes each. STOCK: 78 crops x ~76 s + G1 extras ≈ 1.7-1.9 h on one card. TTA record: ~1 min.
  One `[internal reference removed] claim fastcopy` (3 h), released by the job's last line. `[internal reference removed] status` is checked before launch;
  the job stops if another claim or a job of Sebastien's appears before it starts.
- Outputs in `[internal reference removed]`: `plan.json`, `copies/fc_<arm>.json`, `g1_hist.npz`, `g3_tta.json`,
  `fastcopy_summary.json`. Copy grids stay local (git-ignored).
- **Kill date:** 2026-10-12. Not scored by then: closed as not run.
