# Pre-registration: HECATE-FAST-GATE: is conv3d_fast `fp32_taps` in bf16 equivalent to stock bf16 for Hecate (2.4 µm and 9.6 µm)? (partner, [internal reference removed])

- **Date:** 2026-10-09 (morning CDT). Pushed with the code, before any gate output exists (the commit time is the proof).
  **Owner:** partner (lane B machine). **Decision owner** for team use: Sebastien. Kind: tooling validation, no team hypothesis.
- **Why:** CAPACITY.md row "Hecate 2.4 / 9.6 (bf16)" says "NO (bf16, never gated)". The copy model passed the same kind of
  gate (`[internal reference removed]`, PASS, [internal reference removed]). Hecate is a ResNet-3D (Bottleneck [3,8,36,3]) with a 3D U-Net
  decoder. **Count (CPU, measured before this push):** 56 3x3x3 Conv3d layers with more than one input channel that the fast lane
  rewrites (50 bottleneck `conv2` + 6 decoder blocks) in each checkpoint; 112 1x1x1 convs and the stem (7x7x7 at 2.4 µm, 3x3x3 with
  one input channel at 9.6 µm) stay stock. **Speed probe (measured, one real central patch, bf16 autocast, batch 1, GPU 1, after warm-up):**
  2.4 µm 0.16 s stock vs 0.08 s fp32_taps (2.0x); 9.6 µm 0.040 s vs 0.025 s (1.6x). Both above the 1.3x stop rule, so the gate is run.
- **Truth rule:** no labels, no humans, no ratings. Every number compares an arm with a reference computed on the same input.

## Data (published scroll data already cached; nothing private, no rating pages, no keys)

- The PC-0841 surface volumes rendered earlier for the repro (`[internal reference removed]`): `[internal reference removed]`
  (109 x 2784 x 6496, 2.4 µm, 1.044 cm²) and `R_sv96.npy` (27 x 697 x 1626, 9.6 µm). Checkpoints `hecate_2.4um.pth`, `hecate_9.6um.pth`
  (`[internal reference removed]`). The ink map `team_ink.tif` is not read.
- **Patches (G1):** the model's own patch grid at its default stride (patch/2: 256 px at 2.4 µm with 64 layers, 64 px at 9.6 µm with
  16 layers), central depth window as `hecate.predict` uses it, non-reversed. Eligible = at least 90 % of the patch columns have a
  nonzero voxel in the window (1,050 eligible at each scale). **40 patches at 2.4 µm, 64 at 9.6 µm**, chosen by index
  `linspace(0, n_eligible-1, N).round()` over the row-major eligible list (`[internal reference removed]`, made before this push).
  No selection on any output.

## Arms (script `[internal reference removed]`, run by `job.sh`; one card, GPU 1, `ROCBLAS_USE_HIPBLASLT=1` for all, no expandable_segments)

Model input exactly as `hecate.predict` builds it (uint8 / divisor, `valid` mask), output = `model(image, valid)`: the 2D logits (the
quantity the PNG is made from; `hecate.py` itself is not modified).
- **REF:** stock `nn.Conv3d`, fp32 (autocast off). The fp32 stock reference of the task.
- **STOCK:** stock `nn.Conv3d`, bf16 autocast (what the repro and every Hecate run uses).
- **F32T:** `conv3d_fast.enable(mode="fp32_taps")`, bf16 autocast. Activation as everywhere: `PYTHONPATH=[internal reference removed] CONV3D_FAST=1
  CONV3D_FAST_MODE=fp32_taps` (here the same call made inside the script; the conv path is recorded as `conv3d_fast+fp32_taps`).
- **REFF (record only):** conv3d_fast default taps in fp32, whole render, to give G2 a fp32 yardstick.

## Gates (fixed now; no bar is loosened after any output; an amendment may only be made before output exists and is dated)

- **G1, bf16 noise, pooled, per scale.** Per patch, over all 256 x 256 (2.4 µm) or 64 x 64 (9.6 µm) logit values: dS = |STOCK - REF|,
  dF = |F32T - REF|. Pooled over the scale's patches: p99 (`np.percentile` on the concatenated values) and mean.
  **Pass (each scale separately):** pooled p99(dF) / p99(dS) <= 1.10 AND pooled mean(dF) / mean(dS) <= 1.10.
  Reported: per-patch ratios (median, range, share of patches over 1.10), and the same distances in probability grey levels.
- **G2, output level (whole render, batch 1, both arms through `hecate.predict`, stride default, bf16).** The final uint8 ink map
  of F32T vs STOCK. Two pixel sets: (a) *support*: columns with any nonzero voxel in the central window (set by the input, not by
  any output); (b) *nonzero*: support pixels where max(STOCK, F32T) >= 1. **Pass (each scale):** the fraction of pixels with
  |F32T - STOCK| <= 1 grey is >= 99 % in BOTH sets. Reported: Pearson correlation, mean and max |diff|, and, as a record, the same
  fraction of each arm against the whole-render fp32 map (REFF).
- **G3, speed and memory (per scale).** Per patch: median forward time STOCK vs F32T over the G1 patches (synchronized, after
  2 warm-up forwards; kernel builds not counted). Whole render: wall time of `predict` for each arm after a warm-up forward, batch 1,
  with s per cm². **Pass:** (i) patch speedup >= 1.3x AND whole-render speedup >= 1.3x; (ii) peak reserved VRAM <= **30.5 GB** in the
  F32T whole render, in the G1 process, and in the batch-4 F32T render (2.4 µm; Sebastien's K0 setting). Also recorded: STOCK batch 4.
- **Verdict:** **PASS** = G1, G2 and G3 pass at both scales. **FAIL** = any fails (a scale that passes alone is reported, but the
  CAPACITY row follows the verdict per scale as stated in the result log). **VOID** = a render does not complete, or a step does not
  run on the intended conv path (checked by the arm's `conv_path` field).

## What each outcome changes

- **PASS (per scale):** `fp32_taps` bf16 may be used for Hecate runs at that scale in place of stock bf16; CAPACITY row and
  `capacity.json` switch to YES with the measured speedups. Sebastien decides team use.
- **FAIL / VOID:** no change: Hecate stays stock bf16; the row records which gate failed.
- Not claimed: anything about other models, other scrolls, or whether Hecate's ink map is right (this is equivalence to stock bf16).
  The 3D output path (`--output-3d`) and `--reverse` are not tested (the reverse run uses the same layers on flipped input).

## Compute and process

- 40 + 64 patches x 3 arms: about a minute. Whole renders at batch 1: 2.4 µm stock ~210 s, F32T ~110 s, REFF; 9.6 µm ~50 s each;
  batch-4 pair at 2.4 µm ~7 min. About 20 minutes on one card. One `[internal reference removed] claim hecate-gate --hours 1.5`, released by the job's
  last line (also on failure). Before every step the job runs `[internal reference removed] --home [internal reference removed] status` and
  `[internal reference removed] status`, and stops if a v6 remote job is running or queued (Seb's work has priority).
- Outputs in `[internal reference removed]`: `plan.json`, `g1_<scale>.json`, `render_<scale>_<arm>_b<batch>.json`, `hecatefast_summary.json`.
  Arrays and maps stay in git-ignored `[internal reference removed]` and are deleted afterwards.
- **Kill date:** 2026-10-12. Not scored by then: closed as not run.
