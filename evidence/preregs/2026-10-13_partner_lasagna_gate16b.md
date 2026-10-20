# Pre-registration: LASAGNA-GATE16b. Is `conv3d_fast` acc32_bias on the 3 naive-kernel layers equivalent to stock bf16 for lasagna predict3d, and what is the measured end-to-end speed-up? (partner, [internal reference removed])

- **Date:** 2026-10-13. Pushed before any gate16b output exists. **Owner:** partner (lane B). Decision owner for team use: Sebastien.
- **Parents:** `[internal reference removed]` (GATE16, result `[internal reference removed]`: fp32_taps FAIL, R-hat 2.061, U 2.168, L 1.904, e2e 10.3x),
  `[internal reference removed]` + `..._amend1.md` (the rule, A3 wording). Diagnosis: `[internal reference removed]` (written and committed with this file; all diagnosis runs are exploratory, not gate evidence).
- **Truth rule:** no labels, no humans, no keys, no w00. Every error is against the fp32 stock reference computed on the same input tile.

## Why this mode (from DIAG.md, short)
fp32_taps is the correctly rounded layer (best per layer) yet doubled the end-to-end error; the doubling comes from layer L1 alone (a coherent output offset made downstream; mechanism not isolated).
Per-layer accuracy therefore does not predict end-to-end error, so ONE mode is chosen on its arithmetic plus the 3-tile end-to-end diagnosis: **acc32_bias** (taps and bias summed in fp32, one rounding;
end-to-end 0.47 / 1.11 / 0.31 x stock on tiles 0-2). `taps` is not carried (worse than stock per layer and end to end on all 3 diagnosis tiles). `fp32_taps` is not re-run.
**Disclosed selection effect:** the 3 diagnosis tiles are tiles 0-2 of the 16 gate tiles. A PASS therefore means "no worse than stock on these 16 tiles of this model", not a claim on fresh tiles. One mode only, so the unchanged one-sided 95 % bound (no Bonferroni needed).

## Setup (frozen; identical to GATE16 except the mode)
- Model/data/command: as GATE16 (`lasagna_model_current.pt`, PHerc0841 crop x4372 y3722 z9572, 256^3, `predict3d` as `scripts/las_predict.sh`). Script `[internal reference removed]` (diff to `las_gate16.py`: FASTMODE, paths, REF/e2e-stock reuse only).
- **16 tiles:** `numpy.random.default_rng(20261012).choice(125, 16, replace=False)` sorted, asserted 125 tiles. Same as GATE16.
- **Arms:** REF = the GATE16 fp32 REF arrays (`[internal reference removed]`, stock fp32, HIPBLASLT=0), reused, not recomputed. STOCK = stock bf16 autocast, HIPBLASLT=1, re-run in this job
  (expected bit-identical to GATE16's STOCK_k, as shown on 3 tiles in DIAG.md; compared and reported). FAST = bf16 autocast, `conv3d_fast.layer_mode(..., "acc32_bias")`... exactly the 3 layers
  `shared_encoder.stages.0.blocks.0.conv1.conv`, `shared_encoder.stages.0.blocks.0.conv2.conv`, `shared_decoder.stages.5.convs.0.conv`; every other Conv3d pinned to stock. HIPBLASLT=1, `expandable_segments:False`.
- **Rule (unchanged):** E_c = summed |raw output - REF| per 192^3 tile; R = ratio of sums FAST/STOCK; 2,000 bootstrap resamples, seed 20261012, one-sided 95 % U and 5th percentile L; **PASS iff U <= 1.10**;
  n = 16; per-tile r reported. A3 wording as in GATE16: U <= 1.10 PASS; U > 1.10 and L > 1.00 FAIL (worse than stock); U > 1.10 and L <= 1.00 NOT SHOWN EQUIVALENT; r < 0.8 makes a non-PASS NOT SHOWN EQUIVALENT.
  Reported, not gates: mean |err| raw / share of mean |REF| / 8-bit sigmoid grey / share of voxels > 1 grey, for stock and fast; OC line at n=16.
- **Speed (measured):** (i) forward time per selected tile, stock vs fast (2 warm-up forwards, median). (ii) end-to-end: one complete 125-tile `predict3d` pass FAST (wall of the whole process, from the job).
  The stock end-to-end wall is reused from GATE16 (2098.8 s, same card, same command; it was not re-run to save about 35 GPU-min; reported as reused). Speed claim only if e2e factor >= 2.0x AND the rule PASSES.

## What each outcome changes
- PASS and e2e >= 2.0x: the 3-layer acc32_bias path is equivalent in bf16 rounding to stock on these 16 tiles of this model (not a reading tolerance), with the measured speed-up; use in a team lasagna run is Sebastien's call.
  Not claimed: other crops/scrolls/models, downstream fitted surfaces, `overlap 64`, any claim about the surface being right.
- FAIL / NOT SHOWN EQUIVALENT: stock bf16 stays for lasagna; no further mode is tried under this prereg (a new mode needs a new prereg and a new reason).

## Compute
`[internal reference removed] job set --card 1` (as GATE16); nothing displaced. About 10 min (16 tiles stock + fast) plus about 4 min (fast pass). Kill date 2026-10-26.
Outputs: `[internal reference removed]` (`gate16b.json`, `both.json`, `e2e_fast.json`, logs, RESULT.md); tensors in git-ignored `[internal reference removed]`.
