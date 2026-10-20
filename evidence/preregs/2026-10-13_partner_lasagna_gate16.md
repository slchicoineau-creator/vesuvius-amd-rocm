# Pre-registration: LASAGNA-GATE16. Is `conv3d_fast` fp32_taps on the 3 naive-kernel layers equivalent to stock bf16 for lasagna predict3d, and what is the measured end-to-end speed-up? (partner, [internal reference removed])

- **Date:** 2026-10-13. Pushed before any gate output exists. **Owner:** partner (lane B). Decision owner for team use: Sebastien.
- **Parents:** `[internal reference removed]` (S0-S2 done, `[internal reference removed]`, d1d185d2:
  projected 12.7x >= 4.6x, "the 16-tile gate with pick 1's rule decides"), `[internal reference removed]` + `..._amend1.md` (the rule).
  **This gate runs only if the FASTPATH-RULER S2 did not KILL the rule** (condition checked and quoted in the result).
- **Truth rule:** no labels, no humans, no keys, no w00. Every error is against the fp32 stock reference computed on the same input tile.

## Setup (frozen)
- Model/data: `lasagna_model_current.pt`, public PHerc0841 crop x4372 y3722 z9572, 256^3, the exact `predict3d` command of
  `[internal reference removed]` (`--scaledown 2 --cos-scaledown 1`, tile 192, 125 tiles per pass). The tiles are the ones the tool builds
  (real preprocessing); the script `[internal reference removed]` only replaces `run_tile_inference` / `load_model` on the adapter.
- **16 tiles:** the tool's tile call indices (0..124), `numpy.random.default_rng(20261012).choice(125, 16, replace=False)`, sorted. Asserted at run time that
  the pass has 125 tiles. No selection on any output.
- **Arms**, per selected tile, same input tensor:
  - **REF:** stock `nn.Conv3d`, fp32 (autocast off), `ROCBLAS_USE_HIPBLASLT=0` at process start (the registered fp32 reference setting).
  - **STOCK:** stock `nn.Conv3d`, bf16 autocast (what upstream predict3d does), `ROCBLAS_USE_HIPBLASLT=1` at process start.
  - **FAST:** bf16 autocast, `conv3d_fast.layer_mode(..., "fp32_taps")` on **exactly 3 layers**, everything else stock (all other Conv3d are pinned to the stock
    forward on the instance, so nothing else is rewritten): `shared_encoder.stages.0.blocks.0.conv1.conv`, `shared_encoder.stages.0.blocks.0.conv2.conv`,
    `shared_decoder.stages.5.convs.0.conv` (the 3 kept layers of `s2_profile.json`). `ROCBLAS_USE_HIPBLASLT=1`, `expandable_segments:False`.
  - REF runs in its own process (the setting is fixed at process start); STOCK and FAST in a second process.
- **Rule (unchanged, from the ruler):** E_c = summed |raw model output - REF| over all voxels and channels of the 192^3 output tile; R = ratio of sums FAST/STOCK;
  2,000 bootstrap resamples, seed 20261012, one-sided 95 % U (and 5th percentile L); **PASS iff U <= 1.10**. **n = 16 tiles** (the brief's choice; the rule's OC
  was measured at n = 20, so the n = 16 FAIL rate at true R = 1.00 / 1.05 / 1.10 is also reported from the copy-model residuals).
- **Verdict wording (amendment A3):** quote R-hat, U, L, per-tile Pearson r (fast vs stock error), OC line. U <= 1.10 = PASS. U > 1.10 with L > 1.00 =
  FAIL (worse than stock). U > 1.10 with L <= 1.00 = NOT SHOWN EQUIVALENT (stock path kept). r < 0.8: a non-PASS is NOT SHOWN EQUIVALENT whatever L is.
  A4-style fields reported, not gates: mean |err| of stock and fast (raw, share of mean |REF|, in 8-bit sigmoid grey) and the share of voxels with grey error > 1.
- **Speed (measured, not projected):**
  (i) forward time per selected tile, stock vs fast (synchronized, 2 warm-up forwards on tile 1, median) ;
  (ii) **end-to-end:** two complete 125-tile `predict3d` passes, STOCK then FAST (3 layers), same command, wall time of the whole process measured by the job
  (includes start-up and I/O; seconds per tile = wall / 125). **Only the measured end-to-end factor (stock wall / fast wall) is claimed.** If it is < 2.0x the speed
  claim is killed (vet band) even when the rule PASSES; the 12.7x projection is not quoted as a result.

## What each outcome changes
- Rule PASS and end-to-end >= 2.0x: the 3-layer fast path is *equivalent in bf16 rounding to stock on these 16 tiles of this model* (not a reading tolerance), with the
  measured speed-up; any use in a team lasagna run is Sebastien's call. Not claimed: other crops/scrolls, other models, the fitted surfaces downstream, the
  `overlap 64` setting, any claim about the surface being right.
- FAIL / NOT SHOWN EQUIVALENT: stock bf16 stays; the speed-up is reported as measured but not usable under the rule.

## Compute
`[internal reference removed] job set --card <free card>` (the previous lasagna run's mechanism); nothing displaced. REF 16 tiles + STOCK/FAST 16 tiles (about 10 min stock + fast),
two full passes (stock about 35 min, fast a few minutes). If the transient `Cannot load libmagma.so` error recurs it is diagnosed (env / LD path) before any retry and reported.
Kill date: 2026-10-26. Outputs: `[internal reference removed]` (`gate16.json`, `ref.json`, `both.json`, `e2e_*.json`, logs); tensors in git-ignored `[internal reference removed]`.
