# Pre-registration: LASAGNA-SCOPE2. Does the frozen `conv3d_fast` acc32_bias mode stay equivalent to stock bf16 on two more public crops (other scrolls)? (partner, Beast)

- **Date:** 2026-10-13. Pushed before any scope2 model output exists. **Owner:** partner (lane B).
- **Purpose:** widen the package scope caveat "one crop, one scroll, one model" (GATE16/16b/16c all on PHerc0841 crop x4372 y3722 z9572). Same model, same frozen mode; new inputs only. No labels, no humans, no keys, no w00, no private scroll: every error is against the fp32 stock reference of the same tile.
- **Parents:** `..._lasagna_gate16c_confirm.md` (result PASS, R-hat 0.409, U 0.516, e2e 16.3x), GATE16b, rule `..._fastpath_ruler.md` + `..._fastpath_ruler_amend1.md`.

## Crop choice (written rule, applied before any output)
- Candidates: the scrolls named in the brief with a public masked 9.36 um (1.2 m, 113 keV) volume: PHerc0139, PHerc0500P2, PHerc0814. (A 7.91 um PHerc1667 volume is NOT in the public bucket listing of 2026-10-13: PHerc1667 has only 2.399 um and 1.129 um volumes, which this 9.4 um model does not target. private scrolls excluded.)
- Draw: `np.random.default_rng(20261014).choice(3, 2, replace=False)` over the alphabetical list [0139, 0500P2, 0814] = [1, 2] -> **PHerc0500P2** (`20250820143440-9.362um-1.2m-113keV-masked`, 4196 x 4196 x 7057) and **PHerc0814** (`20250804134230-9.362um-1.2m-113keV-masked`, 5784 x 5784 x 19390).
- Crop size 256^3 (same as 0841). Origin = the same RELATIVE position as the 0841 crop: x0 = round(0.584 W - 128), y0 = round(0.500 H - 128), z0 = round(0.500 D - 128). Result: 0500P2 `x2322 y1970 z3400`, 0814 `x3250 y2764 z9567`.
- Validity rule (input only, no model output): crop must have >= 30 % non-zero voxels, else it is moved +256 in x until it does. Measured on the downloaded input: 0500P2 56.3 % non-zero (partly masked), 0814 99.98 % (0841 crop: 99.92 %). Both valid, no move. Disclosed: the origins were computed and the inputs' grey statistics were looked at before this commit; no model was run.
- Scope honesty: both crops are the same scan class (113 keV, 1.2 m, 9.362 um) as the 0841 crop; they differ in scroll, sample and mask coverage, not in scanner settings.
- Data fetch: `work/rocm_ports/scripts/fetch_scope_crop.py` (raw public chunks, +1 chunk margin) -> git-ignored `work/rocm_ports/data/lasagna_scope/<scroll>/vol.zarr/0`.

## Frozen (no change allowed)
- **Mode / code:** `conv3d_fast` mode `acc32_bias` on exactly the 3 layers `shared_encoder.stages.0.blocks.0.conv1.conv`, `shared_encoder.stages.0.blocks.0.conv2.conv`, `shared_decoder.stages.5.convs.0.conv`; all other Conv3d pinned to stock. `lanes/B_ink/release/amd_rocm/conv3d_fast.py` git blob 72bd4f7c6a52ad2e641cea3b09897b7b4c6e6a41 (unchanged since 28a4c1d2; unmodified in the working tree).
- **Scripts:** `work/rocm_ports/out/lasagna_scope2/las_scope2.py` (sha256 eeca87e6965d41ac218ad20236ccd9ce56b5d89e9bc91c6bd342ae20b393720c) and `job_scope2.sh` (sha256 c1f1bb0dd52469e918b8e34c9af37daf68b8618773475f1d3ca5ef13301ba1d6). Diff to `las_gate16c.py`: crop name from env `LS_CROP`, input zarr and `--crop-xyzwhd` from the crop folder, output/cache per crop, tile draw (below), Bonferroni bounds (below), the copy-residual OC block removed. Model `lasagna_model_current.pt`, `predict3d` flags, card 1, bf16 autocast, `expandable_segments:False` as GATE16c.
- **Tiles:** 16 per crop out of the 125 `predict3d` calls (the script asserts 125 calls in the e2e run), from ONE generator `rng = default_rng(20261014)`: first `rng.choice(125, 16, replace=False)` for PHerc0500P2, second for PHerc0814, each sorted. Computed before the run: **0500P2 [4, 13, 19, 41, 49, 53, 62, 66, 69, 70, 73, 86, 91, 101, 111, 117]; 0814 [9, 23, 26, 45, 48, 56, 71, 83, 85, 91, 102, 103, 109, 114, 121, 122]**.

## Arms and rule (per crop, same as GATE16c; Bonferroni for 2 crops)
- REF = stock fp32 (autocast off, `ROCBLAS_USE_HIPBLASLT=0`), STOCK = stock bf16, FAST = acc32_bias bf16 (HIPBLASLT=1); all computed fresh per crop.
- E_c = summed |raw output - REF| per 192^3 tile; R = ratio of sums FAST/STOCK; 2,000 bootstrap resamples, seed 20261012; **one-sided 97.5 % upper bound U (percentile 97.5) and 2.5th percentile L** (Bonferroni, 2 crops); **PASS iff U <= 1.10**. U > 1.10 and L > 1.00: FAIL (worse than stock). U > 1.10 and L <= 1.00: NOT SHOWN EQUIVALENT. Per-tile Pearson r < 0.8 makes a non-PASS NOT SHOWN EQUIVALENT (amendment 1 wording). Each crop is judged on its own; no pooling. Reported, not gates: the 95 % bound, per-tile ratios, mean |err| raw / grey after sigmoid / share of voxels > 1 grey, forward time.
- **Speed (per crop):** forward per selected tile, stock vs fast (2 warm-ups, median); one complete 125-tile `predict3d` pass each, stock and fast, whole-process wall. A speed factor is quoted for a crop only if its rule PASSES and its e2e factor >= 2.0x.

## What each outcome changes
- Per crop PASS: the package scope statement lists that scroll/crop as measured equivalent. Both PASS: scope = 3 crops, 3 scrolls, one model, one scan class. A FAIL or NOT SHOWN EQUIVALENT on a crop is stated in the package as is (including a per-crop speed claim withdrawn); acc32_bias is then not claimed for that scroll. No further mode is tried under this prereg.
- Not claimed in any case: other models, other scan classes (2.4 um, 7.9 um), fitted surfaces downstream, `overlap 64`, anything about the surface being right.

## Compute
`gpu-share job set --card 1` (per-card claims; card 0 and other sessions' jobs/claims untouched; if card 1 is held the job waits and the wait is reported). About 60 min per crop (ref+both 10, stock e2e 35, fast e2e 4, load). Kill date 2026-10-26. Outputs `work/rocm_ports/out/lasagna_scope2/<scroll>/` (`scope2.json`, `ref.json`, `both.json`, `e2e_*.json`, logs); tensors in git-ignored `work/cache/lasagna_scope2/`.
