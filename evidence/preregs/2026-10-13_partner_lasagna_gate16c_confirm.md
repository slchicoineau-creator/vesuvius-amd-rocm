# Pre-registration: LASAGNA-GATE16c (confirmatory). Does the frozen `conv3d_fast` acc32_bias mode on the 3 naive-kernel layers stay equivalent to stock bf16 on FRESH tiles? (partner, Beast)

- **Date:** 2026-10-13. Pushed before any gate16c output exists. **Owner:** partner (lane B). Decision owner for team use: Sebastien.
- **Purpose:** GATE16b (`prereg/2026-10-13_partner_lasagna_gate16b.md`, result `work/rocm_ports/out/lasagna_gate16b/RESULT.md`: PASS, R-hat 0.427, U 0.540, e2e 17.2x) chose acc32_bias after a 3-tile look at tiles 0-2 of its own 16 gate tiles (disclosed selection effect). This run lifts that caveat on tiles that played no part in the choice.
- **Parents:** GATE16 (`..._lasagna_gate16.md`), GATE16b, rule `prereg/2026-10-12_partner_fastpath_ruler.md` + `..._amend1.md`. No labels, no humans, no keys, no w00: every error is against the fp32 stock reference of the same input tile.

## Frozen (no change allowed)
- **Mode:** `conv3d_fast` mode `acc32_bias`, exactly the 3 layers `shared_encoder.stages.0.blocks.0.conv1.conv`, `shared_encoder.stages.0.blocks.0.conv2.conv`, `shared_decoder.stages.5.convs.0.conv`; every other Conv3d pinned to stock.
  Code: `lanes/B_ink/release/amd_rocm/conv3d_fast.py` (git blob 72bd4f7c6a52ad2e641cea3b09897b7b4c6e6a41, sha256 94f4db958bfe9b6fc5a8c194d545ee5e3e8a2f6e560bb97536740436d892b309, last changed in 28a4c1d2; unmodified in the working tree).
  Gate script `work/rocm_ports/out/lasagna_gate16c/las_gate16c.py` (sha256 1ad48871b68b0828fb334b175d5f3f6b81fcf9c3acb91a555b71be4a1c451171) and `job_gate16c.sh` (sha256 3e468f8b8fd38cdde570bfa7794bd8bd4b6379352f75e25109757e782894d4f9);
  diff to `las_gate16b.py`: output paths, tile selection, REF mode re-enabled (fresh fp32 REF), `G16`/`REFC` point at the gate16c folder (stock end-to-end is re-run here). Scoring code, rule module, bootstrap seed 20261012 unchanged.
- Model/data/command as GATE16 (`lasagna_model_current.pt`, PHerc0841 crop x4372 y3722 z9572, 256^3, `predict3d` as `scripts/las_predict.sh`). Card 1, bf16 autocast, `expandable_segments:False`.

## Fresh tiles
- 125 tiles; excluded: the 16 GATE16/GATE16b tiles = `default_rng(20261012).choice(125, 16, replace=False)` sorted = [1, 37, 44, 46, 52, 67, 72, 76, 86, 91, 94, 102, 104, 116, 122, 124]
  (this set contains the 3-tile look, which was tiles 0-2 of it = calls 1, 37, 44). Pool = the other 109 tiles in ascending order.
- Draw: `pool[i] for i in default_rng(20261013).choice(109, 16, replace=False)`, sorted. Computed before the run: **[6, 9, 12, 23, 30, 35, 42, 56, 74, 88, 89, 90, 99, 118, 120, 123]** (the script asserts pool size 109 and disjointness).

## Arms
- REF = stock fp32 (autocast off, `ROCBLAS_USE_HIPBLASLT=0`), computed fresh for these 16 tiles exactly as GATE16 did. STOCK = stock bf16 autocast, HIPBLASLT=1, fresh. FAST = bf16 autocast, acc32_bias on the 3 layers, HIPBLASLT=1.

## Rule (unchanged, incl. amendment 1 / A3 wording)
- E_c = summed |raw output - REF| per 192^3 tile; R = ratio of sums FAST/STOCK; 2,000 bootstrap resamples, seed 20261012; one-sided 95 % upper bound U and 5th percentile L; **PASS iff U <= 1.10**; n = 16.
- U > 1.10 and L > 1.00: FAIL (worse than stock). U > 1.10 and L <= 1.00: NOT SHOWN EQUIVALENT. Per-tile r < 0.8 makes a non-PASS NOT SHOWN EQUIVALENT.
- One mode, one test, so the one-sided 95 % bound needs no multiplicity correction. Reported, not gates: per-tile ratios, mean |err| raw / share of mean |REF| / 8-bit sigmoid grey / share of voxels > 1 grey (stock and fast), OC line at n = 16.
- **Speed (fresh):** forward time per selected tile, stock vs fast (2 warm-up forwards, median); and one complete 125-tile `predict3d` pass each, stock and fast, whole-process wall measured by the job. Speed claim only if e2e factor >= 2.0x AND the rule PASSES.

## What each outcome changes
- PASS and e2e >= 2.0x: the caveat on GATE16b is lifted: acc32_bias on the 3 layers is equivalent to stock bf16 on fresh tiles of this model; the measured end-to-end factor from this run is the one quoted (GATE16b history kept).
  Not claimed: other crops/scrolls/models, downstream fitted surfaces, `overlap 64`, any claim about the surface being right.
- FAIL / NOT SHOWN EQUIVALENT: acc32_bias is not confirmed; the speed-up is not claimable; stock bf16 stays. No further mode is tried under this prereg.

## Compute
`gpu-share job set --card 1` (as before). Another session holds a priority-2 claim named `localcc` at times: not touched; if active the job waits and the waiting time is reported. About 10 min (ref + both) plus about 35 min (stock e2e) plus about 4 min (fast e2e). Kill date 2026-10-26.
Outputs: `work/rocm_ports/out/lasagna_gate16c/` (`gate16c.json`, `ref.json`, `both.json`, `e2e_*.json`, logs, RESULT.md); tensors in git-ignored `work/cache/lasagna_gate16c/`.
