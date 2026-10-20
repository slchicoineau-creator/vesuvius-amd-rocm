# LASAGNA-GATE16b result (2026-10-13, partner, Beast): PASS for acc32_bias on the 3 layers; measured end-to-end 17.2x

Prereg `evidence/preregs/2026-10-13_partner_lasagna_gate16b.md` (2322b0ca, pushed before the run); diagnosis `DIAG.md`. Card 1 via gpu-share, bf16 autocast, ROCBLAS_USE_HIPBLASLT=1, the same 16 tiles (seed 20261012) as GATE16.
REF = GATE16's fp32 arrays (reused). STOCK re-run: all 16 STOCK arrays are bit-identical to GATE16's. FAST = `conv3d_fast` acc32_bias on exactly the 3 registered layers, every other Conv3d pinned to stock. One mode, one-sided 95 % bound as registered.

## Rule (n = 16; unit = summed |raw output - fp32 REF| per 192^3 tile)
R-hat **0.427**, **U 0.540**, L 0.364, per-tile Pearson r 0.990 -> U <= 1.10: **PASS** (registered verdict: no worse than stock on these 16 tiles). Note added 2026-10-14 (post hoc): the ratio 0.43 is dominated by tiles whose input is mostly zero fill (the local zarr held only the crop's chunks); on the 7 tiles of this set with at least 69 % real data the ratios are 1.01-1.13 (stock is not beaten there). See `RESULTS.md` section 10, zero-fill caveat.
Per-tile ratios (gate16b.json) are bimodal: 8 tiles at 0.29-0.47, 1 at 0.81, 7 at 1.01-1.13; no tile is above 1.13. (7 of 16 tiles are slightly worse than stock, up to 13 %; the pooled ratio is dominated by the large-error tiles, which improve.) OC line at n = 16: FAIL 11.7 % at true R = 1.00, 84.9 % at 1.05, 99.2 % at 1.10.
Absolute size: stock mean|err| 0.00688 raw (1.27 % of mean|REF|), 0.405 grey after sigmoid (10.5 % of voxels > 1 grey), max 0.361; fast 0.00294 raw (0.54 %), 0.171 grey (0.48 % of voxels > 1 grey), max 0.070.
Caveat (disclosed in the prereg): the mode was chosen with the help of a 3-tile diagnosis on tiles 0-2 of these 16. PASS = "equivalent in bf16 rounding to stock on these 16 tiles of this model" (not a reading tolerance, not fresh tiles).

## Speed (measured)
Forward per tile (16 tiles, median): stock 16.51 s, fast 0.686 s = 24.1x. End to end, whole 125-tile predict3d process: stock 2098.8 s (REUSED from GATE16, not re-run), fast 122.1 s (0.98 s/tile) = **17.2x**. Above the 2.0x bar.
Not claimed: other crops/scrolls/models, fitted surfaces downstream, other overlaps (measured at the default overlap 64), anything about the surface being right.
Files: gate16b.json, both.json, e2e_fast.json, las_gate16b.py, job_gate16b.sh, logs.
