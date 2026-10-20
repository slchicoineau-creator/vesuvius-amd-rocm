# LASAGNA-GATE16c result (2026-10-13, partner): PASS on 16 fresh tiles; acc32_bias confirmed; measured end-to-end 16.3x

Prereg `evidence/preregs/2026-10-13_partner_lasagna_gate16c_confirm.md` (7de8a076, pushed before the run). Frozen mode (conv3d_fast.py blob 72bd4f7c, unchanged), 16 tiles drawn with seed 20261013 from the 109 tiles outside GATE16/16b (and outside the 3-tile look):
[6, 9, 12, 23, 30, 35, 42, 56, 74, 88, 89, 90, 99, 118, 120, 123]. REF (fp32) computed fresh, STOCK re-run, FAST = acc32_bias on the 3 registered layers. Card 1, no GPU-scheduler claim was active, no waiting (job started 09:18:53, 13 s after queuing).

## Rule (n = 16; unit = summed |raw output - fp32 REF| per 192^3 tile)
R-hat **0.409**, **U 0.516**, L 0.350, per-tile Pearson r 0.994 -> U <= 1.10: **PASS**. (GATE16b on its 16 tiles: R-hat 0.427, U 0.540.)
Per-tile ratios: 9 tiles at 0.29-0.47, 0.85, 0.96, 1.00, and 4 tiles at 1.11-1.13 (no tile above 1.13): the same bimodal pattern as GATE16b. OC line at n = 16 (copy-model residuals): FAIL 11.7 % at true R = 1.00, 84.9 % at 1.05, 99.2 % at 1.10 (not measured for the 97.5 % bound). Note added 2026-10-14 (post hoc): 7 of these 16 tiles have under 10 % real input data (zero fill beyond the downloaded chunks, ratios 0.29-0.31); on the 6 tiles with at least 69 % real data the pooled ratio is 1.07 (U 1.12, L 1.03). See `RESULTS.md` section 10, zero-fill caveat.
Absolute size: stock mean|err| 0.00751 raw (1.38 % of mean|REF|), 0.442 grey after sigmoid (11.7 % of voxels > 1 grey), max 0.365; fast 0.00307 raw (0.57 %), 0.179 grey (0.59 % of voxels > 1 grey), max 0.070.

## Speed (all fresh)
Forward per tile (median of 16): stock 16.54 s, fast 0.686 s = 24.1x. End to end, whole 125-tile predict3d process: stock 2105.5 s, fast 128.8 s = **16.3x** (GATE16b: 17.2x with a reused stock wall). Above the 2.0x bar.
Not claimed: other crops/scrolls/models, fitted surfaces downstream, other overlaps (measured at the default overlap 64), anything about the surface being right.
Files: gate16c.json, ref.json, both.json, e2e_stock.json, e2e_fast.json, las_gate16c.py, job_gate16c.sh, logs.
