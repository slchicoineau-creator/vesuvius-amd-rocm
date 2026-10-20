# LASAGNA-SCOPE2 result (2026-10-13, partner): acc32_bias PASS on both extra crops (PHerc0500P2, PHerc0814)

Prereg `evidence/preregs/2026-10-13_partner_lasagna_scope2.md` (b13692c3) + amendment 1 (a76c053d, 0500P2 has 100 predict3d calls, tile draw redone over 100 before the rerun), both pushed before the outputs they govern. Frozen mode/code (`conv3d_fast.py` blob 72bd4f7c, 3 layers, rest stock), model `lasagna_model_current.pt`, 256^3 crops, 16 tiles per crop, REF fp32 + STOCK + FAST computed fresh per crop, card 1. Rule per crop with the Bonferroni 97.5 % one-sided bound (2 crops), PASS iff U <= 1.10.

| crop (public 9.362 um, 113 keV, 1.2 m) | crop xyzwhd | tiles | R-hat | U (97.5 %) | L (2.5 %) | per-tile r | verdict | forward stock / fast | whole predict3d stock / fast | e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| PHerc0500P2 (`20250820143440`) | 2322 1970 3400 256^3 | 16 of 100 | 0.429 | 0.574 | 0.354 | 0.985 | **PASS** | 16.55 / 0.689 s (24.0x) | 1668.3 s / 85.8 s (100 calls) | **19.4x** |
| PHerc0814 (`20250804134230`) | 3250 2764 9567 256^3 | 16 of 125 | 0.920 | 0.963 | 0.878 | 0.762 | **PASS** | 16.58 / 0.690 s (24.0x) | 2094.9 s / 114.8 s (125 calls) | **18.2x** |
(0841, GATE16c, for comparison: R-hat 0.409, U 0.516 at 95 %, e2e 16.3x.) 95 % bounds, reported only: 0500P2 0.537, 0814 0.957.

Absolute size (mean |err| vs fp32 REF, raw / grey after sigmoid / share of voxels > 1 grey): 0500P2 stock 0.00695 / 0.411 / 10.3 %, fast 0.00298 / 0.174 / 0.63 %, max raw 0.361 vs 0.070. 0814 stock 0.00235 / 0.137 / 0.23 %, fast 0.00216 / 0.125 / 0.011 %, max raw 0.085 vs 0.045.

## Honest reading
- 0814: fast is only about 8 % better than stock overall (R 0.92) and tiles range 0.74-1.09 (four tiles at 1.00-1.09, none above 1.09); PASS with margin 0.14 on the bound, but this crop shows no large gain in accuracy, only equivalence. The per-tile r of 0.76 is low because stock errors are small and similar across tiles (reported; it only matters for a non-PASS).
- 0500P2: partly masked (56 % non-zero input); 5 of the 16 drawn tiles (calls 15, 39, 55, 56, 58) have byte-identical fp32 REF and identical errors (same tile content), so there are about 12 distinct tiles, and those 5 sit at ratio 0.29. Sensitivity, reported only: keeping one of the five copies (12 tiles) gives R-hat 0.580, U 0.788 (97.5 %): still PASS. The tile ratios otherwise run 0.43-1.10.
- Amendment 1 disclosure: the first 0500P2 draw (125-call assumption) failed before scoring (indices beyond 100); no verdict for it exists; its 13 partial tensors are kept in `old_draw1/`, unused.
- Scope now: 3 crops of 3 scrolls (0841, 0500P2, 0814), one model, one scan class (113 keV / 1.2 m / 9.36 um). Not shown: 2.4 um or 7.9 um scans, other models, fitted surfaces, other overlaps (measured at the default overlap 64), surface correctness. A 7.91 um PHerc1667 volume is not in the public bucket (only 2.399 and 1.129 um).
OC line at n = 16 for the 95 % rule (copy-model residuals): FAIL 11.7 % at true R = 1.00, 84.9 % at 1.05, 99.2 % at 1.10; not measured for the 97.5 % bound. Note added 2026-10-14 (post hoc): on the 12 PHerc0814 tiles with at least 69 % real input data (zero fill beyond the downloaded chunks otherwise) R-hat is 0.956, U 0.987; the 5 identical 0500P2 tiles are near-empty input. See `RESULTS.md` section 10.
Files: `PHerc0500P2/scope2.json`, `PHerc0814/scope2.json`, `ref/both/e2e_*.json`, logs, `job.log`, `job_amd1.log`.
