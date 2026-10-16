# LASAGNA-GATE16 result (2026-10-13, partner, Beast): FAIL (worse than stock); measured end-to-end speed-up 10.3x but not usable under the rule

Prereg `prereg/2026-10-13_partner_lasagna_gate16.md` (978906e7, pushed before any output). Condition: FASTPATH-RULER S2 did not kill the rule (taps/acc32_bias false pass at 1.10: 1.2 % / 0.4 %).
Card 1 (gpu-share card job), bf16 autocast, ROCBLAS_USE_HIPBLASLT=1 (REF process 0), 16 of the 125 tiles (seed 20261012, call indices in gate16.json). FAST = fp32_taps on exactly the 3 registered layers, all other Conv3d pinned stock. The job was paused once by another claim and restarted from zero; the reported steps ran uninterrupted.
No `libmagma.so` / oneMKL error occurred in any log.

## Rule (n = 16; unit = summed |raw output - fp32 REF| per 192^3 tile)
R-hat **2.061**, **U 2.168**, L 1.904, per-tile r 0.994 -> U > 1.10 and L > 1.00: **FAIL (worse than stock)**.
OC line (n = 20): FAIL 5.7 % at R = 1.00, 81 % at 1.05, 99.5 % at 1.10; at n = 16 (copy residuals): 11.7 % / 84.9 % / 99.2 %. Per-tile ratios in gate16.json (r 0.994, so this is not a pairing artefact).
Absolute size: stock mean|err| 0.00688 raw (1.27 % of mean|REF|), 0.405 grey after sigmoid (10.5 % of voxels > 1 grey); fast 0.01418 raw (2.62 %), 0.840 grey (21.3 % > 1 grey); max |err| 0.36 vs 0.38 raw.
So fast roughly doubles stock's bf16 error on this model; both are small in absolute terms. Not a reading tolerance either way.

## Speed (measured)
Forward per tile (16 tiles, median): stock 16.59 s, fast 1.35 s = 12.3x. End-to-end, whole 125-tile predict3d process (start-up and I/O included): stock 2098.8 s (16.79 s/tile), fast 203.9 s (1.63 s/tile) = **10.3x** measured. Both well above 2.0x, but the claim is NOT usable: the rule FAILS.

## Reading
Stock's 3 slow layers run MIOpen's naive bf16 kernel, which has its own (smaller) rounding error against fp32 here; fp32_taps rounds activations and weights to bf16 first and sums taps in fp32, which on these full-resolution layers gave twice stock's output error. The cheap `taps` / `acc32_bias` modes were not tested (a different path needing its own gate; offered, not run). Decision for Sebastien/Chris: keep stock bf16, or pre-register a gate for a different mode on those 3 layers (the 10x speed is available if a mode passes).
Files: gate16.json, ref/both/e2e_*.json, *.log, las_gate16.py, job_gate16.sh.
