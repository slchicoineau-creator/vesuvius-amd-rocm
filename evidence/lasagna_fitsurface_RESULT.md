# LASAGNA-FITSURFACE: RESULT (partner, lane B, Beast; AI-written)

Job finished 2026-10-14 07:46 CDT (gpu-share rc 0, `JOB_DONE`). Preregistration: `evidence/preregs/2026-10-13_partner_lasagna_fitsurface.md` (083b308e), amendment 1 (0ae5cac5, margin copy, GATE16-REAL), amendment 2 (6986cfe4, Stage 0 ladder R = 2000 / 1900 / 2100 and `cyl_init` steps 0). All numbers below were recomputed from the raw files on 2026-10-14 (method at the end).

## Verdict

1. **Fitted-surface test (the registered decision): PASS as registered, on a narrower test than first registered.** Decision statistic M = 2.050 vx (19.2 um) against bar B = 6.757 vx (63.3 um); M <= B.
2. **GATE16-REAL (secondary, real-data tiles only): FAIL (worse than stock).** R-hat 1.105, U 1.113, L 1.096, r 0.978 on 16 tiles that all read real scan data; the rule needs U <= 1.10 for a PASS.
3. **How the two read together (amendment 1 wording, below):** the fitted-surface PASS is read as "the surfaces fitted from a not-shown-equivalent prediction". The package may not keep "within the 1.10 margin" for the lasagna fast path on real data.
4. **Scope limit (amendment 2), stated plainly:** this test compares the shipped patch fit on FAST vs REF predictions from a shared analytic cylinder shell; the registered cylinder-shell step could not run on a crop-sized prediction box.
5. **Open, not decided here:** whether this narrower test meets Sebastien's condition 2 is OPEN (issue #7, request WC-20261014-01). The decision is his, with the judge.

## The frozen rules, quoted verbatim from the prereg files

From `evidence/preregs/2026-10-13_partner_lasagna_fitsurface.md`:

> **Bar B = the largest REF-vs-REF seed-to-seed distance** (3 pairs: s1-s2, s1-s3, s2-s3 of REF), the registered rule's "largest same-arm seed-to-seed distance of the reference".
> **Decision statistic M = the median over the 9 cross pairs REF_si vs FAST_sj** (i, j in 1..3, including the 3 same-seed pairs).
> **PASS iff M <= B.** FAIL otherwise. INCONCLUSIVE if Stage 0 cannot run, or fewer than 9 + 3 fits finish.
> **Reported, not deciding:** the MAX over the 9 cross pairs, and whether it would also be <= B (the strict reading that failed the spiral cross-device check); FAST-vs-FAST and STOCK-vs-STOCK spreads; STOCK-vs-REF (9 pairs, median and max); FAST-vs-STOCK; all individual pair distances; the absolute size in vx and um (9.366 um/vx) and against the sheet scale (band width median 96 um = about 10 vx). If B is below 0.5 vx the bar is tighter than the fit's own grid noise: the registered verdict stands and is reported with that remark.
> STOCK-vs-REF is read as a scale only: if STOCK and FAST are both inside B the test cannot separate them (said plainly); it does not change the verdict.

From amendment 1, GATE16-REAL (same file):

> **PASS iff U <= 1.10**; U > 1.10 and L > 1.00 = FAIL (worse than stock); U > 1.10 and L <= 1.00 = NOT SHOWN EQUIVALENT; per-tile r < 0.8 turns a non-PASS into NOT SHOWN EQUIVALENT.
> GATE16-REAL PASS: the 3-layer `acc32_bias` fast path is within 1.10 of stock bf16's error on 16 real-data tiles of this crop (one crop, one scan, still no claim beyond that); the package may keep its "within the 1.10 margin" wording with this number added. NOT SHOWN EQUIVALENT or FAIL: the package states that the fast path is not shown equivalent to stock on real-data tiles and offers it only as an unvalidated speed option or withdraws the lasagna row; the fitted-surface result, whatever it is, is then read as "the surfaces fitted from a not-shown-equivalent prediction".

From amendment 2 (`evidence/preregs/2026-10-14_partner_lasagna_fitsurface_amend2.md`):

> So this test now measures "surface fitted by the shipped patch fit from FAST vs REF predictions, from the same start", not the cylinder-shell step. The shell step cannot run on a crop-sized prediction box at all (root cause 2), so it was never testable here.
> A PASS here says nothing about the cylinder-shell step or the whole-scroll chain (already listed as not claimed).
> The decision whether this narrower test still answers Sebastien's condition 2 is his, with the judge; the RESULT will say so.

## Numbers: fitted-surface test (`fitsurface.json`, `R_used.txt` = 2000)

Distance = `fit_compare.py` point-to-surface, mean of the two directional medians, after densifying each 5 x 5 vertex tifxyz to a pitch of 5.56 vx (all 9 fits: valid share 1.0). 1 vx = 9.366 um. Sheet scale: median band width 96 um, about 10.2 vx.

| quantity | vx | um | share of band width |
|---|---|---|---|
| **B** = largest REF-vs-REF (REF2-REF3) | 6.757 | 63.3 | 0.66 |
| **M** = median of 9 REF-vs-FAST pairs (the 5th, REF2-FAST1) | 2.050 | 19.2 | 0.20 |
| max of 9 REF-vs-FAST (REF3-FAST2), reported only | 6.832 | 64.0 | 0.67 |
| strict reading, max <= B? | **no** (6.832 > 6.757, by 0.075 vx = 0.7 um) | | |

Groups (each of the 9 or 3 pair distances is in `fitsurface.json`):

| group | n | median (vx) | max (vx) | pairs, in vx (s1-s2, s1-s3, s2-s3 for same-arm) |
|---|---|---|---|---|
| REF-REF (sets B) | 3 | 5.736 | 6.757 | 2.036, 5.736, 6.757 |
| FAST-FAST | 3 | 4.427 | 4.679 | 1.799, 4.427, 4.679 |
| STOCK-STOCK | 3 | 6.054 | 6.405 | 2.379, 6.054, 6.405 |
| **REF-FAST (decides)** | 9 | **2.050** | 6.832 | 0.164, 1.888, 4.239, 2.050, 0.271, 4.405, 5.931, 6.832, 0.894 (REF si vs FAST sj, i then j) |
| REF-STOCK (reported) | 9 | 2.341 | 6.758 | 0.593, 2.094, 5.787, 2.341, 0.089, 6.441, 6.105, 6.758, 0.426 |
| FAST-STOCK (reported) | 9 | 2.180 | 6.593 | 0.563, 2.096, 5.988, 2.180, 0.282, 6.593, 4.783, 4.403, 0.570 |

Post hoc description (my split of the same numbers; not a registered statistic): of the 9 REF-vs-FAST pairs, 3 use the same seed point and 6 do not.

| split | REF-FAST (vx) | REF-STOCK (vx) | FAST-STOCK (vx) |
|---|---|---|---|
| same seed point (prediction effect plus GPU run-to-run) | 0.164, 0.271, 0.894 (median 0.271; 1.5, 2.5, 8.4 um) | 0.593, 0.089, 0.426 (median 0.426) | 0.563, 0.282, 0.570 (median 0.563) |
| different seed point (seed effect) | median 4.322 | median 5.946 | median 4.593 |

## Numbers: GATE16-REAL (`gate16_real.json`)

16 tiles chosen by the registered draw ([4, 13, 17, 38, 47, 49, 59, 60, 64, 65, 66, 78, 84, 96, 106, 117]; matches `selected_calls` in `e2e_*.json`). All 125 tiles read real data (`coverage.json`: min real share 1.0, no chunk missing). Rule `work/ink/fastpath_ruler/rule.py` (sha256 f1d6c0f6...), unchanged.

| quantity | value |
|---|---|
| R-hat (ratio of summed errors, FAST / stock bf16, both against fp32 REF) | 1.105 |
| U (one-sided 95 % bound, 2,000 resamples, seed 20261012) | 1.113 |
| L (5th percentile) | 1.096 |
| per-tile r | 0.978 |
| verdict | **FAIL (worse than stock)** (U > 1.10 and L > 1.00) |
| per-tile ratios | all 16 above 1.00 (1.075 to 1.142); 8 of 16 above 1.10 |
| mean absolute raw error vs REF, stock / FAST | 0.00200 / 0.00221 (0.35 % / 0.39 % of mean absolute REF output) |
| mean absolute error in 0-255 sigmoid grey, stock / FAST | 0.115 / 0.127 grey levels |
| share of voxels with grey error above 1 level, stock / FAST | 2.9e-7 / 2.8e-8 |
| max absolute raw error, stock / FAST | 0.0211 / 0.0201 |
| operating characteristic of the rule at n = 16 (copy-model residuals, reported) | FAIL in 11.7 % of draws at true ratio 1.00, 84.9 % at 1.05, 99.2 % at 1.10 |

Plain reading: on tiles that read only real data the fast path has about 10 % more error against the fp32 model than stock bf16 has (earlier post hoc estimate on partly real tiles: R 1.067, U 1.092). In absolute size both errors are small (well under one grey level). The FAIL is the registered rule's wording for "above the 1.10 margin and above stock", not a statement that the output is visibly wrong. At n = 16 a true ratio near 1.10 fails this rule almost every time, so this result is consistent with a true ratio of about 1.10.

## Stage 0, plumbing, job

- Stage 0 (REF, seed 1) passed at the first rung R = 2000 (`R2000/stage0_check.json`: 5 x 5 vertices, valid share 1.0, inside crop +-64 vx 1.0, area 40,000 vx^2). There is no `stage0.json` because it is only written on failure; the failed first attempt (R = 900 / 600 / 1500, exit 78) is `stage0_attempt1_FAIL_exit78.json`, root cause in amendment 2 (the tool's radius-2000 cylinder never touched the data; a crop-sized box cannot drive the `cyl_init` step).
- All 9 fits and 3 shells finished, exit 0. The decision (REF and FAST) was scored at 07:04, before the STOCK predictions were made, as amendment 1 ordered; the final score at 07:46 re-wrote `fitsurface.json` with STOCK added and the decision block is identical.
- A FAST prediction attempt crashed at 01:11 (python 3.14 forkserver re-import, fixed by `FS_MODE`, commit 452443f0; prediction maths unchanged; the FAST predictions used come from the 01:13 re-run).
- Frozen-file check on 2026-10-14: `job_fitsurface.sh` 27b2797d..., `init_cyl_steps0.json` f996db33... (differs from shipped `configs/init_cyl.json` cdc9deeb... by the one line `"steps": 0`), `score.py` 2b2ab413..., `check0.py` 35cfe5c1..., `axis.py` 7a000054..., `mkman.py` 3a7a04e3..., `gate_recheck.py` a0af805b..., `fetch_margin.py` d8eee3ca..., `check_coverage.py` 6b338fe2..., `fit_compare.py` f6a7d355..., `rule.py` f1d6c0f6...: all equal to the hashes in the prereg and amendments. `las_fitsurface_pred.py` is 1a524f53... (amendment 1 listed 190c729b...; the difference is the declared `FS_MODE` fix of 452443f0, diff of 3 lines).

## GPU cap accounting (cap 4.0 GPU-h = 14,400 s, cumulative over restarts)

Sum of `logs/gpu_seconds.jsonl` = **5,563 s = 1.545 GPU-h**; cap not reached, exit 79 not triggered, 2.45 GPU-h unused. The running total printed at each restart (`used_s` 0, 104, 2917, 2923, 2927, 3173) matches the file line by line. Main items: `pred_ref` 2,618 s, `pred_stock` 2,143 s, `pred_fast` 104 s (crashed) + 124 s, the 8 failed attempt-1 steps about 170 s, 240 s booked as an upper-bound estimate for the two scratch single-rung checks of amendment 2, the 9 fits 13 to 53 s each, 4 shells 6 to 28 s. CPU steps (fetch, coverage, `gate_recheck`, score) are not counted, as registered. Card 1 only (`HIP=1` in every start line).

## What may and may not be claimed

May be said (internal; the package stays "pending Sebastien + judge"):
- By the frozen rule the fitted-surface test is a PASS: on this one crop (PHerc0841 9.366 um, x4372 y3722 z9572, 256^3), the shipped patch fit (`configs/init.json`, 1000 + 500 steps, depth 1, 200 x 200 vx) started from one shared analytic cylinder and a seed point gives surfaces from FAST predictions whose median distance to the REF-fitted surfaces (2.05 vx, 19 um) is inside the REF fits' own largest seed-to-seed distance (6.76 vx, 63 um). With the same seed point the FAST and REF surfaces are 0.16 to 0.89 vx apart (post hoc view).
- GATE16-REAL failed: on 16 tiles that read only real data, the 3-layer `acc32_bias` fast path was not within the 1.10 margin of stock bf16's error against fp32 (R-hat 1.105, U 1.113). The package must say the fast path is not shown equivalent to stock on real-data tiles (offered only as an unvalidated speed option, or the lasagna row withdrawn: Sebastien and the judge decide which).

May not be said:
- That the fast path was shown to give the same fitted surface as fp32 in the sense of Sebastien's condition 2. The test is narrower than registered (cylinder-shell step not run; all three arms start from the same analytic shell) and whether it meets his condition is open. The registered "re-run any number that goes into a submission on stock" stands until he says otherwise.
- That the test separates FAST from stock. It does not: STOCK also passes (REF-STOCK median 2.34 vx, max 6.76 vx), as the prereg says it must be read ("the test cannot separate them"). The bar is loose: B is 0.66 of a sheet width and is set by how far the fit moves when the seed point moves 16 vx. M is the 5th of 9 sorted pairs and 6 of the 9 are cross-seed pairs, so M mostly measures the seed effect, not the prediction effect.
- That the strict (max) reading holds. It does not: 6.832 vx > 6.757 vx (STOCK-vs-REF max 6.758 vx also sits 0.001 vx above B). Both are cross-seed pairs of the same size as the seed spread itself; this is reported as registered, not used to change the verdict.
- Anything about the cylinder-shell step, the real umbilicus, many shells, the whole-scroll chain, other crops, scrolls, scales or models, `overlap` other than 64, or that either surface is the right sheet. The fitted patches are near-flat 5 x 5 vertex patches through the seed (post hoc: plane normal z component about -0.12 for seeds 1 and 2 and about -0.02 for seed 3 in REF; most of the REF seed-to-seed distance is that change of tilt between seed 3 and seeds 1/2).
- A speed claim from this job (the wall times are not a benchmark).

Needs a human decision: (1) Sebastien, with the judge: does this narrower test meet condition 2 (#7, WC-20261014-01)? (2) The lasagna row wording in the AMD package after GATE16-REAL FAIL.

## Method of the check

`score.py` (sha256 2b2ab413...) re-run on 2026-10-14 on the nine `winding_0000.tifxyz` files in `R2000/score/` (CPU only): every field of every pair equals `fitsurface.json` exactly (max absolute difference 0) and the decision block is identical. Pair `d` = mean of the two directional medians verified for all 54 pairs. Group medians, max, B, M recomputed from the pair values; bootstrap of GATE16-REAL re-run with `rule.verdict` on the stored per-tile errors (identical to `gate16_real.json`). Scratch scripts and re-run output: the session scratchpad only; no result file of the job was changed.

## Files

`fitsurface.json`, `gate16_real.json`, `e2e_fast.json`, `e2e_ref.json`, `e2e_stock.json`, `coverage.json`, `R_used.txt`, `R2000/` (axis, shells, fits, `stage0_check.json`), `logs/` (`job.log`, `gpu.log`, `gpu_seconds.jsonl`, per-step logs), `stage0_attempt1_FAIL_exit78.json`. Prediction tensors are in git-ignored `work/cache/lasagna_fitsurface/`.
