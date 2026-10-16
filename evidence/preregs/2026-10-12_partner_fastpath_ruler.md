# Pre-registration: FASTPATH-RULER. One frozen decision rule for "is a conv3d_fast mode equivalent to stock?", with its operating characteristic measured before it is used (partner, Beast)

- **Date:** 2026-10-12. Committed and pushed before any step below is run. **Owner:** partner (lane B). Source: Opus vet
  a private team brief (pick 1). **Decision owner** for team use: Sebastien.
- **Why:** the fast-path gates so far (FASTCOPY-GATE, 1-grey checks) used one-off thresholds whose false-fail and power were never
  measured; a one-crop ratio is mostly luck (a116c24). This fixes one rule and measures how often it is wrong.
- **Truth rule:** no w00, no humans, no labels, no keys. Every error is vs the fp32 REF on the same input.

## Rule (frozen)

- Unit = a patch (a crop). E_c(X) = summed absolute error of mode X vs fp32 REF over the patch's support.
- R = (sum over patches of E_c(fast)) / (sum over patches of E_c(stock)) (pooled ratio of sums).
- Bootstrap: resample patches with replacement, 2,000 resamples, seed 20261012; U = 95th percentile of R (one-sided 95 % upper bound).
- **PASS iff U <= M = 1.10 over n = 20 patches.** M is FASTCOPY-GATE's existing bar (no new tuning). M and n are not changed after any result
  below; never tune M or n to make Hecate 2.4 um fail or pass.

## Steps

- **S1 (CPU, run now).** Data: `work/ink/fastcopy/copies/fc_STOCK.json` (`g1.crops`, 78 crops, same-size 128x384x384 crops, so E_c is proportional
  to the stored per-crop mean abs error vs REF; the common factor cancels in R). Arms: STOCK (`stock.mean`), F32T (`f32t.mean`).
  - S1a, equivalent arm: 10,000 draws of 20 of the 78 crops without replacement; apply the rule to R = F32T/STOCK. Observed F32T/STOCK pooled
    ratio is about 0.89 (below M), so any FAIL is a false fail. **Calibration PASS iff false-fail rate <= 5 %.**
  - S1b, null (true ratio 1, reported, not a gate): same draws, numerator = STOCK error of the crop, denominator = STOCK error of a different crop
    drawn from the same 20 (a random derangement). This breaks pairing and is a stricter variance test; reported as false-fail rate.
  - S1c, synthetic power (reported): F32T errors times 1.44 and times 1.19 (the `taps` and `acc32_bias` ratios from the vet); fraction of draws that FAIL.
  - If S1a > 5 %: report the smallest n in {20, 25, 30, 40, 50, 60} giving <= 5 %, and use it for S3.
  - Caveat recorded in advance: the 78 crops overlap 0.5 along z, so they are not independent; bootstrap over crops is therefore optimistic.
- **S2 (GPU, NOT RUN).** Same 78 crops: REF fp32, `taps`, `acc32_bias`; stock sums already stored. Rule must FAIL `taps` (1.44x) on >= 95 % of
  20-crop draws; report power at `acc32_bias` (1.19x). If that power is < 80 %, write "1.19x not resolvable at n" before S3.
- **S3 (GPU, NOT RUN).** Hecate 2.4 um, 20 seeded disjoint 64x256x256 patches in region R stratified by support fraction (Stage 0: check R holds 20
  disjoint patches with real support, else block bootstrap). fp32 (hipBLASLt 0), stock bf16 and `fp32_taps`; apply the frozen rule. The old 1-grey
  FAIL stays filed; this verdict is a new result either way. Via gpu-share claim, no run outside it.

## Kill rule

- S1a false-fail > 5 % at every n up to 60: the rule (with M = 1.10) is KILLED as a decision tool on this data; S2/S3 are not run with it and
  nothing is claimed about any fast path from it.
- S2 fails to reject `taps` on >= 95 % of draws: KILLED (cannot even catch a 1.44x error).
- Otherwise the rule stands for S3 with the measured operating characteristic quoted next to its verdict.
- Not claimed: anything about other models, other scrolls, or which copy is on the right sheet. Kill date: 2026-10-26, not run by then = closed as not run.

## Outputs

`work/ink/fastpath_ruler/{ruler_s1.py,ruler_s1.json}`.
