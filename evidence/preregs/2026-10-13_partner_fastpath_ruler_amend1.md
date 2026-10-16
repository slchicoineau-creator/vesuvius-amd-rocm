# FASTPATH-RULER amendment 1 (2026-10-13, partner, Beast)

Parent prereg: `prereg/2026-10-12_partner_fastpath_ruler.md` (50cef304). Committed BEFORE S2/S3 run. Text adopted verbatim from the judge review `reviews/2026-10-13_partner_judge_fastpath_margin.md` §4 (3d0598d2).

**FASTPATH-RULER amendment 1 (2026-10-13, before S2/S3; reason: judge review `reviews/2026-10-13_partner_judge_fastpath_margin.md`).**
The rule (R ratio of sums, 2,000 resamples, one-sided 95 % U, PASS iff U <= 1.10, n = 20) and S1 are unchanged.
**A1. S2 positive control replaced.** The 1.44x / 1.19x figures are per-layer noise ratios (a116c24 §2), not output ratios (§3: taps
0.987, acc32_bias 0.629). S2 still runs REF, `taps` and `acc32_bias` on the 78 crops. For each mode it reports the 78-crop output ratio
R78 and the rule's FAIL rate over 10,000 20-crop draws (seed 20261012). The old kill line "S2 fails to reject `taps` on >= 95 %" is
withdrawn. The new known positive is each measured mode's own per-crop errors rescaled to R78 = 1.10 and to R78 = 1.20 (the residual
pattern is kept). **KILL if the rule passes the rescaled R78 = 1.10 case on > 5 % of draws** (false pass at the margin), for any mode.
Reported, not a gate: the FAIL rate at the rescaled R78 = 1.00 (judge's prediction from F32T: about 6 %) and at 1.05 (about 81 %).
**A2. S1c relabelled.** S1c's "x1.44 / x1.19" are true ratios of 1.286 / 1.063 vs stock. The 0.91 is a FAIL rate inside the margin, not
power at 1.19x.
**A3. Verdict wording and pairing.** Every S2/S3 verdict quotes: R-hat, U, the 5th-percentile bound L, the per-patch Pearson r of fast vs
stock error, and the OC line (FAIL 5.7 % at R = 1.00, 81 % at 1.05, 99.5 % at 1.10; `judge_20261013/judge_sim.json`).
U <= 1.10 = **PASS**. U > 1.10 with L > 1.00 = **FAIL (worse than stock)**. U > 1.10 with L <= 1.00 = **NOT SHOWN EQUIVALENT**, which
keeps the stock path. If r < 0.8, a non-PASS is NOT SHOWN EQUIVALENT whatever L is.
**A4. S3 adds reported fields (not gates):** stock and fast mean absolute error in output grey levels and as a share of mean |ref|, and the
share of support pixels whose fast-vs-REF and stock-vs-REF error exceeds 1 grey, so the reader can see the absolute size behind the ratio.
No GPU time is added. M and n are not changed.

Decision owner note: PASS means "no worse than stock rounding" (conservative), not a reading tolerance; Sebastien asked to accept this on #7.
