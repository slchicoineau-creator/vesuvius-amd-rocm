# FASTPATH-RULER S2 and S3 result (2026-10-13, partner, Beast)

Prereg `prereg/2026-10-12_partner_fastpath_ruler.md` as amended by `..._amend1.md` (c17bb1ac). Code committed before the run (014d4d4f). Raw: `s2.json`, `s2_raw.json`, `s3.json`, `s3_plan.json`, logs.
Card 1 via `gpu-share job set --card 1`; ROCBLAS_USE_HIPBLASLT=1 (S2, S3 bf16 arms), 0 (S3 REF fp32). The job was paused once by another claim (`neural-sheet9`) and restarted from zero; the reported run is the uninterrupted second one.
OC line: FAIL 5.7 % at R = 1.00, 81 % at 1.05, 99.5 % at 1.10 (`judge_20261013/judge_sim.json`, n = 20).

## S2 (copy model, 78 FASTCOPY crops; REF/taps/acc32_bias recomputed, stock errors from fc_STOCK.json)
Check: recomputed mean|REF| equals the stored value on every crop (max rel. deviation 0.0).
| mode | R78 (R-hat) | U | L | r | FAIL rate, 10,000 x 20 crops: observed / resc. 1.00 / 1.05 / 1.10 / 1.20 |
|---|---|---|---|---|---|
| taps | 0.833 | 0.856 | 0.809 | 0.935 | 0 % / 0.65 % / 54.4 % / 98.8 % / 100 % |
| acc32_bias | 0.849 | 0.885 | 0.811 | 0.893 | 0 % / 14.1 % / 84.9 % / 99.6 % / 100 % |
| (F32T, check vs S1) | 0.893 | 0.926 | 0.858 | 0.935 | 0 % / 5.4 % / 81.8 % / 99.4 % / 100 % |
**Kill line (A1): the rule passes the rescaled R78 = 1.10 case on 1.2 % (taps) and 0.4 % (acc32_bias) of draws; both <= 5 %. The ruler is NOT killed.**
Verdicts on all 78 crops: both modes PASS (they are better than stock bf16 on the copy model's output, not just equal). The judge's prediction of about 6 % FAIL at R = 1.00 holds for F32T (5.4 %) and acc32_bias is worse (14 %, lower r 0.89), taps better (0.65 %).

## S3 (Hecate 2.4 um, 20 disjoint 64x256x256 patches of R_sv24, seed 20261012)
Stage 0: R holds 250 disjoint patches; support fraction is 1.0 on every one, so "stratified by support" is degenerate and the 20 are a seeded random draw.
R-hat 0.9994, **U 1.020**, L 0.976, r 0.9965 -> **PASS** (U <= 1.10): fp32_taps bf16 is no worse than stock bf16 against the fp32 reference on this model.
A4 fields (support columns, logits / 8-bit sigmoid grey): stock mean|err| 0.1152 logit (2.26 % of mean|ref|), 1.640 grey (3.99 %), 25.08 % of pixels > 1 grey;
fast 0.1151 logit (2.26 %), 1.646 grey (4.01 %), 25.03 % > 1 grey. Share of support pixels with error > 1 logit about 6e-6 for both.
Note: this is the same pair HECATE-FAST-GATE failed on a fixed 1-grey fast-vs-stock criterion (98.96 % vs 99 %); under the ruler (error against fp32) it passes. PASS means "no worse than stock rounding", not a reading tolerance. Sebastien decides team use.
