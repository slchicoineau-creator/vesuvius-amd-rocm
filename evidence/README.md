# Evidence

Copies of the small result files and pre-registrations that `RESULTS.md` and `README.md` cite. All data behind them is public
Vesuvius Challenge data (PHerc0841, PHerc1447, PHerc0826, and for the lasagna rows PHerc0500P2 and PHerc0814). Files are unchanged copies, except `preregs/`, where
private path/scroll-name mentions were replaced by neutral words ("a private cache folder", "private-scroll", "nothing private"; 2026-10-14: in `2026-10-12_partner_fastpath_ruler.md` a private brief path became "a private team brief", in `2026-10-13_partner_lasagna_scope2.md` two mentions of a private scroll name became "no private scroll" / "private scrolls excluded").
The lasagna `*_RESULT.md` files carry notes dated 2026-10-14 marked post hoc (zero-fill caveat, `RESULTS.md` section 10).
Paths and issue numbers inside the preregs refer to the team's private working repository and are not shipped.
`lasagna_fitsurface_RESULT.md` is kept as written on 2026-10-14. Its "pending" and "needs a human decision" lines were
decided on 2026-10-15: no lasagna speed-up is offered (`RESULTS.md` section 10).

| File | Used for |
|---|---|
| `bench_convs.json` | per-layer speeds (RESULTS section 2) |
| `bf16exact.json` | per-layer noise per mode (section 3) |
| `fastcopy_summary.json`, `2026-10-08_fastcopy_gate_result.md` | copy-model gate and timings (section 3) |
| `v8in_gate.json` | v8-in gates G2 / G2s / G2c / G3 (section 4) |
| `canon_gate_summary.json`, `hecatefast_summary.json` | canonical and Hecate gates (sections 5, 6) |
| `nvidia/` | the RTX 5070 Ti printouts (README, "On NVIDIA") |
| villa PR #2113, "spiral-fitting: let AMD (ROCm) GPUs run the fitter without the CUDA-only packages": https://github.com/ScrollPrize/villa/pull/2113 | the spiral fitter patch (section 10) |
| villa PR #2114, lasagna CPU `.contiguous()` fix: https://github.com/ScrollPrize/villa/pull/2114 | the small lasagna fix (section 10) |
| villa PR #2115, lasagna ROCm backend: https://github.com/ScrollPrize/villa/pull/2115 | the lasagna patch for AMD GPUs (section 10) |
| `lasagna_gate16_RESULT.md`, `lasagna_gate16.json` | GATE16: `fp32_taps` on the 3 layers, FAIL (section 10) |
| `lasagna_gate16b_RESULT.md`, `lasagna_gate16b.json`, `lasagna_gate16b_DIAG.md` | GATE16b: `acc32_bias` PASS; the diagnosis that chose it (section 10) |
| `lasagna_diag_layer_bf16in.json`, `lasagna_diag_layer.json`, `lasagna_diag_e2e.json`, `lasagna_diag_chan.json` | raw numbers behind the per-layer and end-to-end diagnosis claims in section 10 |
| `lasagna_gate16c_RESULT.md`, `lasagna_gate16c.json` | GATE16c: 16 fresh tiles (section 10) |
| `lasagna_scope2_RESULT.md`, `lasagna_scope2_PHerc0500P2.json`, `lasagna_scope2_PHerc0814.json` | SCOPE2: two more scrolls' crops (section 10) |
| `lasagna_fitsurface_RESULT.md`, `lasagna_fitsurface.json`, `lasagna_gate16_real.json` | FITSURFACE: GATE16-REAL (16 tiles that read only real data), FAIL; fitted-surface test on a narrower design than first registered (section 10) |
| `fastpath_ruler_S2S3.md` | the error-ratio rule's checks on the copy model |
| `fit_compare_*.json` | spiral fit comparisons (section 10) |
| `preregs/` | the pre-registrations that fixed the bars before the runs (the fast-path ruler and its amendment, the four earlier gates, the lasagna gates, scope runs and fitted-surface test with its amendment 2) |

Not included: the GEMM table (README "Settings that matter", RESULTS section 1) has no result file; it is from notes.
