# Results

Every number below was read from a result file written by the run itself, unless marked "(from the team log)", which
means it comes from our written notes and we did not re-open the underlying file for this release. All measurements:
2x AMD Radeon AI PRO R9700 (gfx1201), ROCm 7.2.2, PyTorch 2.14.1+rocm7.2, one card per job, public scroll data. (Exception: the lasagna
measurements in section 10 ran in lasagna's own virtual environment, PyTorch 2.11.0+rocm7.2, Python 3.14.)

"Stock" = unmodified `nn.Conv3d`. "Fast" = `conv3d_fast.py`. "Mode" = `enable(mode=...)`; the default is `taps`.

## 1. Matrix multiply speed vs `ROCBLAS_USE_HIPBLASLT` (from the team log; NOT re-measured for this release)

No benchmark script or output file for this table was kept, so it is not backed by a file in this package. Treat it as an
observation from our notes (2026-10-01) and re-measure on your card before relying on it; the one-line check is
`torch.randn(8192, 8192, device="cuda", dtype=...) @ ...` timed with the variable set to 0 and to 1.

8192 x 8192 GEMM, one card.

| setting | fp16 | bf16 | fp32 |
|---|---|---|---|
| `=0` | 13 TFLOPS | 11 TFLOPS | 13 TFLOPS |
| `=1` | 116 TFLOPS | 117 TFLOPS | 1 TFLOPS |

## 2. Per-layer convolution speed (3-D encoder of the v8-in model, fp16, batch of 16 tiles of 96x256x256)

Source: `evidence/bench_convs.json`. Representative rows (stock ms -> fast ms):

| layer | stock | fast | speed-up |
|---|---|---|---|
| 3x3x3, 64->64 channels, stride 1 (3 such layers) | 246.0 ms (5.7 TFLOPS) | 34.9 ms (39.8 TFLOPS) | 7.04x |
| 3x3x3, 128->128, stride 2 | 65.2 ms (10.7 TFLOPS) | 38.6 ms (18.0 TFLOPS) | 1.69x |
| 3x3x3, 256->256, stride 2 | 17.0 ms | 15.4 ms | 1.10x |
| 3x3x3, 512->512, stride 2 | 6.1 ms | 6.7 ms | **0.90x** (slower; the policy still rewrites it) |
| 1x1x1, 64->64 | 3.8 ms | 9.5 ms | 0.40x (kept stock) |
| 1x1x1, 64->256 | 8.6 ms | 22.1 ms | 0.39x (kept stock) |
| 7x7x7 single-channel stem, stride (1,2,2) | 199.1 ms | 210.8 ms | 0.94x (kept stock) |

The three stride-2 3x3x3 layers together (88.3 ms -> 60.7 ms) give about 1.45x; per layer 0.90-1.69x.

This is why the patch only touches 3x3x3 layers with more than one input channel.

## 3. Neural-tracing Copy model (bf16, 8x128x384x384 crop)

Sources: the mode comparison on one real crop (full forward + 71 per-layer measurements + 8 sub-crops), and the
equivalence/speed run over 78 crops and a whole copy job on one strip of PHerc0841 (public scroll data).

| | stock | `taps` | `fp32_taps` |
|---|---|---|---|
| one forward, seconds | 75.6 | 1.70 | **0.87** |
| peak reserved memory, GB | 10.0 | 28.6 | 23.0 |

(In the 78-crop run: stock 75.97 s -> `fp32_taps` 0.86 s per forward. The 3.04 s figure of that run is the reference
arm "fast fp32" (`conv3d_fast` default `taps`, run in **fp32**, autocast off, `ROCBLAS_USE_HIPBLASLT=0`), not bf16 `taps`.)

**Whole copy job** (one strip, 78 crops, no test-time augmentation, `--bbox-overlap 0.5`): stock bf16 6528 s, `fp32_taps` bf16
354.2 s -> **18.4x** faster than stock. The third arm of that run, the reference, is `conv3d_fast` default `taps` in **fp32**
(autocast off, `ROCBLAS_USE_HIPBLASLT=0`): 495.5 s. It is not a bf16 `taps` result.
Peak reserved memory: 29.7 GB stock bf16, 20.3 GB `fp32_taps` bf16, 31.1 GB for the fp32 reference arm.
Stock-path speed with test-time augmentation was not measured (the VC3D Copy default is mirror TTA).
Source: `evidence/2026-10-08_fastcopy_gate_result.md`, `evidence/fastcopy_summary.json`.

**Equivalence, 78 crops, `fp32_taps` vs stock bf16** (error against what was measured, pooled over crops): p99 1.004
vs 1.242 for stock (ratio 0.81), mean 0.190 vs 0.213 (ratio 0.89). Per-crop median ratio 0.94 (range 0.44-1.13);
3 of 78 crops (3.8 %) above the bar. The separate downstream check (7618 pairs) showed D = 0 for both stock and `fp32_taps`; that check was uninformative
(neither bf16 copy changed any step decision), so the PASS rests on the 78-crop noise comparison above.

**Which bf16 mode?** Noise added by each mode, relative to stock bf16's own noise (reference: fp32 maths on the
bf16-rounded input, weights and bias), over the 71 fast-path layers, each fed its real input:

| mode | median rms ratio | range | sum of layer times |
|---|---|---|---|
| stock | 1.00 | 1.00-1.00 | 75.3 s |
| `taps` | 1.44 | 1.27-1.54 | 1.23 s |
| `acc32_bias` | 1.19 | 0.83-1.34 | 1.33 s |
| `taps_as_channels` | 1.00 | 1.00-1.00 | 1.12 s |
| `fp32_taps` | 0.96 | 0.68-1.00 | 0.71 s |

On the whole real crop all modes passed a "within 1.1x of stock's distance to fp32" bar (`fp32_taps`: 0.535x at p99,
0.621x mean). On the 8 sub-crops of the same crop the whole-model ratio was noisy for every mode (`fp32_taps`
0.48-1.65 at p99, passing 6 of 8; `taps` 0.71-1.20, passing 4 of 8; `taps_as_channels` 0.96-1.15, passing 6 of 8).
The reading we take from this: per-layer, `fp32_taps` never added noise (71 of 71 layers at or below stock);
whole-model ratios on single crops are mostly luck, so use several crops (the 78-crop run above).

In **fp32**, fast `taps` against stock fp32 on the whole crop: p99 abs diff 0.0002, mean 4e-5 (mean |activation| 42.4;
fp32 on the fast path took ~3 s vs 679 s stock).

Memory caveat: the 8-variant test-time-augmentation call peaked at 29.1 GB reserved with `fp32_taps` (28.9 GB in the
78-crop run) -- over a 28 GB budget on a 32 GB card shared with other work.

## 4. v8-in 9 µm ink model (public PHerc1447 data, one whole winding, inference)

Baseline 9.0 tiles/s per card (63,558 tiles). Quality bar: at least 99.5 % of pixels within one grey level of the stock
output, and a change in AUC and AP of at most 0.002.

| configuration | wall time for the winding | speed-up vs stock | within 1 grey level |
|---|---|---|---|
| stock | about 7060 s (9.0 tiles/s) | 1x | (reference) |
| `conv3d_fast` | 3493 s | 2.02x | 100 % (passed) |
| + single-channel stem rewrite | 2840 s | 2.49x | 100 % (passed) |
| + stem + `torch.compile` | 1689 s | 4.18x | 100 % (passed) |

Training (Lightning fine-tune recipe, `conv3d_fast` default mode): **1.42x** per epoch; final-model correlation 0.973
against a stock run of the same seed, versus 0.965 between stock runs of different seeds.
(Speed-up and training figures: `evidence/v8in_gate.json`.)

## 5. 2.4 µm canonical ink model (ResNet3D-152 + 3-D decoder), fp16 autocast, `fp32_taps`

Reference = stock fp32; 64 tiles for the tile check, a whole render for the output check.

- 56 fast-path layers; all gates passed (tile errors, output within 1 grey level, correlation with reference map 0.95954 vs 0.95954 for the reference run).
- Output difference vs stock fp16: 100 % of support within 1 grey level; max difference 2 grey levels.
- Speed: per tile 0.1625 s -> 0.0816 s (1.99x); whole render at batch 4: 226.8 s -> 109.6 s (2.07x); at batch 1 fast 90.3 s.
- Peak reserved memory 7.2 GB (batch 4) / 1.7 GB (batch 1).

## 6. Hecate

`fp32_taps`, bf16, vs stock bf16. The overall verdict was **FAIL**.

| scale | per-patch noise check | output within 1 grey level | render speed-up | result |
|---|---|---|---|---|
| 9.6 µm | passed | 99.998 % (max 2 grey levels) | 1.66x | **passed** |
| 2.4 µm | passed (p99 ratio 0.99) | 99.31 % of support, 98.96 % of non-zero pixels (max 7 grey levels) -- **bar was 99 % on both; non-zero pixels missed it by 0.04 points** | 1.92x (1.97x per patch) | **failed** |

**Two different checks on the same pair (fp32_taps bf16 vs stock bf16, Hecate 2.4 µm), both reported, neither upgrades the other.**
(1) The first gate (`prereg/2026-10-09_partner_hecate_fast_gate.md`) asked fast vs *stock* to agree within one grey level on at least 99 % of both support and
non-zero pixels; it **failed** (98.96 % of non-zero pixels). That verdict stands as written.
(2) A later pre-registered ruler (`prereg/2026-10-12_partner_fastpath_ruler.md` with `..._amend1.md` (copies in `evidence/preregs/`); result
`evidence/fastpath_ruler_S2S3.md`) asked a different question: is the fast path's error against the **fp32 reference** no larger than stock's? On 20 disjoint
64x256x256 patches of the same public PHerc0841 region: ratio of summed errors R-hat 0.9994, one-sided 95 % upper bound U 1.020 (bar 1.10), lower bound L 0.976,
per-patch r 0.9965 -> **PASS**. Mean error against fp32: stock 1.640 grey, fast 1.646 grey. This means "no worse than stock's own bf16 rounding", **not** a reading
tolerance and not agreement with stock. The two checks differ because stock itself is 1.6 grey from fp32 on average, so fast and stock can differ from each other by
more than one grey level while being equally far from fp32. Whether to use it is your decision; the package does not offer it without your own check.

Plainly: at 2.4 µm Hecate the fast path produced slightly different output than stock (still correlation 0.99999) but
narrowly missed our pre-set 99 % bar (this check used a looser bar than the 99.5 % used for v8-in), so we do **not** recommend it there without your own check. Both the Hecate and the
canonical-model checks ran on public PHerc0841 data (a region along segment w00); the canonical model is
`scrollprize/ink_canonical_2um` on Hugging Face.

## 7. Reproduction on AMD hardware (public PHerc0841 data; from the team log)

Two ink-prediction runs reproduced from NVIDIA reference numbers, one card, about 3.5 minutes per 2.4 µm run:

| run | AMD R9700 | NVIDIA reference |
|---|---|---|
| Hecate on the 2.4 µm scan | 0.837 | 0.832 |
| Hecate on the 9.6 µm scan | 0.745 | 0.746 |

(Correlation with the known letters.)

## 8. Other measurements (from the team log)

- 3-D convolutions reach about 9 TFLOPS in stock MIOpen on gfx1201 (3-9 TFLOPS depending on layer); the 3-D
  convolutions of one big model ran at 2.8-87x faster after decomposition, layer-dependent.
- v8-in inference on one R9700: about 8.3 tiles/s per card in fp16 (96x256x256 tiles), 12 GB peak.

## 9. Speed and VRAM table, v1 (one forward per model, stock vs gated fast path; measured 2026-10-11)

Method (pre-registered, `prereg/2026-10-11_partner_rocm_ports.md`): a fresh process per row, 1 warm-up forward, then the
**median of 3 timed forwards**; memory = `torch.cuda.max_memory_allocated` / `max_memory_reserved` over those forwards, and
the peak of `rocm-smi --showmeminfo vram` sampled every ~0.2 s over the whole process (this includes the ~0.7 GB the card
already used at idle, the CUDA/HIP context, and the loaded weights). One R9700 (`HIP_VISIBLE_DEVICES=0`; the card's power cap was not re-read for this run), ROCm 7.2.2,
torch 2.14.1+rocm7.2, `ROCBLAS_USE_HIPBLASLT=1`, `PYTORCH_ALLOC_CONF=expandable_segments:False`, kernels already built (warm
MIOpen cache). Only paths that passed their equivalence gate are listed as "fast". Raw per-row files: `measurements/*.json`
(they hold all 3 times, the input shape and the conv path).

| model (input, precision) | path | median s per forward | max allocated GB | max reserved GB | rocm-smi peak GB | speed-up |
|---|---|---|---|---|---|---|
| Copy model (1 x 8 x 128 x 384 x 384 crop, bf16, no TTA) | stock | 75.56 | 7.37 | 10.04 | 11.08 | 1x |
| | fast, `fp32_taps` | 0.863 | 18.73 | 20.27 | 21.32 | 87.5x |
| v8-in 9 um ink (16 tiles of 64 x 64 x 24, fp16) | stock | 1.7245 (9.3 tiles/s) | 11.20 | 13.71 | 14.79 | 1x |
| | fast, `conv3d_fast+stem+compile` | 0.4857 (32.9 tiles/s) | 7.84 | 10.67 | 14.68 | 3.55x |
| Canonical 2.4 um ink (1 tile 62 x 256 x 256, fp16) | stock | 0.1611 | 3.55 | 4.84 | 5.91 | 1x |
| | fast, `fp32_taps` | 0.0817 | 1.44 | 1.58 | 2.66 | 1.97x |
| Canonical 2.4 um ink (4 tiles, fp16) | stock | 0.6455 | 5.36 | 6.18 | 7.26 | 1x |
| | fast, `fp32_taps` | 0.3505 | 5.49 | 6.73 | 7.82 | 1.84x |
| Hecate 9.6 um (1 patch 16 x 64 x 64, bf16) | stock | 0.0430 | 1.27 | 1.66 | 2.75 | 1x |
| | fast, `fp32_taps` | 0.0258 | 0.73 | 0.93 | 2.03 | 1.67x |
| Hecate 2.4 um (1 patch 64 x 256 x 256, bf16) | stock | 0.1595 | 3.54 | 4.78 | 5.88 | 1x |
| | fast | not listed: failed its gate (section 6) | | | | |

Reading it:
- The copy model's single-forward speed-up (87x) is larger than its whole-job speed-up (18.4x, section 3) because a job also
  spends time reading the volume and merging; the single forward is the part the patch changes. Its stock peak here is the
  forward alone; the whole stock job peaked at 29.7 GB reserved (section 3). With the 8-variant test-time-augmentation call
  `fp32_taps` took 8.87 s and 28.86 GB reserved (22.10 GB allocated; one measurement from the FASTCOPY-GATE run, not
  repeated here).
- The fast path uses more memory for the copy model (fp32 copies of the layer inputs) and about the same for four canonical
  tiles; it uses less for the other rows.
- v8-in "compile" includes `torch.compile`; the median is over 3 timed forwards after the compile warm-up (the 3 values were
  0.41, 0.49, 0.51 s, so the figure is rough). On the whole 63,558-tile winding the gate measured 4.18x and 37.6 tiles/s
  (section 4); the table figure (3.55x, 32.9 tiles/s) is for one batch of 16 on a warm card and is not a replacement for it.
- These are forward-pass timings, not whole renders. Whole renders from the gate runs (Hecate 9.6: 44.8 s -> 27.0 s;
  canonical at batch 4: 226.8 s -> 109.6 s) are in sections 5 and 6.
- The v8-in and canonical rows use PHerc1447 (v8-in) and PHerc0841 region R (canonical, Hecate) inputs that our gates
  used; the copy-model input is a crop from a PHerc0841 strip at 9.366 um.

## 10. Organizer tools on ROCm: lasagna and the spiral fitter (measured 2026-10-11)

Upstream villa `main` d63d3e968; one R9700 (`HIP_VISIBLE_DEVICES=1`), ROCm 7.2.2. Bars pre-registered in
`prereg/2026-10-11_partner_rocm_ports.md` (reference = CPU torch on the same machine). Full report, scripts, logs and the
two patches (now villa PRs, linked in `evidence/README.md`: #2113 for the spiral fitter, #2114 and #2115 for lasagna), the fit comparisons `evidence/fit_compare_*.json`
and the pre-registrations in `evidence/preregs/`.

**Lasagna** (`predict3d`, public PHerc0841 9.366 µm crop of 256³, model `model_current.pt` from HF `scrollprize/lasagna`):

| check | result |
|---|---|
| install picks a ROCm torch (patched `bootstrap_venv.py`, `--backend rocm` or auto when only ROCm tools are found) | yes, torch 2.11.0+rocm7.2 |
| run completes on the R9700, finite, HIP device in the log | yes |
| default path (bf16 autocast, hard-coded upstream): CPU vs R9700, relative L2, bar 1e-2 | **fails** on 3 of 4 channels: cos 5.8e-3, grad_mag 1.42e-2, ny 2.39e-2, nx 5.45e-2 |
| same forward in fp32 (diagnostic wrapper): CPU vs R9700, bar 1e-4 | within the bar: cos 6.3e-5, grad_mag 8.5e-5, nx 1.9e-5, ny 3.6e-5 (max 1 uint8 level) |
| bf16 vs fp32 on the CPU alone | cos 4.7e-3, grad_mag 1.66e-2, nx 5.86e-2, ny 2.45e-2 |
| inference time, 125 tiles (stock 3-D convolutions) | bf16: CPU 8.3 min, R9700 40.9 min; fp32: CPU 12.0 min, R9700 45.6 min |

Reading it: the R9700 computes the same thing as the CPU (fp32 row). The default bf16 path differs between devices by about
as much as bf16 differs from fp32 on one device, so the registered 1e-2 bar is tighter than the model's own bf16 rounding;
we report it as failed, not as a ROCm defect. Most of the normal-channel error is the uint8 encoding flipping near-horizontal
normals to the other hemisphere (same line in space, see KNOWN_ISSUES). On the stock 3-D convolution path the R9700 is
about 4x slower than a 32-thread Threadripper CPU.

**Lasagna fast path (measured 2026-10-13; three full-resolution layers `shared_encoder.stages.0.blocks.0.conv1.conv`, `...conv2.conv`,
`shared_decoder.stages.5.convs.0.conv` on `conv3d_fast`, everything else stock; 16 of the 125 tiles, bf16, rule = ratio of summed error against the fp32
model, pass iff the one-sided 95 % upper bound <= 1.10).** Evidence: `evidence/lasagna_gate16_RESULT.md` (+ `lasagna_gate16.json`), `evidence/lasagna_gate16b_RESULT.md` (+ `lasagna_gate16b.json`),
`evidence/lasagna_gate16b_DIAG.md` (+ `lasagna_diag_layer_bf16in.json`, `lasagna_diag_layer.json`, `lasagna_diag_e2e.json`, `lasagna_diag_chan.json`), `evidence/lasagna_gate16c_RESULT.md` (+ `lasagna_gate16c.json`), `evidence/lasagna_scope2_RESULT.md` (+ `lasagna_scope2_PHerc0500P2.json`, `lasagna_scope2_PHerc0814.json`); preregistrations `evidence/preregs/2026-10-13_partner_lasagna_gate16.md`, `..._gate16b.md`, `..._gate16c_confirm.md`, `..._lasagna_scope2.md`, `..._lasagna_scope2_amend1.md`.

| mode on the 3 layers | R-hat | upper bound U | lower bound L | per-tile r | verdict | whole 125-tile predict3d | speed-up (one run each; measured only, NOT offered: the last row failed) |
|---|---|---|---|---|---|---|---|
| stock | 1 | - | - | - | - | 2098.8 s | 1x |
| `fp32_taps` (GATE16) | 2.06 | 2.17 | 1.90 | 0.994 | **FAIL (worse than stock)** | 203.9 s | 10.3x measured, **not usable** |
| `acc32_bias`, first gate (GATE16b, 16 tiles) | 0.43 | 0.54 | 0.36 | 0.990 | **PASS** | 122.1 s | 17.2x (stock wall reused from the first run) |
| `acc32_bias`, confirmation (GATE16c, 16 fresh tiles, everything re-run) | 0.41 | 0.52 | 0.35 | 0.994 | **PASS** | 128.8 s (stock 2105.5 s) | **16.3x** |
| `acc32_bias`, other scroll: PHerc0500P2 crop (SCOPE2, 16 of 100 tiles, 97.5 % bound) | 0.43 | 0.57 | 0.35 | 0.985 | **PASS** | 85.8 s (stock 1668.3 s, 100 tiles) | 19.4x |
| `acc32_bias`, other scroll: PHerc0814 crop (SCOPE2, 16 of 125 tiles, 97.5 % bound) | 0.92 | 0.96 | 0.88 | 0.762 | **PASS** (per-tile r below 0.8 only matters for a non-PASS) | 114.8 s (stock 2094.9 s) | 18.2x |
| `acc32_bias`, content-only re-check on the PHerc0841 crop (GATE16-REAL, LASAGNA-FITSURFACE amendment 1, 2026-10-14; 16 tiles drawn from the 93 unused, every tile reads real data; `evidence/lasagna_gate16_real.json`; decided 2026-10-15: speed-up none offered (author's note; review J1)) | 1.105 | 1.113 | 1.096 | 0.978 | **FAIL (worse than stock)** | not re-timed | not claimed |

Raw files: `evidence/lasagna_gate16.json`, `lasagna_gate16b.json`, `lasagna_gate16c.json`, `lasagna_scope2_*.json`. Operating characteristic of the rule at n = 16 (copy-model residuals, 2,000 draws; reported, not a gate; not measured for the 97.5 % bound): FAIL in 11.7 % of draws at a true ratio of 1.00, 84.9 % at 1.05, 99.2 % at 1.10.
Speed: each factor is one run. The 16.3-19.4x spread is mostly fixed per-run overhead inside a 1.4-2.2 minute fast pass (fast wall per call 0.86-1.03 s against a 0.686-0.690 s forward); it is not a scroll effect, and the 0500P2 figure is per 100 calls. The per-tile forward factor is 24.0-24.1x on all three crops. These are measurements, not an offered speed-up; if 16.3x (PHerc0841, everything re-run) is quoted, quote the failed GATE16-REAL row with it.

Reading it: decided 2026-10-15: speed-up none offered for lasagna (author's note; review J1). The first gates were passed only by `acc32_bias`, and that mode failed the real-data re-check in the last row. The failure of `fp32_taps` is not arithmetic quality: per layer it is the most accurate mode (it equals the correctly
rounded exact result; stock's naive kernel is not correctly rounded, 62-78 % of outputs match), yet in the whole model one layer (the first conv of the first encoder stage)
makes a coherent offset downstream that doubles the output error. The mechanism is not isolated; per-layer accuracy did not predict whole-model error here. `acc32_bias`
was chosen for the second gate (GATE16b) after a 3-tile diagnosis, and those 3 tiles were among its 16, so that PASS alone carried a selection caveat (7 of 16 tiles 1.005-1.13 of stock's error, 8 at 0.29-0.47, 1 at 0.81; read the zero-fill paragraph below). The caveat is lifted by GATE16c: the mode frozen, 16 fresh tiles (seed 20261013, none used before), fresh fp32 reference, stock and fast
both re-run: PASS again (R-hat 0.41, U 0.52, 5 of 16 tiles above 1.00 (1.003-1.134), 9 at 0.29-0.47). The scope was then widened by SCOPE2 (prereg `2026-10-13_partner_lasagna_scope2.md` + amendment 1): two more public 9.362 um (113 keV, 1.2 m) crops of 256^3 from other scrolls, PHerc0500P2 and PHerc0814, chosen by a written rule
before any output, mode frozen, fresh fp32 reference per crop, one-sided 97.5 % bound (two crops): both PASS. On 0814 the gain is small (R-hat 0.92, tiles 0.74-1.09 of stock error; on its tiles with at least 69 % real data R-hat 0.96, post hoc), on 0500P2 it is large (R-hat 0.43, but see the zero-fill caveat; the crop is partly masked and 5 of the 16 tiles are identical in content; with the five collapsed to one the bound is still 0.79; the tile ratios otherwise run 0.43-1.10). So the measured scope is three crops of three scrolls (0841, 0500P2, 0814), one model, one scan class (9.36 um, 113 keV, 1.2 m); not shown on
2.4 um or 7.9 um scans, other models, or the surfaces fitted downstream. Measured at predict3d's default overlap 64 (other overlaps not shown). Not a reading tolerance.

**Zero-fill caveat (post hoc, added after an independent review on 2026-10-13; the registered verdicts above stand unchanged).** The gate runs used `--no-download` on a local
copy of the volume holding only the chunks of the 256^3 crop. predict3d places 192^3 tiles on a global 128-voxel lattice around the crop snapped to 128 (5 positions per axis, 125 tiles), each reading 16 voxels
further out, so tiles that reach past the downloaded chunks read **zeros** there (0841: 64 chunks of 128^3 present; 0814: 125). The large "much better than stock" gain in the pooled ratio comes from
those tiles (ratio 0.29-0.47, with little or no real data in the input), not from scroll data. Recomputed per tile from the stored errors and the chunk presence (post hoc, same bootstrap, the
reviewer's numbers): on tiles with at least 69 % real input data, **PHerc0841 GATE16b+GATE16c pooled, n = 13: R-hat 1.067, U 1.092, L 1.041** (about 7 % above stock's error, just inside the 1.10 margin);
GATE16c alone, n = 6: R-hat 1.072, U 1.121, L 1.025; the 5 tiles with at least 90 % real data: R-hat 1.106. **PHerc0814, n = 12 such tiles: R-hat 0.956, U 0.987** (about 4 % below stock's).
The same five identical PHerc0500P2 tiles (ratio 0.29) are the same near-empty input. So the supportable claim is: on these crops the 3-layer `acc32_bias` fast path stayed within the 1.10 margin of stock bf16's
error on the registered sampling; on real PHerc0841 data alone it was slightly worse than stock (about 7 %), on PHerc0814 slightly better. The content-only rerun now exists (next paragraph) and supersedes this sentence for real PHerc0841 data.

**Content-only re-check and fitted-surface test (LASAGNA-FITSURFACE, run 2026-10-14; status: decided 2026-10-15: speed-up none offered (author's note; review J1)).** Preregistration `evidence/preregs/2026-10-13_partner_lasagna_fitsurface.md` (083b308e) with amendment 1 (0ae5cac5, in the same file: predictions on a margin copy of the volume so that all 125 tiles read real data; coverage check: minimum real share 1.0) and amendment 2 (6986cfe4, `evidence/preregs/2026-10-14_partner_lasagna_fitsurface_amend2.md`). Result file: `evidence/lasagna_fitsurface_RESULT.md` (+ `evidence/lasagna_fitsurface.json`, `evidence/lasagna_gate16_real.json`).
- **GATE16-REAL, FAIL (worse than stock).** The registered rule, unchanged, on 16 tiles of the PHerc0841 crop that read only real data, fp32 reference, stock bf16 and the 3-layer `acc32_bias` fast path all re-run on the margin copy: R-hat 1.105, U 1.113, L 1.096, per-tile r 0.978; all 16 tiles above 1.00 (1.075 to 1.142), 8 of 16 above 1.10. In absolute size both errors are small (mean absolute raw error 0.00221 fast against 0.00200 stock, 0.39 % against 0.35 % of the mean absolute reference output; about 0.13 against 0.11 grey levels of 255). So on real data the fast path is **not shown equivalent to stock**: it has about 10 % more error against the fp32 model than stock bf16 has. The earlier PASS rows above stand as registered, but they were pooled with tiles that read zero fill (section above); on the fully real tiles of GATE16-REAL the same rule gives FAIL. The lasagna fast path is therefore not offered: decided 2026-10-15: speed-up none offered (author's note; review J1). `RUNBOOK.md` section 10 keeps the recipe for experiments only.
- **Fitted-surface test, PASS as registered, on a narrower test than first registered.** Median of the 9 REF-vs-FAST surface distances M = 2.05 vx (19 um) against the bar B = 6.76 vx (63 um, the largest REF-vs-REF seed-to-seed distance); M <= B. Scope limit, stated plainly: this compares the shipped patch fit on FAST vs REF predictions from a shared analytic cylinder shell; the registered cylinder-shell step could not run on a crop-sized prediction box (amendment 2). It says nothing about the cylinder-shell step, the real umbilicus or the whole-scroll chain, and it is one crop. It does not separate the fast path from stock bf16 (stock also passes: REF-vs-STOCK median 2.34 vx), the bar is loose (0.66 of a sheet width, set by how far the fit moves when the seed point moves 16 vx), and the strict reading (largest REF-vs-FAST pair no larger than B) is not met (6.83 vx against 6.76 vx). With the same seed point the FAST and REF surfaces are 0.16 to 0.89 vx apart (post hoc view).
- **Open.** Whether this narrower test meets Sebastien's condition 2 (fit one surface from the FAST output and one from the fp32 REF on the same crop and show the two match) is OPEN: issue #7, request WC-20261014-01. Until he answers, numbers that go into a submission are re-run on stock.

**Spiral fitter** (`spiral-fitting/fit_spiral.py`, public PHerc0826 volume 20250821151701, slices 10208-11208, published
tracks, 4,000 optimizer steps):

| check | result |
|---|---|
| installs on Linux + ROCm (patch moves `cupy-cuda12x`, `cucim-cu12`, `brook-cu12` to a `cuda` group that `uv sync` still installs by default) | yes, torch 2.12.1+rocm7.2, native modules build |
| 3 seeds complete on the R9700, finite | yes, 6.5 optimizer steps/s, about 11 min per fit |
| seed-to-seed on the R9700 (`fit_compare.py` point-to-surface median) | 0.56, 0.72, 0.80 voxels; turn offset +0 on 95-96 % of points; outer sectors agree |
| ROCm vs reference | measured against an NVIDIA run of the same three seeds (the fitter has no CPU path, so the registered CPU reference was impossible and the reference moved to CUDA; a deviation from the pre-registration). See the paragraph below the table |

**Spiral fitter, ROCm vs the NVIDIA reference** (reference: RTX 5070 Ti, villa `main` d63d3e968 unpatched, same tracks and
config, seeds 1/2/3, 4,000 optimizer steps each). Files: `evidence/fit_compare_cuda_vs_rocm.json`,
`evidence/fit_compare_cuda_seeds.json`, `evidence/fit_compare_rocm_seeds.json`.

Deviations from the pre-registration, stated up front: (1) the registered reference was CPU torch; the fitter has no CPU
path, so the reference is the NVIDIA run; (2) 4,000 optimizer steps per fit instead of the 30,000 of the earlier pilot recipe
(about 11 min per fit on the R9700 instead of about 85).

The bar, quoted verbatim from `prereg/2026-10-11_partner_rocm_ports.md` (evidence copy in `evidence/preregs/`):

> PASS if the median ROCm-vs-reference distance (`fit_compare.py` point-to-surface median, and
> per-sector turn agreement where it applies) is <= the largest same-device seed-to-seed distance on the reference
> device. Turn assignment must agree on every sector the reference seeds agree on among themselves.

The reference device is the NVIDIA card, whose same-device seed-to-seed medians are 0.60 / 0.57 / 0.63 voxels, so the
registered bar is **0.63 voxels**. (An earlier internal report judged the result against 0.80 voxels, the ROCm seed spread.
That was not the registered bar and is not used here.)

Point-to-surface median (voxels), CUDA fit (rows) vs ROCm fit (columns):

| | ROCm s1 | ROCm s2 | ROCm s3 |
|---|---|---|---|
| CUDA s1 | 0.24 | 0.71 | 0.59 |
| CUDA s2 | 0.67 | 0.26 | 0.65 |
| CUDA s3 | 0.56 | 0.70 | 0.20 |

**Verdict against the registered bar: NOT MET (no pass claimed).** The prereg does not fix how the nine pair distances are pooled; the check passes only if they are pooled as the median of all nine (0.59), and fails as the maximum or as the different-seed median. The largest cross-device distance (0.71) is above 0.63, and so
is the median of the six different-seed pairs (0.66). For description only: the median over all nine pairs is 0.59; the three
same-seed pairs are 0.20-0.26 voxels; turn assignment agrees on every sector in all nine pairs (outer sector range 0; turn
offset +0 on 95.1-97.2 % of points). The prereg does not say how the nine distances are pooled into "the median", and we do
not choose a pooling after seeing the numbers; the bar was not met under the strict reading and we do not claim a pass. The
different-seed cross-device distances (0.56-0.71) sit around the NVIDIA device's own seed spread (0.57-0.63) and the ROCm
device's (0.56-0.80), so no device effect is visible above seed noise, but that is a description, not a pass.
