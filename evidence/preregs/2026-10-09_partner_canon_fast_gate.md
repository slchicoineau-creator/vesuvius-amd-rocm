# Pre-registration: CANON-FAST-GATE: is conv3d_fast `fp32_taps` under fp16 autocast equivalent to stock fp16 for the canonical 2.4 µm ink reader? (partner, [internal reference removed])

- **Date:** filed 2026-10-09 (evening CDT) by Chris's main session, together with
  `canon_gate.py` + `job.sh` (in `[internal reference removed]`), **before any gate output exists** (the commit time is the proof).
  **Owner:** partner (lane B machine). **Decision owner** for team use: Sebastien. Kind: tooling validation, no team hypothesis.
- **Why:** Sebastien routes canonical-read jobs (`[internal reference removed]`, `scrollprize/ink_canonical_2um`) to [internal reference removed],
  where the model runs at stock speed (CAPACITY.md: "the canonical model: stock speed here, ≈ 2-3× slower per card"). The fast
  lane passed for the copy model ([internal reference removed]) and Hecate 9.6 ([internal reference removed]) but **FAILED for Hecate 2.4** ([internal reference removed], G2 nonzero 98.96 % vs 99 %).
  The canonical reader is the risky case: **same encoder as Hecate 2.4** (ResNet3D-152, Bottleneck [3, 8, 36, 3], 7x7x7 stem),
  3D U-Net decoder, 2.4 µm.
- **Facts measured on CPU before this push** (`stage0.json`, `plan.json`): checkpoint `r152_3ddec_v2_l5_epoch13.ckpt`
  1,549,556,213 B, sha256 `36dd0de8…73c3e0` = the Hugging Face LFS oid at revision `075855bc`; pickle holds only
  `OrderedDict`, `FloatStorage`, `LongStorage`, `_rebuild_tensor_v2` (loads with `weights_only=True`); strict load: 0 missing,
  0 unexpected keys, `with_norm` true; 130.4 M parameters. **56 Conv3d layers take the fast path** (50 bottleneck `conv2`, 3 of them
  stride 2, + 6 decoder `ResConvBlock3D` convs); 112 1x1x1 convs and the 7x7x7 single-channel stem stay stock. The checkpoint
  carries a GradScaler state (`MixedPrecisionPlugin`, scale 65536): it was **trained with fp16 AMP**, and `infer_crop.py` runs
  `torch.autocast('cuda')`, whose default dtype is **fp16** (checked: `torch.get_autocast_dtype('cuda')` = float16). So the
  production precision is fp16, not bf16 as in [internal reference removed], and the gated pair here is fp16.
- **Lesson from [internal reference removed] used in the design:** at 2.4 µm two *independent* low-precision roundings disagree with each other by
  more than 1 grey on ~1 % of pixels even when each is equally close to fp32 ([internal reference removed] post-hoc: stock 72.57 % vs fp32_taps 72.62 %
  within 1 grey of the fp32 map). A bar on arm-vs-arm agreement fails an equivalent arm by luck. **Every gate here compares each
  low-precision arm with the stock fp32 reference** and asks whether fp32_taps is further from it than stock is.
- **Truth rule:** no labels, no humans, no ratings. Every number compares an arm with a reference computed on the same input.
  The team map `team_ink.tif` is used only as the downstream yardstick (Pearson r, below) and the Stage 0d plumbing control.

## Data (published scroll data already cached on [internal reference removed]; nothing private, no rating pages, no keys)

- `[internal reference removed]`: PHerc0841 region R along w00, rendered at 2.403 µm, uint8 (109, 2784, 6496), 1.044 cm², centre
  layer 54 = w00 (cached since the 10-01 repro; the same input as [internal reference removed]'s 2.4 µm arm). On this region the canonical model
  reproduces the team map at r 0.96 (Sebastien's box, `[internal reference removed]` 11:36 K0 amendment, `[internal reference removed]`).
- **Window:** infer_crop's `--offsets` arithmetic, 62 layers starting at 54 − 31 + o (o = 0 → layers 23-84).
- **G1 tiles:** infer_crop's own grid (tile 256, stride 128, end tile appended): 21 x 50 = 1,050 tiles; eligible = ≥ 90 % of the
  tile's columns have a nonzero voxel in the o = 0 window: all 1,050 are eligible (support = 100 % of R). **64 tiles** chosen by
  `linspace(0, 1049, 64).round()` over the row-major list; tile k gets config `[(fwd,0), (fwd,0), (rev,0), (fwd,−20)][k % 4]`
  → 32 forward at o = 0, 16 reversed (the team's blank control direction), 16 forward at o = −20 (the deepest offset Sebastien's
  DEPTH-GRID uses). `plan.json` (made before this push). No selection on any output.
- **G2 renders:** the whole of R, o = 0, forward (the reading direction on R).

## Arms (script `canon_gate.py`, run by `job.sh`; one card, GPU 1, no expandable segments, MIOpen cache in a private cache folder)

Model loaded with villa's `load_model` (via `infer_crop`, one visible card asserted, so no DataParallel). Whole renders call
**`infer_crop.run` unchanged**; arms differ only by the conv path and by an autocast override inside the object `run` calls.
G1 input = run's tile preprocessing (clip 0-200, /200), output = `model.forward(t)`: the 64 x 64 logits (pre-sigmoid).
- **REF:** stock `nn.Conv3d`, fp32 (autocast off, TF32 off). The fp32 stock reference. Whole render at batch 1 with
  `ROCBLAS_USE_HIPBLASLT=0` (fp32 GEMMs run at ~1 TFLOPS under hipBLASLt 1 here); in the G1 process hipBLASLt 1 (recorded: REF
  tile 0 under 0 vs 1, and with TF32 on vs off; expected equal to fp32 rounding).
- **S16:** stock, fp16 autocast = `infer_crop.py` as Sebastien runs it. `ROCBLAS_USE_HIPBLASLT=1`.
- **F16T:** `conv3d_fast.enable(mode="fp32_taps")`, fp16 autocast (input, weight rounded to fp16 as stock's input; taps in fp32;
  one rounding). `ROCBLAS_USE_HIPBLASLT=1`. Production activation: `PYTHONPATH=[internal reference removed] CONV3D_FAST=1
  CONV3D_FAST_MODE=fp32_taps` (here the same `enable` call made inside the script).
- **SB16 / FB16T (record only, G1 tiles only):** the same two in bf16, so the bf16 question is answered for free. Never gated.
- **Conv-path check (every forward):** a counter on the fp32_taps function: F16T / FB16T must call it exactly 56 times per
  forward, the stock arms 0 times, in G1 and in every render (else VOID).

## Stage 0 (VOID if any fails; no verdict is written as a result)

- **0a** weights size + sha256 as above; **0b** strict state-dict load (0 missing, 0 unexpected); **0c** 56 fast layers on the real
  model (all three measured on CPU before this push: pass); **0d positive control:** Pearson r of the REF map and of the S16 map
  against the team map (score.py recipe: both to 9.6 µm/px with `zoom(order=1)`, Gaussian σ 1) **≥ 0.90** (0.96 on Sebastien's
  box; a wrong window, normalisation or direction drops it far below: Hecate's reversed map scored 0.18). **0e** every render completes.

## Gates (fixed now; no bar is loosened after any output; an amendment may only be made before output exists and is dated)

Notation: dX = |X − REF| per logit (G1); for maps, grey g = floor(255 · clip(p, 0, 1)) exactly as the tile code (`ws.tile`)
makes the reading pictures; D(X, set) = share of pixels in the set with |g_X − g_REF| > 1.

- **G1, numerics (64 tiles, all 64 x 64 logits).** Pass needs all four:
  (a) **pooled** over the tiles: p99(dF16T) / p99(dS16) ≤ **1.10** AND mean(dF16T) / mean(dS16) ≤ **1.10** ([internal reference removed]'s bar);
  (b) **worst tile, max-abs:** max over tiles of max |F16T − REF| ≤ **2.0 x** the same for S16;
  (c) **worst tile, correlation:** 1 − min over tiles of r(F16T, REF) ≤ **2.0 x** (1 − min over tiles of r(S16, REF));
  (d) **finite:** no tile where F16T has a non-finite logit and S16 does not (fp16 overflow guard).
  Why 2.0 for (b)/(c): one tile is a noisy ruler; in the 2026-10-08 instrument check, single sub-crops of modes whose per-layer
  noise equals stock's ranged up to 1.65x (p99) / 1.88x (mean) against stock. A defect (wrong layer, wrong rounding) is far
  outside that. Recorded: per-tile p99 ratios (median, range, share over 1.10), ratios per config (fwd 0 / rev 0 / fwd −20), the
  same distances in grey levels (255 · |σ(F) − σ(REF)|), and the bf16 pair against REF.
- **G2, decisions on the whole render** (REF batch 1, S16 and F16T batch 4 = infer_crop's default). Pass needs all of:
  (a) **grey agreement with fp32:** D(F16T) ≤ **1.10 x D(S16) + 0.001** on BOTH sets: *support* (input-defined: any nonzero voxel
  in the window; 18.08 M pixels) and *nonzero* (support pixels with g_REF ≥ 1; defined by the reference, the same set for both arms).
  The 0.001 allowance keeps a near-zero D (fp16 rounds ~8x finer than bf16) from turning a few thousand scattered pixels into a
  verdict: 0.1 % of R is ~18 k pixels (~1 mm² summed over 1.04 cm²) of 2-grey moves, and one pixel of the 800 px reading tile
  averages ~66 map pixels, so unclustered moves of that size do not change a tile grey level (the tile record below checks it).
  (b) **thresholded ink mask** (p ≥ 0.5, the sigmoid's decision point and grey 127/128): share of support pixels where the arm's mask
  differs from REF's: Dm(F16T) ≤ **1.10 x Dm(S16) + 0.0002** (0.02 % ≈ 3.6 k pixels ≈ 0.2 mm²).
  (c) **downstream metric** (the number the team reports for canonical maps on 0841: Pearson r vs the team map, score.py recipe):
  |r(F16T) − r(REF)| ≤ **max(1.10 x |r(S16) − r(REF)|, 0.001)**. 0.001 is 20x tighter than the PC-0841 repro tolerance (0.81-0.85
  around 0.83) and below the third decimal anyone reports.
  Recorded (never decide): Dice at 0.3 / 0.5 / 0.7, grey correlation, mean and max |Δgrey| vs REF; **[internal reference removed]-style F16T vs S16 within
  1 grey** (support; nonzero = max(F, S) ≥ 1) for comparison with Hecate 2.4's 98.96 %; the 800 px reading tile (ws.tile code) of
  each arm vs REF's, share within 1 grey; F16T batch 1 vs batch 4 within 1 grey.
- **G3, speed and memory.** (i) Per tile: median forward time over the 64 G1 tiles (batch 1, after 2 warm-up rounds; kernel
  builds not counted) S16 → F16T ≥ **1.3x**, AND whole render batch 4 S16 → F16T ≥ **1.3x** (after one warm-up forward each).
  (ii) Peak reserved VRAM ≤ **30.5 GB** in the G1 process and in both F16T renders (batch 4 and 1). Recorded: s/cm² of every
  render, REF and S16 peaks, bf16 per-tile times.
  **Stop rule:** if the G1 per-tile speedup is below 1.3x, the renders are not run and the verdict is FAIL (G3).
- **Verdict** (order frozen in `decide()`): VOID if Stage 0 fails or an arm leaves its conv path; FAIL (G3) if the stop rule fires;
  VOID if a render is missing or 0d fails; **PASS** = G1, G2 and G3 all pass; **FAIL** = any fails (the result log names which).

## What each outcome changes

- **PASS:** Chris may add the remote_allow job type **`canon-infer-fp32taps-v1`** (draft in `remote_allow_draft.json`):
  infer_crop with `CONV3D_FAST=1 CONV3D_FAST_MODE=fp32_taps`, fp16 autocast, batch 4 (batch 1 too if the gate measured it faster
  AND its F16T b1-vs-b4 record is ≥ 99.9 % within 1 grey). CAPACITY.md row + `capacity.json` + `speed_table.json`
  (`canon-2p4-fp16-fp32taps`, measured s/cm²) updated; record `conv_path = conv3d_fast+fp32_taps`. Sebastien decides team use.
- **FAIL / VOID:** no change: canonical reads on [internal reference removed] stay stock fp16 (`canon-infer-stock-v1`, which needs no gate: it is
  infer_crop as it is). The result log names the failed gate; a re-gate needs its own prereg.
- Not claimed: other scrolls or scans, offsets beyond those tested (0 in G2; 0 and −20 and reversed in G1), batch sizes other
  than 1 and 4, bf16 production use (recorded only), or whether the canonical map is right (this is equivalence to stock fp16).

## Compute and process

- **GPU (estimate, not measured: no canonical run on [internal reference removed] yet):** S16 render ~3 min (Hecate 2.4 stock 168 s/cm²; the canonical
  model on the 5070 Ti ~70 s/cm², DEPTH-READ 12 maps in 33 min; [internal reference removed] stock 2-3x slower), F16T renders ~1.5-2 min each, REF stock
  fp32 render ~15-35 min (stock fp32 / bf16 = 9x on the copy model), G1 ~5 min incl. load and MIOpen builds. **About 30-50 min
  on one card; claim 2 h.** One `[internal reference removed] claim canon-gate --hours 2`, released by the job's trap (also on failure). Before
  every GPU step `job.sh` reads `remote.py status` (text lines; a running / paused / accepted job stops it; [internal reference removed]'s check read it
  as JSON and failed open) and `[internal reference removed] status` claims.
- Outputs next to the script: `plan.json`, `stage0.json`, `g1.json`, `render_<arm>_b<batch>.json`, `canon_gate_summary.json`.
  Arrays and maps in git-ignored `[internal reference removed]`, deleted after the result log.
- CPU selftest (`canon_gate.py --selftest`, tiny synthetic model, the real `infer_crop.run`): decision rules, scorer equivalence,
  an identical arm (must PASS), a 3 % weight-broken fast arm (must FAIL G1), a [internal reference removed]-like arm pair (disagree with each other on
  ~17 % but equally far from REF: must pass G2). ALL_PASS before this push.
- **Kill date:** 2026-10-16. Not scored by then: closed as not run.
