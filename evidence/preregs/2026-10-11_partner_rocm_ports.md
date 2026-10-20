# Pre-registration: organizer tools on AMD/ROCm (lasagna, spiral fitter) + AMD package v1 speed/VRAM table

- Date: 2026-10-11 (pushed before any GPU output of these runs). Owner: Chris's side ([internal reference removed]), team contribution
  (Chris Thomas + Sebastien Chicoineau). Small-wins register rows 2 and 2b (`[internal reference removed]`).
- Upstream: ScrollPrize/villa main d63d3e968 (2026-10-11), fresh shallow clone in `[internal reference removed]` (git-ignored).
- Machine: [internal reference removed], 2x AMD Radeon AI PRO R9700 32 GB (gfx1201), ROCm 7.2.2. Data: public PHerc0841 (and other public
  open-data scrolls) only. No [internal reference removed] data, no answer keys, ratings or [internal reference removed] files.

## Questions
1. **Lasagna** (`lasagna/`): can its install path select a ROCm torch build instead of silently falling back to CPU
   (`scripts/bootstrap_venv.py` picks the build from `nvidia-smi` only), and does a run on the R9700 give the same
   result as the reference run?
2. **Spiral fitter** (`spiral-fitting/`): can it install and run on Linux + ROCm without the CUDA-only packages
   (`cupy-cuda12x`, `cucim-cu12`, `brook-cu12`, torch from the cu128 index), and does a fit match the reference fit?
3. **Speed/VRAM table** for the organizers' 3-D models on the R9700: stock path vs the already-gated fast path.

## Reference and bars (frozen now)
- **Reference run:** the same commit, inputs, config and seed on CPU torch on [internal reference removed] (and, if Sebastien supplies one,
  his RTX 5070 Ti run; both are reported, CPU is the bar).
- **Deterministic tensor outputs** (a single forward pass or a fixed computation, e.g. a normals/SDF volume):
  PASS if relative L2 error vs the reference is <= 1e-4 in fp32 (<= 1e-2 for a declared bf16/fp16 path).
- **Optimisation fits** (lasagna fit, spiral fit; not bit-reproducible across devices): run 3 seeds on the reference
  device and 3 on ROCm. PASS if the median ROCm-vs-reference distance (`fit_compare.py` point-to-surface median, and
  per-sector turn agreement where it applies) is <= the largest same-device seed-to-seed distance on the reference
  device. Turn assignment must agree on every sector the reference seeds agree on among themselves.
- **Run completes:** no NaN/Inf, no silent CPU fallback (the run log must show the HIP device), exit 0.
- **Speed/VRAM table:** median of 3 timed runs after 1 warm-up, `torch.cuda.max_memory_allocated` + rocm-smi peak;
  only paths that already passed their equivalence gate (CANON/HECATE/FASTCOPY gates, `RESULTS.md`) are listed
  as "fast"; the table reports measurements only (no claims beyond them).
- **No moving the bar after seeing results.** A failing bar means the port is not released as working; the gap goes
  into `KNOWN_ISSUES.md` and the root cause is diagnosed (no tolerance loosening, no workaround that hides it).

## Compute rules
- GPU only through `[internal reference removed] claim <name> --hours <=1` per step, never while a v6 queue job is approved or running
  (scroll jobs first); one card per task (`ROCR_VISIBLE_DEVICES=0,1` + `HIP_VISIBLE_DEVICES=n`).
- fp32 runs with `ROCBLAS_USE_HIPBLASLT=0`, bf16/fp16 with `=1` (CLAUDE.local.md bench).

## Release
Drafts only, in `[internal reference removed]`. Upstream PRs (expected: a `rocm` backend for lasagna's bootstrap, ROCm
dependency markers for spiral-fitting) go up only with both partners' OK and Chris's own written commentary
(villa CONTRIBUTING). Results land in `[internal reference removed]` with sources.
