# Known issues

Observed on 2x Radeon AI PRO R9700 (gfx1201), ROCm 7.2.2, PyTorch 2.14.1+rocm7.2, Ubuntu. Stated as observed; we did
not find root causes in the driver.

- **GPU resets can wipe fan and power-cap settings.** After a GPU reset (we saw several, during allocator page faults
  and a cross-device hang), the fan curve and the power cap we had configured were gone and the card was back at its
  default power limit (300 W instead of our 275 W). If you configure either, re-apply them after a reset (we do it from
  a small service that can be restarted). On this setup the second card ran hotter (junction 88-94 C at 275 W under
  training).
- **Allocator `expandable_segments:True` gave random faults** (NaN loss, illegal-address and launch-failure errors).
  Keep it off for training and long inference; see README.
- **Long trainings can fragment the allocator**: validation ran out of memory with memory still free. See README
  (`max_split_size_mb:128`).
- **SageAttention corrupts output on gfx1201.** Use PyTorch's built-in scaled-dot-product attention (SDPA) instead.
- **Cross-device runs can hang the GPU** (we saw a hang followed by a reset when one process mixed both cards). Keep one
  process on one card, and select it with `HIP_VISIBLE_DEVICES` (do not combine it with `ROCR_VISIBLE_DEVICES`
  filters pointing at the same device: a double filter left zero visible GPUs).
- **hipBLASLt vs rocBLAS** (README): fp32 matrix multiplies fall to 1 TFLOPS with hipBLASLt on, bf16/fp16 fall to
  11-13 TFLOPS with it off. There is no single global setting; set it per job.
- **3-D convolutions are slow in stock MIOpen on gfx1201** (about 3-9 TFLOPS). `conv3d_fast.py` works around it for
  3x3x3 layers; other 3-D kernel shapes are not helped.
- **`conv3d_fast.py` modes other than `fp32_taps` are not bf16-equivalent** to stock on every input (older 1-grey-level definition); default `taps`
  and `acc32_bias` add noticeably more rounding noise per layer on the copy model, although both passed the later error-ratio ruler there (`RESULTS.md` section 6). Hecate at 2.4 µm failed our first equivalence bar (fast vs stock within 1 grey level) but passed a later, different ruler (error against fp32 no worse than stock); see `RESULTS.md` section 6.
- **Lasagna zero fill and tested setup:** the gate runs read zeros past the downloaded crop chunks; on real PHerc0841 data alone `acc32_bias` was about 7 % above stock's error (post hoc), and the registered re-check on 16 all-real tiles (GATE16-REAL, 2026-10-14) FAILED: U 1.113 > 1.10 (`RESULTS.md` section 10). The tested setup is exactly the 3 layers of `RUNBOOK.md` section 10; a global `enable(mode="acc32_bias")` (70 layers) is untested.
- **Lasagna: `fp32_taps` on the 3 slow layers FAILED its gate** (error 2.06x stock against the fp32 model) although it is 10.3x faster and the most accurate mode per layer; only `acc32_bias` passed the first gates (16 tiles, then 3 crops of 3 scrolls at 9.36 um), and it then failed the real-data re-check, so no lasagna speed-up is offered (`RESULTS.md` section 10). Per-layer accuracy did not predict whole-model error.
- `torch._C._accelerator_setAllocatorSettings` is a private PyTorch API and may change between versions.
- **Setup traps found while writing `RUNBOOK.md` (2026-10-11, fresh Python 3.14 venv):**
  - Installing the name `vesuvius` from PyPI instead of the villa source folder replaced `zarr` 3 with 2.18 and
    `torchvision` with a build without ROCm: `RuntimeError: operator torchvision::nms does not exist`. Install `torch` and
    `torchvision` together from the ROCm index first, and install villa's package from its folder.
  - Reading LZW-compressed layer TIFFs needs `imagecodecs` (`KeyError: <COMPRESSION.LZW> requires the 'imagecodecs' package`).
  - villa's `vesuvius` package (main, 2026-10-11) requires Python 3.14. The copy-model script imports `vc`, the C++
    bindings of Volume Cartographer (no wheel); we did not build it on Linux/ROCm and ran the script against a small
    pure-Python stand-in that is not part of this package. **The copy-model rows are therefore not reproducible from this
    package** (they need VC3D's `vc` binary). villa `main` has changed the `vc` calls since the commit we
    measured on (`f4570bfa6`).
  - Every ROCm process prints `(null): No such file or directory` four times at start-up. Harmless in all our runs; not
    investigated.
- **Organizer tools (2026-10-11, villa main d63d3e968; details in RESULTS section 10):**
  - Lasagna's `bootstrap_venv.py` only looks at `nvidia-smi`; without it, it silently installs CPU torch. Our patch adds a
    `rocm` backend. Until it is upstream, pass the ROCm torch yourself.
  - Lasagna `predict3d --device cpu` crashed for any `--scaledown` above 0 (`ValueError: X rows must be contiguous`;
    `_pyrdown3d` returns a strided view). One-line `.contiguous()` fix in our patch; values unchanged. NVIDIA runs hide it.
  - Lasagna's normal channels (`nx`, `ny`, uint8) flip a normal to the nz >= 0 hemisphere before encoding, so normals lying
    in the xy plane change sign under tiny numeric noise (bf16 rounding, another device). Compare normals as unoriented lines,
    not byte by byte.
  - Lasagna's inference runs in bf16 autocast (hard-coded). Expect about 1e-2 relative differences between any two devices
    in bf16; in fp32 our R9700 matched the CPU within 1e-4.
  - On the stock MIOpen 3-D path, lasagna inference on one R9700 was about 4x slower than a 32-thread CPU (17 s per tile; target ~3.7 s/tile, the CPU's rate). The first tile is
    NOT slow (17.3 s in the log); an extra ~340 s per run comes from lasagna's own GPU hand-off (`[gpu_pause] pause failed:
    timed out`, 300 s default wait), not from MIOpen compilation. Confirmed 2026-10-13 (start to tile 1 = 20 s with no stale owner). The 16.9 s/tile is 3 full-resolution layers (192^3, 32/64 channels) on MIOpen's naive bf16 conv kernel (16.1 s of 16.5 s); the other 79 convs are fast. See the lasagna profile result (our working files, not in this repository).
  - Lasagna's cylinder extension needs libigl 2.6 headers (2.5.0 fails to build); they are not pip-installable
    (`LIBIGL_INCLUDE_DIR`, `EIGEN_INCLUDE_DIR`).
  - villa PR #2055 (opened 2026-10-12, vincentl065: "spiral-fitting: read SQLite-backed track DBMs on Python < 3.13 (enables
    Windows + AMD ROCm runs)") is a different fix from ours: it changes how the track store is read on Python < 3.13, while our
    villa PR #2113 moves the CUDA-only packages into a dependency group. We have not tested the two together.
  - The spiral fitter has no CPU path. It runs on ROCm unchanged through the torch `cuda` API once the three CUDA-only
    packages are out of the way. `uv sync` in `spiral-fitting` uses the cu128 torch index, so ROCm users install with
    `uv pip` (recipe in our patch's README lines). The CUDA-only packages are used only by the GPU track-extraction backend;
    the Kimimaro CPU backend (`.[cpu]`) is the fallback.
- **One input is a weak equivalence test.** `check_equivalence.py` on a single input printed fast/stock error ratios above
  1.1 for paths that passed their gate on many inputs (measured with the older whole-model-cast regime of `check_equivalence.py`, not under autocast, which the gates used; the
  new default `--autocast auto` matches the gates, and these numbers were not re-run; canonical model, one tile, fp16: 1.23 mean / 1.43 p99; v8-in, 4
  tiles, fp16: 1.33 / 1.44; Hecate 9.6 one patch: 1.06 / 1.07). For Hecate 2.4 um (the path that failed its gate) one patch
  printed 1.87 / 1.95. The per-tile ratios in the gate runs ranged about 0.65 to 1.36, so use several real inputs. For v8-in
  the gate was the downstream AUC / AP change and agreement of the output map, not this noise ratio.
