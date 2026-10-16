# Running the Vesuvius Challenge 3-D models on AMD GPUs (ROCm)

Authors: Chris Thomas (GitHub Baz248) and Sebastien Chicoineau, Vesuvius Challenge team; `conv3d_fast.py` written by Chris Thomas with Claude. Licence: MIT. Status: first release, October 2026. Tested on one AMD machine; outputs checked on one NVIDIA card (see "On NVIDIA").

The Vesuvius Challenge's open-source models are mostly written for NVIDIA cards. They are plain PyTorch, so they
*run* on AMD cards, but on the newest AMD generation (RDNA4, "gfx1201", for example the Radeon AI PRO R9700) they ran
far slower than they should, and a few default settings caused random crashes. This repository collects what we
measured, the settings that fixed it, and one small file (`conv3d_fast.py`) that makes the 3x3x3 convolutions 2.8-7x faster on those cards
(18x for a whole neural-tracing Copy job, 2-4x for the ink models), with equivalence checked on real scroll data.

## Author's note (Chris Thomas)

I noticed really quickly the Vesuvius Challenge's AI models (ink detection, sheet tracing, lasagna) ran extremely slow on my cards. I also noticed they were much quieter than normal and when I checked they were really not pulling the amount of power I expected. The task was also logging some random crashes. Upon some research I found that the task can improve the speed by splitting the 3D Images into 3 2D images. After this discovery I asked for a a full research on the topic finding a few more levers that could speed it up further.  

AI Opus 5.5 used to summarize the details below.  

What we found and fixed:  
1. The slow 3-D math. AMD's math library (MIOpen) has fast code for 2-D image math, but for 3-D it falls back to slow generic code. The file conv3d_fast.py rewrites each 3-D step as three 2-D steps that add up to exactly the same answer. The card then uses its fast 2-D code.  
2. Two settings that matter:  
A math-library switch (ROCBLAS_USE_HIPBLASLT). Turn it on for half-precision jobs and off for full-precision jobs: the wrong setting is slower.  
A memory setting that, left at its default, caused random crashes. We turned it off.  
3. Every speed-up was checked against the normal version on real scroll data before we called it usable. Each check had a pass/fail rule written down and pushed before the test ran.  

Results, from the package files:  
Sheet-copying model  
What it does: grows and copies a scroll sheet surface  
Speed-up: about 18x (6,528 s down to 354 s, whole job of 78 pieces); not reproducible from this package (needs VC3D's vc)  
Quality check: passed  

v8-in ink model (9 µm)  
What it does: finds ink  
Speed-up: 2x (up to 4.2x with extras)  
Quality check: passed, 100% of pixels within 1 grey level  

Canonical ink model (2.4 µm)  
What it does: finds ink  
Speed-up: about 2x  
Quality check: passed  

Hecate ink model (9.6 µm)  
What it does: finds ink  
Speed-up: 1.66x  
Quality check: passed  

Hecate ink model (2.4 µm)  
What it does: finds ink  
Speed-up: none offered. The fast path missed its bar (98.96 % vs 99 %); a later 'no worse than stock rounding' check passed, but the failed gate stands.  

Lasagna  
What it does: finds which way the sheets run  
Speed-up: none offered. The 3-layer fast path is 16x faster but failed its real-data re-check (about 10 % more error than stock).  
Quality check: it passed its first registered gate (acc32_bias, 3 layers) but FAILED the registered real-data re-check (GATE16-REAL, 2026-10-14: error ratio U 1.113 against the 1.10 bar). The failed gate stands. `RUNBOOK.md` section 10 keeps the recipe for experiments only (`RESULTS.md` section 10).  

SETTINGS THAT MATTER  

ROCBLAS_USE_HIPBLASLT: turn it on for half-precision jobs and off for full-precision jobs. The wrong choice is up to 13x slower. (This comes from notes, not a saved file.)  

expandable_segments (a memory setting): left on, it caused random crashes, such as a broken training loss and false out-of-memory errors. We turn it off.  

End of AI summary.  

If you are using AMD Cards you should be able to check really quickly if it helps speed up the process for them.  

## Who this helps

Anyone with an AMD Radeon or Instinct GPU who wants to run the models from
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) and related releases:

- the neural-tracing "Copy" model (surface growing / sheet copying),
- the lasagna `predict3d` model (it runs on ROCm; no speed-up is offered for it, see `RESULTS.md` section 10),
- the v8-in 9 µm ink model,
- the 2.4 µm canonical ink model,
- Hecate (ink model, 2.4 µm and 9.6 µm versions).

Most of the speed-up matters for RDNA4. Older cards (RDNA3, CDNA) may already pick better kernels; the same
check below tells you in two minutes whether it helps you.

## Upstream changes (villa pull requests)

Three small fixes from this work are proposed to [ScrollPrize/villa](https://github.com/ScrollPrize/villa) itself, so AMD users
do not need this repository for them (open as of 2026-10-16):

- [villa #2113](https://github.com/ScrollPrize/villa/pull/2113): spiral fitter, the CUDA-only packages move to a default `cuda`
  dependency group, plus a ROCm install recipe.
- [villa #2114](https://github.com/ScrollPrize/villa/pull/2114): lasagna, fixes the `predict3d --device cpu` crash
  "X rows must be contiguous" (needed for a CPU reference run).
- [villa #2115](https://github.com/ScrollPrize/villa/pull/2115): lasagna, `scripts/bootstrap_venv.py` installs the ROCm build of
  PyTorch on AMD machines.

## What is in here

| File | What it is |
|---|---|
| `conv3d_fast.py` | Drop-in patch for `nn.Conv3d`: runs each 3x3x3 convolution as three 2-D convolutions that add up to the same result. |
| `check_equivalence.py` | Compares stock vs patched output (and timing) on **your** model and input. Has a CPU `--selftest`. |
| `tests/test_conv3d_fast.py` | CPU tests: the patch equals `nn.Conv3d` in float64, outputs and gradients. |
| `RUNBOOK.md` | From a fresh Ubuntu + ROCm 7.2 box to running each model on public PHerc0841, every command, with what was verified. |
| `examples/` | `fetch_pherc0841.py` (public test inputs), `builders.py` (model loaders for the check), `run_with_fast.py` (turn the patch on for any script). |
| `RESULTS.md` | The measured numbers and where each came from, including the speed / VRAM table (section 9). |
| `measurements/` | Raw per-row files of the section 9 table and the timing driver that wrote them. |
| `evidence/` | Copies of the result files, patches and pre-registrations that `RESULTS.md` cites. |
| `KNOWN_ISSUES.md` | Things that went wrong on this hardware, stated as observed. |

## Tested setup

Everything below was measured on one workstation. Other setups may differ.

- 2x AMD Radeon AI PRO R9700 (32 GB each, gfx1201), Threadripper PRO 9955WX
- Ubuntu, kernel 7.0
- ROCm 7.2.2
- PyTorch 2.14.1+rocm7.2 (HIP 7.2.53211), numpy 2.5.3, zarr 3.4.0
- Python 3.12 in a `uv` virtual environment

## Install

Install PyTorch (and torchvision, if you use it) from the ROCm wheel index **first**, before anything that depends on
torch, so that no package pulls in a CUDA build. `RUNBOOK.md` has the full path, from fresh install to each model:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install numpy zarr          # then the packages of the model you want to run
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```

PyTorch's ROCm build answers to the `torch.cuda` API, so model code that says `cuda` runs unchanged. If a package
you install later replaces torch with a CUDA build, reinstall torch from the ROCm index last. (One more
install-order trap: `opencv-contrib-python` must be installed after other OpenCV packages, otherwise `cv2.ximgproc`
goes missing.)

## Settings that matter (and why)

These come from measurements on the setup above, not from guesses.

### 1. `ROCBLAS_USE_HIPBLASLT`: depends on the number format

An 8192 x 8192 matrix multiply on one R9700 (**from our notes of 2026-10-01; no benchmark script or output file was kept, so
this table is not backed by a file in this package. Re-measure on your card**):

| setting | fp16 | bf16 | fp32 |
|---|---|---|---|
| `ROCBLAS_USE_HIPBLASLT=0` | 13 TFLOPS | 11 TFLOPS | 13 TFLOPS |
| `ROCBLAS_USE_HIPBLASLT=1` | 116 TFLOPS | 117 TFLOPS | **1 TFLOPS** |

So: fp16 / bf16 jobs want `=1` (about 9x faster), fp32 jobs want `=0` (13x faster). Set it per job, not globally:

```bash
ROCBLAS_USE_HIPBLASLT=1 python run_bf16_job.py
ROCBLAS_USE_HIPBLASLT=0 python run_fp32_job.py
```

### 2. `PYTORCH_ALLOC_CONF=expandable_segments:False`

With `expandable_segments:True`, a training library that sets it by default gave, at random: NaN loss at step 0,
`hipErrorIllegalAddress` inside a MIOpen convolution, and `hipErrorLaunchFailure`, with the log line
"expandable_segments: memory mapping failed with OOM ... trying to map 20971520 bytes" while 30 GB was free.
With it off, 3 of 3 training runs were clean; with it on, the NaN reproduced. It changes only memory placement, not
the maths. Set it explicitly (torch >= 2.14 reads `PYTORCH_ALLOC_CONF`; older versions read
`PYTORCH_HIP_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF`):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:False
```

Check what your libraries set: some training code (and some service files) default it to `True`.

### 3. `max_split_size_mb:128` for long trainings with large validation tensors

In one long fine-tuning run the allocator became fragmented: long-lived gradient and optimizer-state blocks pinned
the large segments that the validation step later needed, and validation ran out of memory around step 150 although
total memory was not exhausted. Telling the allocator not to split blocks of 128 MiB or more fixed it. It affects
only tensor placement, not the maths (bit-identical results across the two allocators are not claimed on ROCm,
because MIOpen's algorithm choice can depend on the workspace it is given). It can be set at runtime, after
`import torch` and before the model is built:

```python
import torch
torch._C._accelerator_setAllocatorSettings("max_split_size_mb:128")   # note: a private torch API
torch.cuda.set_per_process_memory_fraction(0.85)                     # optional headroom
```

(Setting `PYTORCH_ALLOC_CONF` to this value would replace `expandable_segments:False`; if you want both in the
environment, use `PYTORCH_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128`.)

## The slow 3-D convolution path on RDNA4

Symptom: a 3-D CNN that should take seconds takes minutes, with the GPU "busy" but only a few TFLOPS used.

Cause: on gfx1201, MIOpen has tuned kernels for 2-D convolutions (for example a Winograd assembly kernel,
`ConvWinoFuryRxS`, about 25 TFLOPS in fp16 for 3x3), but for 3-D convolutions it falls back to a generic
"im2col + matrix multiply" kernel (`GemmFwdRest`, about 3 TFLOPS) or a naive kernel.

See it on your own model:

```bash
MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=5 python your_script.py 2> miopen.log
grep -i -E "GemmFwdRest|ConvWino|Naive" miopen.log | sort | uniq -c | sort -rn | head
```

If the 3-D layers show `GemmFwdRest` or a naive solver, you are on the slow path.

The fix is plain maths: a 3x3x3 convolution equals the sum of three 3x3 convolutions, one per depth slice. Each of
those is a 2-D convolution, which uses the fast kernels. `conv3d_fast.py` does exactly this and PyTorch's autograd
differentiates it. The fp64 gradient tests (`tests/`) show it also works for training; training *equivalence* was gated only for
the v8-in model (RESULTS section 4).

```python
import conv3d_fast
conv3d_fast.enable()            # before building / running the model; conv3d_fast.disable() undoes it
```

What it changes: only `nn.Conv3d` layers with groups = 1, dilation 1, zero padding and a 3x3x3 kernel with more than
one input channel. 1x1x1 layers stay stock (the decomposition makes them 2.5-3x slower), and so does a 7x7x7
single-channel stem, where it is a wash (0.94x). An optional `enable(stem=True)` rewrites that stem as one 2-D
convolution. The patch only acts on GPU tensors.

Per-layer speed-ups measured with this policy (fp16, batch of 16 tiles, one R9700) range from 2.8x to 7x for
3x3x3 layers at stride 1 and about 1.45x summed over the stride-2 layers (per layer 0.90x-1.69x; the 512-channel stride-2 layer is slower than stock
and is still rewritten by the policy; `RESULTS.md` section 2).

### Which mode to use

Float32 and float64 are fine in every mode: the result equals stock up to summation order (float64 relative error
about 2e-15 in `tests/`, and about 4e-5 mean absolute difference on activations of mean magnitude 42 in a full fp32
model, see `RESULTS.md`).

In **bf16 / fp16** the rounding differs from stock, and the modes are not equal:

| mode | what it does | quality in bf16 (copy model, see RESULTS.md) |
|---|---|---|
| `taps` (default) | per-tap 2-D convs, summed in the run precision | adds more rounding noise than stock (median 1.44x per layer, range 1.27-1.54); passed our bar for the v8-in ink model (the ruler there was the downstream AUC/AP change, not this per-layer noise), but not safe for the copy model |
| `acc32_bias` | taps and bias summed in fp32; each tap's 2-D output is first rounded to bf16, then the sum is rounded once | between the two per layer; not equivalent on every crop on the copy model by the older check; passed the later ruler on the copy model (78 crops) and the lasagna gates (16 tiles, then 16 fresh tiles, then two more scrolls' crops, see `RESULTS.md`; for lasagna only as the **3-layer** setup in `RUNBOOK.md` section 10: a global `enable(mode="acc32_bias")` rewrites all 70 eligible layers and is NOT the tested setup; the later registered re-check on 16 all-real PHerc0841 tiles FAILED, U 1.113 > 1.10, so no lasagna speed-up is offered) |
| `taps_as_channels` | one 2-D conv with the 3 taps stacked as channels | per-layer noise equals stock exactly |
| `fp32_taps` | taps computed and summed in fp32 on bf16-rounded inputs (close to stock's arithmetic, but stock's naive kernel is not correctly rounded on lasagna's layers) | **the only mode whose per-layer noise was never above stock** (71 of 71 layers on the copy model, median 0.96x); the one we use for the copy model. Per-layer noise is not whole-model error: on lasagna this mode FAILED its whole-model gate |

`fp32_taps` costs extra memory (fp32 copies) but was also the fastest bf16-safe mode in our test (0.87 s vs 1.27 s
for `taps_as_channels`, 75.6 s stock). Use it when in doubt, **except for lasagna, where it failed** (per-layer accuracy did not predict whole-model error; see `RESULTS.md` section 10):

```python
conv3d_fast.enable(mode="fp32_taps")
```

Whatever you choose, **verify on your own model** (next section). Whole-model noise on a single crop is partly luck:
on sub-crops of one crop every mode ranged from roughly 0.3x to 1.9x of stock's own bf16 noise, so judge on several
real inputs, not one.

## How to verify on your own model

```bash
python check_equivalence.py --selftest        # CPU only, a few seconds: checks the script itself

python check_equivalence.py --model mypkg.mymodule:build_model --weights ckpt.pt \
    --input crop.npy --dtype bf16 --mode fp32_taps --fp32-ref
```

By default (`--autocast auto`) a bf16/fp16 check runs the forward under `torch.autocast`, **which is how every gate in
`RESULTS.md` was run** (weights and input stay in fp32, convolutions run in the low precision). `--autocast off` gives the
older behaviour: the whole model and input are cast to the low precision, a different numeric regime that may print a larger
ratio for the same path (the one-tile printouts quoted in `KNOWN_ISSUES.md` were made this way; we did not re-run them under
autocast). Weights are loaded with `strict=True`: a key mismatch stops the run (`--ignore-keys` names exceptions).

It prints the maximum, mean and 99th-percentile absolute difference between stock and patched output, the fraction
of values within a tolerance, seconds per forward and the speed-up. With `--fp32-ref` it also runs a stock fp32
forward and reports each path's error against it. The number to look at in bf16/fp16 is the ratio
fast-error / stock-error; about 1.1 or lower means the patch adds no rounding noise beyond what bf16 already has. Note that
the recommended v8-in mode (`taps`) prints about 1.44 per layer; for v8-in the ruler was the downstream AUC/AP change and the
agreement of the output map, not this ratio (`KNOWN_ISSUES.md`).
Run it on a few real inputs. We called a configuration equivalent only if, on a whole real example (a whole
scroll winding, or a set of ~64-80 tiles/crops), the output stayed within one grey level for at least 99.5 % of
pixels and the downstream score (AUC / AP, or a correlation with the reference map) moved by at most 0.002. That is the first definition. A second one, used for the lasagna model and for the Hecate 2.4 um "later ruler" row, is the error-ratio rule (`evidence/preregs/2026-10-12_partner_fastpath_ruler.md`): summed error against an fp32 reference, fast over stock, one-sided bootstrap upper bound at most 1.10, on 16 tiles (lasagna) or 20 patches (Hecate). It measures bf16 rounding noise only.


## Results at a glance

Full table, sources and caveats in `RESULTS.md`.

| model | what | stock | with conv3d_fast | equivalence |
|---|---|---|---|---|
| Neural-tracing Copy model (**not reproducible from this package**: needs VC3D's `vc` binary, see `RUNBOOK.md` section 7) | one forward, 8x128x384x384 crop, bf16, `fp32_taps` | 75.6 s | 0.87 s | noise 0.81x stock (78 crops, no TTA, `--bbox-overlap 0.5`, one strip) |
| Copy model (same caveat) | whole copy job, 78 crops, no TTA | 6528 s | 354 s (about 18x) | as above |
| v8-in 9 µm ink | inference, tiles/s | 1x | 2.02x (+stem 2.49x, +stem+compile 4.18x) | 100 % of pixels within 1 grey level |
| v8-in 9 µm ink | one training epoch | 1x | 1.42x | final r 0.973 vs 0.965 between two stock seeds |
| 2.4 µm canonical ink | tile / whole render | 1x | about 2.0x | passed |
| Hecate 9.6 µm | render | 1x | 1.66x | passed |
| Hecate 2.4 µm | render | 1x | 1.92x | **failed** its pre-set bar (98.96 % of non-zero pixels within 1 grey level, bar 99 %) |
| Hecate 2.4 µm, later ruler | error against fp32, 20 patches | 1x | - | **passed** "no worse than stock rounding" (U 1.020, bar 1.10); a different question from the row above, which stands as failed; not a reading tolerance. `RESULTS.md` section 6 |
| Lasagna `predict3d` (3 slow layers) | 125 tiles, bf16 | 2099 s (re-run 2106 s) | **NOT OFFERED: GATE16-REAL failed (decided 2026-10-15).** Measurements only: `acc32_bias` 129 s / 122 s (16.3x with everything re-run / 17.2x with a reused stock time; the `fp32_taps` 10.3x **failed** its gate) | `acc32_bias` stayed within the 1.10 margin of stock bf16's error on 16 tiles (U 0.54), on 16 fresh tiles (U 0.52), and on one crop each of two other scrolls (PHerc0500P2 U 0.57, e2e 19.4x; PHerc0814 U 0.96, e2e 18.2x): one 256^3 crop of each of 3 scrolls, one model, 9.36 um scans only. Caveat (post hoc): most of the pooled gain over stock comes from tiles reading zero fill; on tiles of real PHerc0841 data it was about 7 % above stock's error (U 1.09), and the registered re-check on 16 all-real tiles then FAILED (GATE16-REAL, U 1.113 > 1.10, 2026-10-14). Each speed factor is one run. Tested setup = 3 layers only (`RUNBOOK.md` section 10). See `RESULTS.md` section 10 |

Reproducing two ink-prediction runs on PHerc0841 (Hecate on the 2.4 µm scan, and Hecate on the 9.6 µm scan) on the AMD machine gave a correlation of 0.837
for T (0.832 on NVIDIA) and 0.745 for H9 (0.746 on NVIDIA).

Speed and memory per model, stock vs the gated fast path (one forward, median of 3 after a warm-up, one R9700; details in
`RESULTS.md` section 9):

| model | stock | fast | stock / fast peak reserved GB |
|---|---|---|---|
| Copy model, 1 crop, bf16 | 75.6 s | 0.86 s (`fp32_taps`) | 10.0 / 20.3 |
| v8-in 9 µm, 16 tiles, fp16 | 1.72 s | 0.49 s (`+stem+compile`) | 13.7 / 10.7 |
| Canonical 2.4 µm, 1 tile, fp16 | 0.161 s | 0.082 s (`fp32_taps`) | 4.8 / 1.6 |
| Hecate 9.6 µm, 1 patch, bf16 | 0.043 s | 0.026 s (`fp32_taps`) | 1.7 / 0.9 |
| Hecate 2.4 µm, 1 patch, bf16 | 0.160 s | not offered (failed its gate) | 4.8 / - |

## On NVIDIA: leave it off

Sebastien ran the same check on an RTX 5070 Ti (2026-10-11; raw printouts in `evidence/nvidia/`). The outputs match stock, but the patch is slower and uses more
memory there, because cuDNN already runs 3-D convolutions fast:

| model | fast vs stock, within tolerance | speed (fast / stock) | peak memory (stock -> fast) | fast/stock error ratio (mean) |
|---|---|---|---|---|
| Hecate 9.6 µm | 100 % | 0.44x | 380 -> 618 MiB | 1.02 |
| canonical 2.4 µm | 100 % | 0.61x | 755 -> 1326 MiB | 1.18 |
| v8-in | 100 % | 0.82x | 3089 -> 3089 MiB | 1.16 |

So `conv3d_fast` is for AMD/ROCm cards only. On NVIDIA, use stock `nn.Conv3d`. The ROCm settings
(`ROCBLAS_USE_HIPBLASLT`, the allocator setting) do not apply to CUDA either.

## Known limits

- Measured on one machine and one ROCm/PyTorch combination. Other RDNA4 boards, drivers or torch versions may differ.
- The speed-ups are for 3x3x3 convolutions with several input channels. Models dominated by 1x1x1, depthwise
  or large single-channel kernels gain little.
- Peak memory can rise (the copy model with `taps` reached 28.6 GB reserved for one forward; the `fp32_taps` full
  copy job peaked at 20.3 GB, but the 8-variant test-time-augmentation call needed 28.9 GB, over a 28 GB budget).
- bf16 modes other than `fp32_taps` are not bit-compatible with stock, and none is bit-identical. On the copy model `taps` and `acc32_bias` add more rounding noise per layer than stock, although both passed the later error-ratio ruler there (`RESULTS.md` section 6).
- Hecate at 2.4 µm narrowly failed its first equivalence bar (fast vs stock within one grey level) but passed a later, different ruler
  (error against fp32 no worse than stock's); both are in `RESULTS.md` section 6, decide for yourself if that matters for your use.
- Lasagna: no speed-up is offered (decided 2026-10-15). `acc32_bias` on exactly the 3 slow layers passed its first four gates (16 tiles, 16 fresh tiles on that crop, then one crop each of two other scrolls; 9.36 um, one model, default overlap 64) but then failed the registered real-data re-check; `fp32_taps` failed its gate, so do not use it for lasagna. A global `enable(mode="acc32_bias")` rewrites all 70 eligible layers and is untested (`RUNBOOK.md` section 10). The gate runs read zero fill past the downloaded crop chunks; on real PHerc0841 data alone the fast path was about 7 % above stock's error (post hoc), and the registered re-check on 16 all-real tiles FAILED (GATE16-REAL, U 1.113 > 1.10; `RESULTS.md` section 10).
- Results of the equivalence checks are about rounding noise, not about whether the model reads text better.
- The Copy-model rows cannot be reproduced from this package (they need VC3D's `vc` binary; `RUNBOOK.md` section 7).
- Related work: villa PR #2055 ("spiral-fitting: read SQLite-backed track DBMs on Python < 3.13 (enables Windows + AMD ROCm
  runs)", opened 2026-10-12 by vincentl065) fixes a different problem (the track-store reader on Windows / Python 3.12) from our
  spiral patch (CUDA-only packages moved to a dependency group). The two are complementary; see `KNOWN_ISSUES.md`.

See `KNOWN_ISSUES.md` for hardware and driver problems we ran into.

## Tests

```bash
HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" python tests/test_conv3d_fast.py     # CPU only
```

## Credits

Built by the team Chris Thomas (GitHub Baz248) and Sebastien Chicoineau, with Claude. The models themselves belong to their authors (the Vesuvius
Challenge community and organisers); this repository contains none of their weights or code. All scroll data used in
our measurements is public Vesuvius Challenge data (PHerc0841, PHerc1447, PHerc0826, PHerc0500P2 and PHerc0814, named in `RESULTS.md`).
