# Runbook: from a fresh Ubuntu + ROCm 7.2 box to the Vesuvius 3-D models on public PHerc0841

Status: v1. Written and followed on one machine (2x Radeon AI PRO R9700, gfx1201, Ubuntu, kernel 7.0, ROCm 7.2.2)
on 2026-10-11. Each step says how far it was verified: **[GPU]** run on the card, **[CPU]** run without a GPU,
**[read]** taken from a script that ran in our earlier gate runs but not re-followed on a fresh install. All data is public
(Vesuvius Challenge open data, PHerc0841, CC BY-NC 4.0). Replace `PKG` by the folder holding this file.

## 0. Before you start

- ROCm 7.2 installed and the card visible (`rocm-smi` lists it). Installing ROCm itself is not covered here.
- Python from `uv` (https://docs.astral.sh/uv/). Python 3.12 or 3.14 both worked for the ink models; the organizers'
  current `vesuvius` package (villa `main`, 2026-10-11) requires **3.14**, and so do the commits we measured on.
- About 6 GB free for weights and test inputs.
- One card per process. Select it with `HIP_VISIBLE_DEVICES=0` (do not also filter with `ROCR_VISIBLE_DEVICES` on the same
  device; see `KNOWN_ISSUES.md`).

## 1. Environment variables used below

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:False      # on by default in some libraries; random faults on R9700
export HIP_VISIBLE_DEVICES=0
export ROCBLAS_USE_HIPBLASLT=1     # bf16 / fp16 jobs (all models below); use =0 for an fp32 job (README, section 1)
```

Never set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. First run of a new model or input shape can spend several
minutes (we saw up to ~13 min on a large 3-D model) building MIOpen kernels; later runs reuse the cache.

## 2. Python environment  [GPU]

Install torch AND torchvision from the ROCm index in one command, before anything else:

```bash
uv venv --python 3.14 venv && source venv/bin/activate
uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install numpy scipy zarr fsspec aiohttp s3fs tifffile imagecodecs opencv-python-headless safetensors \
               "huggingface_hub>=0.25" numcodecs einops timm pytorch-lightning numba
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
# -> 2.14.1+rocm7.2 7.2.53211 AMD Radeon Graphics
```

(versions that worked for us; zarr 3.4.1, numpy 2.5.2, numcodecs 0.16.5.) Do not `pip install vesuvius` from PyPI: it
resolved to a different package set, replaced zarr and the ROCm torchvision, and broke imports (`operator
torchvision::nms does not exist`). Use the source folder (step 7).

## 3. This package's own tests  [CPU, GPU]

```bash
HIP_VISIBLE_DEVICES="" python PKG/tests/test_conv3d_fast.py        # all "ok"
python PKG/check_equivalence.py --selftest                         # SELFTEST PASS
```

## 4. Weights (public Hugging Face repos)  [GPU]

```bash
hf download scrollprize/hecate                 --local-dir models/hecate --exclude "assets/*"          # 2 x 0.52 GB
hf download scrollprize/ink_canonical_2um      --local-dir models/ink_canonical_2um                    # 1.55 GB
hf download YoussefMoNader/ink-8um-v8in        --local-dir models/ink-8um-v8in --exclude "training/*"  # v8-in code + weights
hf download scrollprize/copy_displacement_latest --local-dir models/copy_displacement_latest            # copy model
```

The canonical model's code is `villa/ink-detection/optimized_inference/model_resnet3d_3d_decoder.py`
(`git clone https://github.com/ScrollPrize/villa`; we used commit `f4570bfa6`, the one our gates ran on).

## 5. PHerc0841 test inputs  [GPU]

```bash
python PKG/examples/fetch_pherc0841.py data
```

Reads a 768 x 768 crop of the w00 surface volumes straight from the S3 bucket over HTTPS
and writes ready-to-run inputs: Hecate 2.4 and 9.6 um patches and crops, a canonical-model tile, v8-in layers
(`data/v8in_layers/00.tif..23.tif`) and tiles. Data sources and region are named in the script.

## 6. Run each ink model, stock vs conv3d_fast

Use `check_equivalence.py` with the builders in `PKG/examples/builders.py` (it times both paths and compares them with a
stock fp32 forward; add `--repeat 3`):

```bash
export PYTHONPATH=PKG/examples HECATE_DIR=$PWD/models/hecate V8IN_DIR=$PWD/models/ink-8um-v8in \
       CANON_CKPT=$PWD/models/ink_canonical_2um/r152_3ddec_v2_l5_epoch13.ckpt \
       VILLA_INFERENCE_DIR=/path/to/villa/ink-detection/optimized_inference
CE="python PKG/check_equivalence.py --fp32-ref"
$CE --dtype bf16 --mode fp32_taps --model builders:hecate96   --input data/hecate96_patch.npy     # Hecate 9.6 um  [GPU]
$CE --dtype fp16 --mode fp32_taps --model builders:canonical  --input data/canon_tile.npy         # canonical 2.4 um [GPU]
$CE --dtype fp16 --mode taps      --model builders:v8in       --input data/v8in_tiles.npy         # v8-in 9 um [GPU]
```

Mode per model = the path that passed its gate (`RESULTS.md`): `fp32_taps` for Hecate 9.6 and the canonical model,
`taps` (the default) for v8-in. **Hecate 2.4 um: run it stock** (`HECATE-FAST-GATE` failed). A single patch or tile is a
noisy test: on our single inputs the fast/stock error ratio printed 1.07 (Hecate 9.6), 1.43 (canonical, p99) and 1.44
(v8-in, p99), and 1.95 for Hecate 2.4; the gates pooled 64 to 78 inputs and a whole render before calling a path
equivalent. Treat one-input output as a smoke test, not a verdict.

Whole-image runs with the models' own command lines, no source edits (`run_with_fast.py` turns the patch on):

```bash
H=models/hecate     # Hecate 9.6 um on the PHerc0841 crop   [GPU]
python $H/hecate.py --checkpoint $H/hecate_9.6um.pth --input data/R_sv96_crop.npy --spacing-um 9.6 --precision bf16 --output stock.png
python PKG/examples/run_with_fast.py --mode fp32_taps $H/hecate.py --checkpoint $H/hecate_9.6um.pth \
       --input data/R_sv96_crop.npy --spacing-um 9.6 --precision bf16 --output fast.png     # 100 % of pixels within 1 grey level of stock, max 1
python $H/hecate.py --checkpoint $H/hecate_2.4um.pth --input data/R_sv24_crop.npy --spacing-um 2.4 --precision bf16 --output h24.png   # stock only [read]

python models/ink-8um-v8in/predict.py --layers data/v8in_layers --output v8_stock.npy         # v8-in  [GPU]
python PKG/examples/run_with_fast.py --stem models/ink-8um-v8in/predict.py --layers data/v8in_layers --output v8_fast.npy
    # conv3d_fast + stem rewrite (gate G2s); max |difference| 0.0013 in probability on this 197 x 197 crop
```

The canonical model has no one-line command in a package we ship: its tile loop is `infer_crop.py`-style code around
`model_resnet3d_3d_decoder.load_model` (see `builders.py` for loading). On the speed table the model is timed on one
62 x 256 x 256 tile, and the whole-render gate (`RESULTS.md` section 5) used the villa tile loop (256 tiles, stride 128).

The v8-in `+stem+compile` path (4.18x in the whole-winding gate) is `torch.compile(model)` after
`conv3d_fast.enable(stem=True)` on the loaded model; our timing driver is `measurements/bench_forward.py`.

## 7. Copy model (neural tracing "Copy")  [CPU part only]

The organizers' tool is `vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py` in villa. What we verified on a
fresh install:

```bash
git clone https://github.com/ScrollPrize/villa && cd villa && git checkout f4570bfa6 && cd vesuvius   # the commit we measured
PYTHONPATH=src python -c "from vesuvius.neural_tracing.inference import infer_rowcol_triplet_wraps"
# -> stops with: ModuleNotFoundError: No module named 'vc'     (every other import of the tool resolves)  [CPU]
```

`vc` is the C++ Python binding of Volume Cartographer (VC3D, GPLv3). We did not build it on Linux/ROCm. Our measurements
used a 57-line pure-Python stand-in that provides only what that script calls (`set_chunk_cache_budget`, a volume
object reading the public Zarr over HTTPS); it is **not** part of this package. On villa `main` (2026-10-11,
`d63d3e968`) the script's `vc` calls changed (`vc.Volume.open_url(path, cache_root=...)`), and `main` requires Python
3.14, so a stand-in would need updating. **The copy-model rows in `RESULTS.md` are not reproducible from this package** (they need VC3D's `vc` binary, which we did not
build on ROCm and do not ship a substitute for). What the tool needs, from the run we timed [read]:

```bash
python PKG/examples/run_with_fast.py --mode fp32_taps -m vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps \
  --tifxyz-path SURFACE_DIR --volume-path s3://vesuvius-challenge-open-data/PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr \
  --checkpoint-path models/copy_displacement_latest/copy_displacement_latest.pth --volume-scale 0 --no-compile --batch-size 1 \
  --tifxyz-voxel-size-um 9.366 --out-dir OUT --output-prefix NAME --volume-cache-dir CACHE --merge-temp-dir TMP \
  --bbox-overlap 0.5 --no-tta
```

with `ROCBLAS_USE_HIPBLASLT=1` and bf16 (the checkpoint's own setting). `SURFACE_DIR` is a tifxyz surface (`x.tif`, `y.tif`,
`z.tif`, `meta.json`) in 9.366 um voxel coordinates. Ours was a strip of the **public** segment s022814
(`PHerc0841/segments/20260221022814-auto_grown_20260220174252405`, native 9.366 um tifxyz): rows 0-177, columns 81-102 of its
188 x 245 grid (3,809 vertices), with normals computed on the full grid.
Use `fp32_taps`: the other modes add more bf16 noise than stock on this model. The 8-variant test-time-augmentation call
needs 28.9 GB reserved, which fits a 32 GB card only with nothing else on it.

## 8. Known failures on the way

See `KNOWN_ISSUES.md`. Those you hit first: CUDA torchvision replacing the ROCm one (step 2), missing `imagecodecs`
(`<COMPRESSION.LZW> requires the 'imagecodecs' package` when reading layer TIFFs), the `vc` module (step 7), and a
`(null): No such file or directory` line printed four times at every ROCm process start (harmless in all our runs; cause
not investigated).

## 9. Speed and memory table

`RESULTS.md` section 9 (one forward per model, stock vs gated fast path). Reproduce a row with `measurements/bench_forward.py`
(team-internal driver: it reads our gate inputs; shown so the method can be audited) or with `check_equivalence.py` above.

## 10. Lasagna `predict3d` fast path (3 layers, `acc32_bias`): experimental, failed its real-data re-check, not recommended  [read]

**Not recommended.** This fast path failed its registered real-data re-check (GATE16-REAL, 2026-10-14: error ratio U 1.113 against the 1.10 bar), so no speed-up is offered for lasagna. The recipe below is kept for experiments only. See `RESULTS.md` section 10 and `evidence/lasagna_gate16_real.json`.

**[read]**: taken from the gate scripts that ran (`las_gate16c.py`, `evidence/preregs/2026-10-13_partner_lasagna_gate16c_confirm.md`); not re-followed on a fresh install.
Read `RESULTS.md` section 10 first, including the zero-fill caveat: on real PHerc0841 data alone the tested path was about 7 % above stock's bf16 error (post hoc, just inside the 1.10 margin), and the registered re-check on 16 all-real PHerc0841 tiles (GATE16-REAL, 2026-10-14) FAILED (U 1.113 > 1.10). Treat it as unvalidated on real data.

**What was tested, exactly:** `conv3d_fast` mode `acc32_bias` on **three** Conv3d layers only,
`shared_encoder.stages.0.blocks.0.conv1.conv`, `shared_encoder.stages.0.blocks.0.conv2.conv` and `shared_decoder.stages.5.convs.0.conv`; **every other Conv3d
pinned to stock** (79 of the 82). Use `fp32_taps` nowhere in lasagna: it failed (2.06x stock's error).

**Warning:** `conv3d_fast.enable(mode="acc32_bias")` or `examples/run_with_fast.py --mode acc32_bias` on its own switches **all 70 eligible 3x3x3 layers**. That configuration
was never gated (untested), and our profile (`fp32_taps`, the only mode profiled there) measured 0.1-0.4x of stock's speed on the 67 layers outside the three. Do not use it for lasagna.

```python
# lasagna_fast_entry.py  -- replaces only load_model; everything else is predict3d as shipped
import sys, types, torch.nn as nn, conv3d_fast as c3        # PKG on PYTHONPATH
import preprocess_cos_omezarr as p                          # lasagna folder of villa on PYTHONPATH
FAST_LAYERS = ["shared_encoder.stages.0.blocks.0.conv1.conv", "shared_encoder.stages.0.blocks.0.conv2.conv",
               "shared_decoder.stages.5.convs.0.conv"]
A = p.LasagnaCosPredict3DAdapter; load0 = A.load_model
def load_model(self, *, device):
    m = load0(self, device=device)
    c3.enable(mode="acc32_bias"); mods = dict(m.named_modules())
    for n in FAST_LAYERS:
        assert isinstance(mods[n], nn.Conv3d) and c3.use_fast(mods[n].weight, mods[n].groups, mods[n].dilation, mods[n].padding_mode, mods[n].padding), n
    for n, mod in mods.items():                              # pin every other Conv3d to stock
        if isinstance(mod, nn.Conv3d) and n not in FAST_LAYERS:
            mod._conv_forward = types.MethodType(c3._orig_conv_forward, mod)
    return m
A.load_model = load_model
if __name__ == "__main__":
    sys.exit(p.cli_main())
```

```bash
export ROCBLAS_USE_HIPBLASLT=1 PYTORCH_ALLOC_CONF=expandable_segments:False HIP_VISIBLE_DEVICES=0
python lasagna_fast_entry.py predict3d --input vol.zarr/0 --output out/pred.lasagna.json --unet-checkpoint lasagna_model_current.pt \
  --crop-xyzwhd X Y Z 256 256 256 --device cuda --scaledown 2 --cos-scaledown 1 --flush-workers 0 --accumulator-workers 0
```

Defaults that were measured: tile size 192 (from the checkpoint), `--overlap 64` (stride 128), `--border 16`; other overlaps are not shown. Our runs used `--no-download` on a local zarr
that held only the crop's chunks: **tiles that reach past the downloaded chunks read zeros, so give predict3d every chunk the tiles read** (about 700 voxels per axis for a 256^3 crop:
from 144 before to 560 after the 128-snapped crop start). Verify on your own tiles with `check_equivalence.py`-style comparison against an fp32 run before relying on it.

