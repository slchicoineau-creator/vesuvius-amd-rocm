#!/usr/bin/env python3
"""Download small public PHerc0841 test inputs (no login) and write the ready-to-run input tensors used in RUNBOOK.md.

Source: Vesuvius Challenge open data (CC BY-NC 4.0), segment w00 surface volumes of PHerc0841, region around
y 10208+, x 9280+ at 2.403 um (the region the ink-model checks used). Needs: numpy, zarr, fsspec, aiohttp, scipy, tifffile, imagecodecs.

  python fetch_pherc0841.py OUTDIR

Writes OUTDIR/hecate24_patch.npy  (1,1,64,256,256) float32 = uint8 / divisor 255  (Hecate 2.4 um input)
       OUTDIR/hecate96_patch.npy  (1,1,16,64,64)   float32                     (Hecate 9.6 um input)
       OUTDIR/canon_tile.npy      (1,1,62,256,256) float32 = clip(0..200)/200  (canonical 2.4 um model input)
       OUTDIR/R_sv24_crop.npy, R_sv96_crop.npy      the uint8 crops (for the hecate.py command line)
       OUTDIR/v8in_layers/00.tif..23.tif (for v8-in predict.py), OUTDIR/v8in_tiles.npy (4,24,64,64) float32
The copy model needs a surface and a different input (see RUNBOOK.md)."""
import os, sys
import numpy as np
import zarr
from scipy.ndimage import zoom

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0841/segments/20260220213127-w00/surface-volumes"
V24 = f"{S3}/2.403um-0.22m-77keV-volume-20260319124803.zarr"
V94 = f"{S3}/9.366um-1.2m-113keV-volume-20250821151531.zarr"
Y0, X0, N = 10208, 9280, 768          # region origin (2.403 um pixels) and crop size


def level0(url):
    g = zarr.open(url, mode="r", zarr_format=2)     # remote Zarr v2 over HTTPS (fsspec)
    return g["0"]


def main(out):
    os.makedirs(out, exist_ok=True)
    a = np.asarray(level0(V24)[0:109, Y0:Y0 + N, X0:X0 + N]); print("2.4 um crop", a.shape, a.dtype, a.mean())
    np.save(f"{out}/R_sv24_crop.npy", a)
    f = 2.403 / 9.366
    b = np.asarray(level0(V94)[0:28, int(Y0 * f):int((Y0 + N) * f), int(X0 * f):int((X0 + N) * f)]); print("9.4 um crop", b.shape, b.mean())
    c = np.clip(np.rint(zoom(b.astype(np.float32), 9.366 / 9.6, order=1)), 0, 255).astype(np.uint8)   # 9.6 um grid as Hecate expects
    np.save(f"{out}/R_sv96_crop.npy", c); print("9.6 um crop", c.shape)
    z = a.shape[0] // 2 - 32
    np.save(f"{out}/hecate24_patch.npy", (a[z:z + 64, 256:512, 256:512].astype(np.float32) / 255.0)[None, None])
    z = c.shape[0] // 2 - 8
    np.save(f"{out}/hecate96_patch.npy", (c[z:z + 16, 32:96, 32:96].astype(np.float32) / 255.0)[None, None])
    t = np.clip(a[1:63, 256:512, 256:512], 0, 200).astype(np.float32) / 200.0     # layers 1..62 as the canonical tile loop
    np.save(f"{out}/canon_tile.npy", t[None, None])
    # v8-in (9 um family) takes 24 layers of the 9.366 um surface volume: 24 numbered TIFFs for predict.py + 4 tiles of 64x64x24
    import tifffile
    ld = f"{out}/v8in_layers"; os.makedirs(ld, exist_ok=True)
    z = (b.shape[0] - 24) // 2
    for i in range(24):
        tifffile.imwrite(f"{ld}/{i:02d}.tif", b[z + i], compression="lzw")
    tiles = np.stack([b[z:z + 24, 64 + 64 * (i // 2):128 + 64 * (i // 2), 64 + 64 * (i % 2):128 + 64 * (i % 2)] for i in range(4)])
    np.save(f"{out}/v8in_tiles.npy", tiles.astype(np.float32) / 255.0)         # (4,24,64,64)
    print("wrote inputs to", out)


if __name__ == "__main__":
    main(sys.argv[1])
