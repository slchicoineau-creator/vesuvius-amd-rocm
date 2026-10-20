"""AMD package v1 speed/VRAM table (evidence/preregs/2026-10-11_partner_rocm_ports.md): one forward pass per model and path.
Note: the input and output folders it uses (see R0 and the paths below) and run_all.sh are our own working files, not in this
repository, so it does not run here as shipped. It is kept as the record of how RESULTS.md section 9 was made.
Method (frozen): 1 warm-up forward, then median of 3 timed forwards; torch.cuda.max_memory_allocated/reserved and a
rocm-smi VRAM peak (sampled every ~0.3 s from a thread, minus the idle usage before the run). Fresh process per row.
Usage (repo root, one card, env set by run_all.sh):  bench.py MODEL ARM [BATCH]
  MODEL: copy | v8in | canon | hecate24 | hecate96
  ARM:   stock | fast   (fast = the gated fast path of that model; see PATHS)
Writes work/amd_pkg/out/<model>_<arm>_b<batch>.json. Reuses the gate scripts' loaders and inputs read-only."""
import json, os, statistics, subprocess, sys, threading, time
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R0 = os.path.abspath(os.path.join(H, "..", ".."))
GB = 1024 ** 3
PATHS = {"copy": {"stock": "stock", "fast": "conv3d_fast+fp32_taps"},
         "v8in": {"stock": "stock", "fast": "conv3d_fast+stem+compile"},
         "canon": {"stock": "stock", "fast": "conv3d_fast+fp32_taps"},
         "hecate24": {"stock": "stock", "fast": "conv3d_fast+fp32_taps"},   # NOT gated (HECATE-FAST-GATE failed): only run for the record if asked
         "hecate96": {"stock": "stock", "fast": "conv3d_fast+fp32_taps"}}


class Smi(threading.Thread):
    def __init__(self, card):
        super().__init__(daemon=True); self.card, self.peak, self.stop_, self.base = str(card), 0.0, False, None
    def read(self):
        o = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json", "-d", self.card], capture_output=True, text=True).stdout
        d = json.loads(o); k = [v for v in d.values() if isinstance(v, dict)][0]
        return float(k["VRAM Total Used Memory (B)"]) / GB
    def run(self):
        while not self.stop_:
            try:
                u = self.read(); self.base = u if self.base is None else self.base; self.peak = max(self.peak, u)
            except Exception:
                pass
            time.sleep(0.2)


def timed(fn, n_warm=1, n=3):
    import torch
    for _ in range(n_warm):
        fn(); torch.cuda.synchronize()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return ts


def b_copy(arm, batch):
    import torch, shutil
    sys.path[:0] = [os.path.join(R0, "work", "ink", "fastcopy")]
    import fc
    scratch = os.path.join(H, "fc_scratch"); os.makedirs(scratch, exist_ok=True)
    fc.OUT = lambda *f: os.path.join(scratch, *f)        # never touch the frozen FASTCOPY-GATE files (plan.json, copies/)
    if not os.path.isdir(fc.OUT("surf", fc.NAME)): fc.strip()
    M, dt, dh = fc.tf9.tool(); cap = {}
    class Stop(Exception): pass
    def pred(args, st, x, use_tta=False, profiler=None):
        cap["x"] = x.detach().clone(); cap["st"] = st; raise Stop
    p0 = M.predict_displacement; M.predict_displacement = pred; tmp = os.path.join(fc.CACHE, "tmp_amdpkg"); os.makedirs(tmp, exist_ok=True)
    try:
        M.run(M.parse_args(fc.base_argv("amdpkg", tmp)))
    except Stop:
        pass
    finally:
        M.predict_displacement = p0; shutil.rmtree(tmp, ignore_errors=True)
    x, st = cap["x"], cap["st"]; assert st["amp_enabled"] and st["amp_dtype"] == torch.bfloat16
    fc.set_path("stock" if arm == "stock" else "conv3d_fast+fp32_taps")
    meta = dict(input=list(x.shape), dtype="bf16 autocast (checkpoint amp)", what="one forward, no TTA", conv_path=fc.conv_path())
    def f():
        with torch.inference_mode(): dh.predict_displacement(None, st, x, use_tta=False)
    return f, meta


def b_v8in(arm, batch):
    import torch
    sys.path[:0] = [os.path.join(R0, "work/v8/models/ink-8um-v8in"), os.path.join(R0, "work/fl/v8")]
    from ink8um import InkDetector, read_stack
    import fastinfer
    st = read_stack(os.path.join(R0, "work/v8/data/pherc1447/w058/layers"))[..., ::-1]
    ys, xs = np.nonzero(st.max(2) > 0); y0, x0 = int(np.median(ys)), int(np.median(xs))
    tiles = np.stack([st[y0 + (i // 8) * 64:y0 + (i // 8) * 64 + 64, x0 + (i % 8) * 64:x0 + (i % 8) * 64 + 64].transpose(2, 0, 1)
                      for i in range(64)]).astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(tiles[:batch])).cuda()
    m = InkDetector.from_pretrained(os.path.join(R0, "work/v8/models/ink-8um-v8in")).cuda().eval()
    m, path = fastinfer.gated_inference_model(m, PATHS["v8in"][arm])
    meta = dict(input=list(x.shape), dtype="fp16 autocast", what=f"one forward of {batch} tiles (64x64x24, upsampled to 96x256x256 inside)", conv_path=path, tiles=batch)
    def f():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16): m(x)
    return f, meta


def b_canon(arm, batch):
    import torch, types
    sys.path[:0] = [os.path.join(R0, "work", "ink", "canongate")]
    import canon_gate as cg
    plan = json.load(open(os.path.join(cg.H, "plan.json"))); t = plan["tiles"][0]
    vol = np.load(cg.VOL, mmap_mode="r"); s0 = cg.window(vol.shape[0], t["offset"])
    x = torch.from_numpy(cg.tile_input(vol[s0:s0 + cg.NFR], t["y"], t["x"], t["dir"])).cuda().repeat(batch, 1, 1, 1, 1)
    _, w = cg.load_real_model(torch.device("cuda")); path = cg.set_arm("S16" if arm == "stock" else "F16T")
    am = cg.ArmModel(w, "S16" if arm == "stock" else "F16T")
    meta = dict(input=list(x.shape), dtype="fp16 autocast", what=f"one forward of {batch} tile(s) 62x256x256", conv_path=path)
    def f():
        with torch.inference_mode(), torch.autocast("cuda"): am.forward(x)
    return f, meta


def b_hecate(scale):
    def b(arm, batch):
        import torch
        sys.path[:0] = [os.path.join(R0, "work", "ink", "hecatefast")]
        import hg
        hecate, c, v = hg.load(scale); m = hg.model_for(scale, hecate)
        pl = json.load(open(hg.OUT("plan.json")))[scale]; y, x0 = pl["patches"][len(pl["patches"]) // 2]
        d, p, _ = m.patch_size; zs = hg.central(m.patch_size, v)
        raw = np.ascontiguousarray(v[zs, y:y + p, x0:x0 + p])
        img = torch.from_numpy(raw.astype(np.float32) / m.divisor)[None, None].cuda().repeat(batch, 1, 1, 1, 1)
        valid = torch.from_numpy(raw.any(0))[None, None, None].expand(batch, 1, d, -1, -1).cuda()
        path = hg.set_arm("STOCK" if arm == "stock" else "F32T")
        meta = dict(input=list(img.shape), dtype="bf16 autocast", what=f"one forward of {batch} patch(es) {d}x{p}x{p}", conv_path=path)
        def f():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): m(img, valid)
        return f, meta
    return b


BUILD = {"copy": b_copy, "v8in": b_v8in, "canon": b_canon, "hecate24": b_hecate("24"), "hecate96": b_hecate("96")}

if __name__ == "__main__":
    model, arm = sys.argv[1], sys.argv[2]; batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    import torch
    assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:False"
    smi = Smi(os.environ["HIP_VISIBLE_DEVICES"]); smi.start(); time.sleep(0.6)
    f, meta = BUILD[model](arm, batch)
    smi.peak = 0.0; ts = timed(f)
    smi.stop_ = True; time.sleep(0.3)
    r = dict(model=model, arm=arm, batch=batch, hipblaslt=os.environ.get("ROCBLAS_USE_HIPBLASLT"), times_s=[round(t, 4) for t in ts],
             median_s=round(statistics.median(ts), 4), max_memory_allocated_gb=round(torch.cuda.max_memory_allocated() / GB, 2),
             max_memory_reserved_gb=round(torch.cuda.max_memory_reserved() / GB, 2), rocm_smi_peak_used_gb=round(smi.peak, 2),
             rocm_smi_idle_before_gb=round(smi.base or 0, 2), device=torch.cuda.get_device_name(0), torch=torch.__version__, hip=torch.version.hip,
             date=time.strftime("%F %T"), **meta)
    json.dump(r, open(os.path.join(H, "out", f"{model}_{arm}_b{batch}.json"), "w"), indent=1); print(json.dumps(r))
