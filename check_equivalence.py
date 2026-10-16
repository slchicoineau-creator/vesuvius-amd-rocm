#!/usr/bin/env python3
"""Compare a model's output with stock nn.Conv3d vs with conv3d_fast, on YOUR model and YOUR input.

Real use (GPU, ROCm or CUDA):
  python check_equivalence.py --model mypkg.mymodule:build_model --weights ckpt.pt \
      --input crop.npy --dtype bf16 --mode fp32_taps --fp32-ref

  --model     "module:callable" that returns an nn.Module (called with no arguments). The module must be importable
              (put its folder on PYTHONPATH).
  --weights   optional state_dict file (torch.load); loaded with strict=True (a missing or unexpected key stops the run).
              If a checkpoint legitimately carries extra/absent keys, list them by exact name with --ignore-keys a,b.
  --input     .npy or .pt/.pth tensor, shaped exactly as the model expects (batch included); or --shape 1,1,96,256,256
              for random normal data (fine for speed, weak for quality: prefer a real crop).
  --dtype     fp32 | bf16 | fp16   (with --autocast off: model weights and input are cast to it; with autocast: it names
              the autocast precision and the model stays fp32)
  --autocast  auto | off | bf16 | fp16.  auto (default) = run under torch.autocast in --dtype when --dtype is bf16/fp16, which is
              how every gate in RESULTS.md was run. off = the older behaviour: cast the whole model and input to --dtype.
  --mode      taps | acc32_bias | taps_as_channels | fp32_taps   (see README)
  --fp32-ref  also run a stock fp32 forward and report, for stock and for fast, the error against it. The ratio
              (fast error / stock error) is the number that matters in bf16/fp16: about 1.0 or lower means the fast path adds
              no more rounding noise than the stock path already has. In fp32 pure stock-vs-fast differences are tiny.

Self-test (CPU only, random tensors, no GPU touched):
  python check_equivalence.py --selftest

What it prints: max abs diff, mean abs diff, p99 abs diff, fraction of values within atol+rtol*|ref|, seconds per
forward for both paths, speed-up, peak GPU memory. The script does not decide pass/fail for you except in --selftest.
"""
import argparse, importlib, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv3d_fast as cf  # noqa: E402

DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "fp64": torch.float64}


def _flat(out):
    """model output -> list of tensors (handles tensor / tuple / list / dict)."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        return [v for v in out.values() if isinstance(v, torch.Tensor)]
    if isinstance(out, (list, tuple)):
        r = []
        for o in out:
            r += _flat(o)
        return r
    return []


def _force_patch(mode):
    """like cf.enable(mode) but without the 'only for GPU tensors' condition, so CPU tensors take the fast path too
    (used by --selftest; on a GPU use cf.enable)."""
    cf._MODE["m"] = mode

    def fwd(self, input, weight, bias):
        if cf.use_fast(weight, self.groups, self.dilation, self.padding_mode, self.padding):
            _force_patch.calls += 1
            return cf.MODE_FNS[mode](input, weight, bias, tuple(self.stride), tuple(self.padding))
        return cf._orig_conv_forward(self, input, weight, bias)
    nn.Conv3d._conv_forward = fwd


_force_patch.calls = 0


def load_strict(model, sd, ignore=()):
    """Load `sd` into `model` with strict=True semantics. Keys named in `ignore` (exact names) may be missing or unexpected;
    any other mismatch raises RuntimeError, so a check never runs silently on randomly initialised layers."""
    ignore = set(ignore)
    res = model.load_state_dict(sd, strict=False)
    bad_m = [k for k in res.missing_keys if k not in ignore]
    bad_u = [k for k in res.unexpected_keys if k not in ignore]
    if bad_m or bad_u:
        raise RuntimeError(f"state_dict mismatch: missing {bad_m[:10]}, unexpected {bad_u[:10]}")
    return res


def _run(model, x, device, autocast, fast, mode, force_cpu=False, repeat=3):
    """returns (outputs as fp64 CPU tensors, seconds per forward (best of `repeat` after one warm-up), peak MiB)."""
    cf.disable()
    if fast:
        _force_patch(mode) if force_cpu else cf.enable(mode=mode)
    is_gpu = device.type == "cuda"
    if is_gpu:
        torch.cuda.reset_peak_memory_stats()
    ts = []
    out = None
    with torch.inference_mode():
        for i in range(repeat + 1):
            if is_gpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.bfloat16 if autocast == "bf16" else torch.float16,
                                enabled=bool(autocast)):
                out = model(x)
            if is_gpu:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    cf.disable()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if is_gpu else 0.0
    return [o.detach().double().cpu() for o in _flat(out)], min(ts[1:]), peak


def _diff(a, b, atol, rtol):
    d = (a - b).abs()
    tol = atol + rtol * b.abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "p99_abs": float(torch.quantile(d.flatten()[:16_000_000].float(), 0.99)),
            "frac_within_tol": float((d <= tol).float().mean()),
            "ref_mean_abs": float(b.abs().mean())}


def _report(name, a_list, b_list, atol, rtol):
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        r = _diff(a, b, atol, rtol)
        print(f"  {name} [output {i}, shape {tuple(a.shape)}]: max {r['max_abs']:.3e}  mean {r['mean_abs']:.3e}  "
              f"p99 {r['p99_abs']:.3e}  within tol {100 * r['frac_within_tol']:.3f} %  (mean |ref| {r['ref_mean_abs']:.3e})")
    return [_diff(a, b, atol, rtol) for a, b in zip(a_list, b_list)]


def compare(model, x, device, dtype, mode, atol, rtol, fp32_ref, force_cpu=False, repeat=3, autocast=None):
    # autocast: None = the model and input are already cast to `dtype`; "bf16"/"fp16" = run under torch.autocast (the gates' regime)
    stock, t_s, m_s = _run(model, x, device, autocast, False, mode, force_cpu, repeat)
    fast, t_f, m_f = _run(model, x, device, autocast, True, mode, force_cpu, repeat)
    print(f"\nstock vs conv3d_fast[{mode}] ({dtype}, {'autocast' if autocast else 'whole-model cast'}):")
    res = _report("fast - stock", fast, stock, atol, rtol)
    print(f"  time per forward: stock {t_s:.3f} s, fast {t_f:.3f} s  -> {t_s / max(t_f, 1e-12):.2f}x"
          + (f"   peak memory stock {m_s:.0f} MiB, fast {m_f:.0f} MiB" if device.type == "cuda" else ""))
    out = {"stock_vs_fast": res, "t_stock": t_s, "t_fast": t_f}
    if fp32_ref:
        import copy
        ref_model = copy.deepcopy(model).float()
        ref, _, _ = _run(ref_model, x.float(), device, None, False, mode, force_cpu, 1)
        print("\nerror against a stock fp32 forward (lower is better):")
        e_s = _report("stock", stock, ref, atol, rtol)
        e_f = _report("fast ", fast, ref, atol, rtol)
        for i, (s, f) in enumerate(zip(e_s, e_f)):
            print(f"  output {i}: fast/stock error ratio  mean {f['mean_abs'] / max(s['mean_abs'], 1e-30):.3f}  "
                  f"p99 {f['p99_abs'] / max(s['p99_abs'], 1e-30):.3f}   (<= ~1.1 means no extra rounding noise)")
        out["ratio"] = [(f["mean_abs"] / max(s["mean_abs"], 1e-30), f["p99_abs"] / max(s["p99_abs"], 1e-30))
                        for s, f in zip(e_s, e_f)]
    return out


def selftest():
    torch.manual_seed(0)
    dev = torch.device("cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv3d(1, 6, 3, padding=1); self.b = nn.Conv3d(6, 6, 3, padding=1)
            self.c = nn.Conv3d(6, 8, 3, stride=2, padding=1); self.d = nn.Conv3d(8, 2, 1)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.d(self.act(self.c(self.act(self.b(self.act(self.a(x)))))))

    model = Net().double().eval()
    x = torch.randn(1, 1, 10, 18, 18, dtype=torch.float64)
    ok = True
    for mode in cf.MODE_FNS:
        _force_patch.calls = 0
        r = compare(model, x, dev, "fp64", mode, atol=1e-9, rtol=1e-9, fp32_ref=False, force_cpu=True, repeat=1)
        worst = max(d["max_abs"] for d in r["stock_vs_fast"])
        frac = min(d["frac_within_tol"] for d in r["stock_vs_fast"])
        good = worst < 1e-10 and frac == 1.0 and _force_patch.calls > 0   # calls > 0: the fast path really ran
        print(f"  -> selftest {mode}: {'PASS' if good else 'FAIL'} (fp64, max abs diff {worst:.2e}, {_force_patch.calls} decomposed conv calls)")
        ok &= good
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="CPU-only check on a tiny random network")
    p.add_argument("--model"); p.add_argument("--weights")
    p.add_argument("--input"); p.add_argument("--shape")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--mode", default="taps", choices=list(cf.MODE_FNS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--atol", type=float, default=1e-2); p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--fp32-ref", action="store_true")
    p.add_argument("--autocast", default="auto", choices=["auto", "off", "bf16", "fp16"])
    p.add_argument("--ignore-keys", default="", help="comma-separated exact key names allowed to be missing/unexpected in --weights")
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.model or not (a.input or a.shape):
        p.error("--model and one of --input/--shape are required (or use --selftest)")
    mod_name, fn_name = a.model.split(":")
    sys.path.insert(0, os.getcwd())
    model = getattr(importlib.import_module(mod_name), fn_name)()
    if a.weights:
        sd = torch.load(a.weights, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        print("load_state_dict:", load_strict(model, sd, [k for k in a.ignore_keys.split(",") if k]))
    if a.input:
        x = torch.from_numpy(np.load(a.input)) if a.input.endswith(".npy") else torch.load(a.input)
    else:
        x = torch.randn(*[int(s) for s in a.shape.split(",")])
    device = torch.device(a.device)
    if device.type != "cuda":
        print("note: conv3d_fast only patches GPU tensors; on CPU this compares stock with stock. Use --selftest for CPU.")
    dt = DT[a.dtype]
    ac = a.autocast
    if ac == "auto":
        ac = a.dtype if a.dtype in ("bf16", "fp16") else "off"
    if ac == "off":
        autocast = None
        model = model.to(device).eval().to(dt)
        x = x.to(device).to(dt)
    else:
        autocast = ac
        model = model.to(device).eval().float()
        x = x.to(device).float()
    compare(model, x, device, a.dtype, a.mode, a.atol, a.rtol, a.fp32_ref, repeat=a.repeat, autocast=autocast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
