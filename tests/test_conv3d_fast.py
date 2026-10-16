"""CPU tests for conv3d_fast (no GPU needed).
Run from the repo root:  python tests/test_conv3d_fast.py    (or: python -m pytest -q tests)"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import conv3d_fast as cf  # noqa: E402

MODES = [("taps", False), ("taps", True)] + [(m, False) for m in cf.MODE_FNS if m != "taps"]
STRIDES = [(1, 1, 1), (2, 2, 2), (1, 2, 2), (2, 1, 1)]


def _run(mode, acc32, x, w, b, s, p):
    old = cf._ACC32["on"]; cf._ACC32["on"] = acc32
    try:
        return cf.MODE_FNS[mode](x, w, b, s, p)
    finally:
        cf._ACC32["on"] = old


def test_fp64_exact_outputs_and_grads():
    """3x3x3, Cin > 1: every mode equals F.conv3d in fp64 (output, input grad, weight grad, bias grad)."""
    g = torch.Generator().manual_seed(0); worst = 0.0
    for mode, acc32 in MODES:
        for s in STRIDES:
            for use_b in (True, False):
                x = torch.randn(2, 5, 7, 9, 8, dtype=torch.float64, generator=g)
                w = torch.randn(4, 5, 3, 3, 3, dtype=torch.float64, generator=g)
                b = torch.randn(4, dtype=torch.float64, generator=g) if use_b else None
                p = (1, 1, 1)
                xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
                br = b.clone().requires_grad_() if use_b else None
                xf, wf = x.clone().requires_grad_(), w.clone().requires_grad_()
                bf = b.clone().requires_grad_() if use_b else None
                yr = F.conv3d(xr, wr, br, s, p); yf = _run(mode, acc32, xf, wf, bf, s, p)
                assert yf.shape == yr.shape and yf.dtype == torch.float64, (mode, s, yf.shape, yr.shape, yf.dtype)
                go = torch.randn(yr.shape, dtype=torch.float64, generator=g)
                (yr * go).sum().backward(); (yf * go).sum().backward()
                pairs = [(yf, yr), (xf.grad, xr.grad), (wf.grad, wr.grad)] + ([(bf.grad, br.grad)] if use_b else [])
                for a, r in pairs:
                    e = float(((a - r).abs().max() / (r.abs().max() + 1e-30)).detach()); worst = max(worst, e)
                    assert e < 1e-12, (mode, acc32, s, use_b, e)
    print("fp64 worst relative error over modes/strides/grads:", worst)


def test_gradcheck_tiny():
    g = torch.Generator().manual_seed(1)
    for mode, acc32 in MODES:
        for s in ((1, 1, 1), (2, 2, 2)):
            x = torch.randn(1, 2, 5, 5, 5, dtype=torch.float64, generator=g, requires_grad=True)
            w = torch.randn(3, 2, 3, 3, 3, dtype=torch.float64, generator=g, requires_grad=True)
            b = torch.randn(3, dtype=torch.float64, generator=g, requires_grad=True)
            assert torch.autograd.gradcheck(lambda x, w, b: _run(mode, acc32, x, w, b, s, (1, 1, 1)), (x, w, b)), (mode, s)


def test_policy_leaves_other_layers_alone():
    """use_fast: only 3x3x3, groups=1, dilation 1, zero padding, Cin > 1. 1x1x1 and single-channel layers stay stock."""
    def pol(cout, cin, k, groups=1, dilation=(1, 1, 1), pm="zeros", padding=(1, 1, 1)):
        return cf.use_fast(torch.zeros(cout, cin, *k), groups, dilation, pm, padding)
    assert pol(4, 5, (3, 3, 3))
    assert not pol(4, 5, (1, 1, 1))                       # pointwise: stock is faster
    assert not pol(4, 1, (3, 3, 3))                       # single input channel
    assert not pol(4, 1, (7, 7, 7))                       # 7x7x7 stem
    assert not pol(4, 5, (3, 3, 3), dilation=(2, 1, 1))
    assert not pol(4, 5, (3, 3, 3), pm="reflect")
    assert not pol(4, 5, (3, 3, 3), padding="same")
    assert not cf.use_fast(torch.zeros(4, 2, 3, 3, 3), 2, (1, 1, 1), "zeros", (1, 1, 1))   # groups != 1
    # stem rewrite is off unless asked for
    assert not cf.use_stem(torch.zeros(4, 1, 7, 7, 7), 1, (1, 1, 1), "zeros", (3, 3, 3))


def test_stem_rewrite_exact_in_fp64():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 1, 12, 14, 14, dtype=torch.float64, generator=g)
    w = torch.randn(3, 1, 7, 7, 7, dtype=torch.float64, generator=g)
    b = torch.randn(3, dtype=torch.float64, generator=g)
    y = cf.stem_as_2d(x, w, b, (1, 2, 2), (3, 3, 3)); r = F.conv3d(x, w, b, (1, 2, 2), (3, 3, 3))
    assert float((y - r).abs().max()) < 1e-10


def test_enable_default_and_bad_mode():
    cf.enable(mode="fp32_taps"); cf.enable()
    assert cf._MODE["m"] == "taps" and not cf._ACC32["on"] and not cf._STEM["on"]
    try:
        cf.enable(mode="nope")
        assert False
    except ValueError:
        pass
    cf.disable()
    assert nn.Conv3d._conv_forward is cf._orig_conv_forward


def test_cpu_tensors_use_stock_path():
    """the patch only acts on GPU tensors (input.is_cuda), so CPU models are untouched even when enabled."""
    cf.enable()
    try:
        m = nn.Conv3d(3, 4, 3, padding=1).double(); x = torch.randn(1, 3, 5, 6, 6, dtype=torch.float64)
        assert torch.equal(m(x), F.conv3d(x, m.weight, m.bias, 1, 1))
    finally:
        cf.disable()


def test_strict_weight_loading():
    """check_equivalence.load_strict: a good checkpoint loads; a bad one (missing / unexpected key) fails; named keys may be ignored."""
    import check_equivalence as ce
    m = nn.Sequential(nn.Conv3d(2, 3, 3), nn.Conv3d(3, 3, 3))
    good = {k: v.clone() for k, v in m.state_dict().items()}
    ce.load_strict(m, good)
    bad_missing = {k: v for k, v in good.items() if k != "1.bias"}
    bad_extra = dict(good); bad_extra["stray.weight"] = torch.zeros(1)
    for bad in (bad_missing, bad_extra):
        try:
            ce.load_strict(m, bad)
            assert False, "a bad checkpoint must fail"
        except RuntimeError:
            pass
    ce.load_strict(m, bad_extra, ignore=["stray.weight"])      # explicitly named: allowed
    ce.load_strict(m, bad_missing, ignore=["1.bias"])
    try:
        ce.load_strict(m, bad_extra, ignore=["other.name"])    # a different name does not excuse it
        assert False
    except RuntimeError:
        pass


def test_autocast_option_runs_on_cpu():
    """compare() accepts autocast=bf16 (the gates' regime) and runs on CPU (patched path forced)."""
    import check_equivalence as ce
    m = nn.Sequential(nn.Conv3d(2, 4, 3, padding=1), nn.ReLU(), nn.Conv3d(4, 2, 3, padding=1)).eval()
    x = torch.randn(1, 2, 6, 8, 8)
    ce._force_patch.calls = 0
    r = ce.compare(m, x, torch.device("cpu"), "bf16", "fp32_taps", 1e-2, 1e-2, True, force_cpu=True, repeat=1, autocast="bf16")
    assert ce._force_patch.calls > 0 and "ratio" in r


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
