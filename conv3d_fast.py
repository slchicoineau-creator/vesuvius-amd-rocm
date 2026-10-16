"""3D convolution as a sum of 2D convolutions over depth slices (for AMD RDNA4 / gfx1201, e.g. Radeon AI PRO R9700).

Why: on gfx1201, MIOpen has hand-tuned 2D kernels (e.g. the Winograd assembly kernel ConvWinoFuryRxS for fp16 3x3,
~25 TFLOPS) but routes every 3D convolution to the generic im2col+GEMM fallback GemmFwdRest (~3 TFLOPS) or a naive
kernel (seen with MIOPEN_LOG_LEVEL=5). A kd x kh x kw convolution is exactly the sum over i < kd of a kh x kw 2D
convolution of the depth slices d*sd + i - pd, so it can run on the fast 2D kernels. Autograd differentiates it, so
training works too. Only the floating-point summation order differs from conv3d.

Use: `import conv3d_fast; conv3d_fast.enable()` before building/running a model patches nn.Conv3d for the process
(groups == 1, dilation 1, zero padding only; anything else falls back to the stock conv3d).

Opt-in bf16/fp16 modes: enable(mode=...). "taps" (default) is the plain decomposition. The others change only the
rounding in bf16/fp16; in fp32/fp64 all modes are exact up to summation order (see tests/test_conv3d_fast.py):
  "acc32_bias"        per-tap conv2d in low precision, taps + bias summed in fp32, one rounding at the end
  "taps_as_channels"  ONE conv2d per layer: the kd depth taps stacked as input channels (kd x input copy), bias inside
  "fp32_taps"         per-tap conv2d in fp32 on the bf16-rounded input/weight/bias (= stock's math), one rounding
See README.md for which mode to use; verify on your own model with check_equivalence.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_orig_conv_forward = nn.Conv3d._conv_forward


def conv3d_as_2d(x, weight, bias, stride, padding):
    B, C, D, H, W = x.shape
    Cout, Cin, kd, kh, kw = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    if pd:
        x = F.pad(x, (0, 0, 0, 0, pd, pd))
    Dp = x.shape[2]
    Dout = (Dp - kd) // sd + 1
    # depth into the batch: (B, D, C, H, W), contiguous once; each tap i takes a strided view of the D axis
    xt = x.transpose(1, 2)
    out = None
    for i in range(kd):
        xi = xt[:, i:i + sd * (Dout - 1) + 1:sd]                      # (B, Dout, C, H, W) view
        xi = xi.reshape(B * Dout, C, H, W)                              # copies when not contiguous
        yi = F.conv2d(xi, weight[:, :, i], None, (sh, sw), (ph, pw))
        if _ACC32["on"]:   # sum the taps in fp32, round once (stock conv3d accumulates in fp32 inside the kernel)
            out = yi.to(torch.promote_types(yi.dtype, torch.float32)) if out is None else out.add_(yi)
        else:
            out = yi if out is None else out + yi
    if _ACC32["on"]:
        out = out.to(yi.dtype)
    Ho, Wo = out.shape[-2:]
    out = out.reshape(B, Dout, Cout, Ho, Wo).transpose(1, 2)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out.contiguous()


def stem_as_2d(x, weight, bias, stride, padding):
    """Single-input-channel kd x kh x kw convolution as ONE 2D convolution with the kd depth taps as input channels
    (exact up to summation order). For v8-in's 7x7x7 stem, which the depth-tap sum above does not speed up."""
    B, _, D, H, W = x.shape
    Cout, _, kd, kh, kw = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    if pd:
        x = F.pad(x, (0, 0, 0, 0, pd, pd))
    Dout = (x.shape[2] - kd) // sd + 1
    xu = x[:, 0].unfold(1, kd, sd)                                    # (B, Dout, H, W, kd) view
    xu = xu.permute(0, 1, 4, 2, 3).reshape(B * Dout, kd, H, W)        # copy: taps become channels
    y = F.conv2d(xu, weight[:, 0], bias, (sh, sw), (ph, pw))           # (B*Dout, Cout, Ho, Wo)
    Ho, Wo = y.shape[-2:]
    return y.reshape(B, Dout, Cout, Ho, Wo).transpose(1, 2).contiguous()


_STEM = {"on": False}
_ACC32 = {"on": False}   # opt-in: in bf16 the plain tap sum can add noticeably more rounding noise than stock conv3d


def use_stem(weight, groups, dilation, padding_mode, padding):
    """Optional (enable(stem=True)): single-input-channel 3D kernels with depth > 1, e.g. the 7x7x7 stem."""
    return (_STEM["on"] and groups == 1 and tuple(dilation) == (1, 1, 1) and padding_mode == "zeros"
            and not isinstance(padding, str) and weight.shape[1] == 1 and weight.shape[2] > 1)


def use_fast(weight, groups, dilation, padding_mode, padding):
    """Per-layer policy from a per-layer benchmark (R9700, fp16, batch 16): decomposing 3x3x3 kernels with
    several input channels wins 2.8-7x at stride 1 and ~1.4x net at stride 2; 1x1x1 kernels are faster on the stock
    path (0.3-0.4x if decomposed) and the 7x7x7 single-channel stem is a wash (0.94x), so those stay stock."""
    kd, kh, kw = weight.shape[2:]
    return (groups == 1 and tuple(dilation) == (1, 1, 1) and padding_mode == "zeros" and not isinstance(padding, str)
            and kd == 3 and kh == 3 and kw == 3 and weight.shape[1] > 1)


# ---------------------------------------------------------------- opt-in bf16 modes
def _lowp(x):
    """the dtype the convolution really runs in: the autocast dtype when autocast is on for x's device, else x.dtype."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return x.dtype


def _taps(x, kd, sd, pd, dtype):
    """yields the (B*Dout, C, H, W) input of each depth tap. One depth-major copy of x in `dtype`; each tap is a view of it
    when sd == 1 and B == 1 (else a copy, one at a time). Dout = (D + 2 pd - kd) // sd + 1."""
    if pd:
        x = F.pad(x, (0, 0, 0, 0, pd, pd))
    B, C, Dp, H, W = x.shape
    Dout = (Dp - kd) // sd + 1
    xt = x.new_empty((B, Dp, C, H, W), dtype=dtype).copy_(x.transpose(1, 2))
    for i in range(kd):
        yield xt[:, i:i + sd * (Dout - 1) + 1:sd].reshape(B * Dout, C, H, W)


def _dout(x, kd, sd, pd):
    return (x.shape[2] + 2 * pd - kd) // sd + 1


def _from_2d(y, B, Dout):
    Ho, Wo = y.shape[-2:]
    return y.reshape(B, Dout, y.shape[1], Ho, Wo).transpose(1, 2).contiguous()


def conv3d_acc32_bias(x, weight, bias, stride, padding):
    """per-tap conv2d in the run precision; the taps AND the bias are summed in fp32 and rounded once."""
    dt = _lowp(x); ct = torch.promote_types(dt, torch.float32); kd = weight.shape[2]; out = None
    Dout = _dout(x, kd, stride[0], padding[0])
    for i, xi in enumerate(_taps(x, kd, stride[0], padding[0], dt)):
        yi = F.conv2d(xi, weight[:, :, i], None, stride[1:], padding[1:])
        out = yi.to(ct) if out is None else out.add_(yi)
    if bias is not None:
        out = out + bias.to(dt).to(ct).view(1, -1, 1, 1)
    return _from_2d(out.to(yi.dtype), x.shape[0], Dout)


def conv3d_taps_as_channels(x, weight, bias, stride, padding):
    """ONE conv2d: input (B*Dout, kd*Cin, H, W) = the depth taps stacked as channels, weight (Cout, kd*Cin, kh, kw), bias in
    the kernel. One accumulation, one rounding, like stock conv3d. Costs a kd x copy of the input (in the run precision)."""
    B, C, D, H, W = x.shape
    Cout, Cin, kd, kh, kw = weight.shape
    sd, pd = stride[0], padding[0]
    dt = _lowp(x)
    x = x.to(dt)                                                       # autocast would cast anyway: copy the small dtype
    if pd:
        x = F.pad(x, (0, 0, 0, 0, pd, pd))
    Dout = (x.shape[2] - kd) // sd + 1
    xt = x.transpose(1, 2)                                             # (B, Dp, C, H, W) view
    xs = torch.cat([xt[:, i:i + sd * (Dout - 1) + 1:sd] for i in range(kd)], dim=2)   # (B, Dout, kd*C, H, W), one copy
    w2 = weight.permute(0, 2, 1, 3, 4).reshape(Cout, kd * Cin, kh, kw)
    y = F.conv2d(xs.reshape(B * Dout, kd * C, H, W), w2, bias, stride[1:], padding[1:])
    return _from_2d(y, B, Dout)


def conv3d_fp32_taps(x, weight, bias, stride, padding):
    """stock conv3d's math on the 2D kernels: input, weight and bias rounded to the run precision, every tap computed and
    summed in fp32 (autocast off), one rounding at the end. Costs fp32 tap copies and fp32 conv2d speed."""
    dt = _lowp(x); ct = torch.promote_types(dt, torch.float32); kd = weight.shape[2]
    with torch.autocast(x.device.type, enabled=False):
        wf = weight.to(dt).to(ct); out = None
        Dout = _dout(x, kd, stride[0], padding[0])
        for i, xi in enumerate(_taps(x.to(dt), kd, stride[0], padding[0], ct)):
            yi = F.conv2d(xi, wf[:, :, i], None, stride[1:], padding[1:])
            out = yi if out is None else out + yi
        if bias is not None:
            out = out + bias.to(dt).to(ct).view(1, -1, 1, 1)
    return _from_2d(out.to(dt), x.shape[0], Dout)


MODE_FNS = {"taps": conv3d_as_2d, "acc32_bias": conv3d_acc32_bias, "taps_as_channels": conv3d_taps_as_channels,
            "fp32_taps": conv3d_fp32_taps}
_MODE = {"m": "taps"}


def layer_mode(module, mode):
    """per-layer override of the global mode (None removes it); used for mixes such as fp32 taps only in some layers."""
    if mode is None:
        module.__dict__.pop("_c3f_mode", None)
    else:
        assert mode in MODE_FNS, mode
        module._c3f_mode = mode


def _fast_conv_forward(self, input, weight, bias):
    if input.is_cuda and use_fast(weight, self.groups, self.dilation, self.padding_mode, self.padding):
        mode = self.__dict__.get("_c3f_mode") or _MODE["m"]
        if mode == "taps":
            return conv3d_as_2d(input, weight, bias, tuple(self.stride), tuple(self.padding))
        return MODE_FNS[mode](input, weight, bias, tuple(self.stride), tuple(self.padding))
    if input.is_cuda and use_stem(weight, self.groups, self.dilation, self.padding_mode, self.padding):
        return stem_as_2d(input, weight, bias, tuple(self.stride), tuple(self.padding))
    return _orig_conv_forward(self, input, weight, bias)


def enable(stem=False, acc32=False, mode="taps"):
    """stem=False, acc32=False, mode="taps" is the default configuration; stem=True adds the single-channel-stem rewrite;
    acc32=True sums the depth taps in fp32 ("taps" mode only); mode picks one of MODE_FNS. Check any non-default
    choice on your own model first (check_equivalence.py)."""
    if mode not in MODE_FNS:
        raise ValueError(f"unknown conv3d_fast mode {mode!r}; known: {tuple(MODE_FNS)}")
    _STEM["on"] = bool(stem)
    _ACC32["on"] = bool(acc32)
    _MODE["m"] = mode
    nn.Conv3d._conv_forward = _fast_conv_forward


def disable():
    nn.Conv3d._conv_forward = _orig_conv_forward
