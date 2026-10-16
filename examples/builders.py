"""Model builders for check_equivalence.py (`--model builders:NAME`). Paths come from environment variables so nothing
is hard-coded:
  HECATE_DIR      folder with hecate.py (from the Hecate Hugging Face release; checkpoints next to it)
  VILLA_INFERENCE_DIR  villa/ink-detection/optimized_inference (provides model_resnet3d_3d_decoder.py)
  CANON_CKPT      path of r152_3ddec_v2_l5_epoch13.ckpt (scrollprize/ink_canonical_2um)
  V8IN_DIR        folder of the YoussefMoNader/ink-8um-v8in download (provides ink8um/)
Each builder returns an eval-mode nn.Module whose forward takes ONE tensor (shapes in RUNBOOK.md)."""
import os, sys
import torch


def _hecate(ckpt):
    sys.path.insert(0, os.environ["HECATE_DIR"])
    import hecate
    return hecate.load_model(os.path.join(os.environ["HECATE_DIR"], ckpt), "cpu").eval()


def hecate24():
    return _hecate("hecate_2.4um.pth")


def hecate96():
    return _hecate("hecate_9.6um.pth")


def canonical():
    sys.path.insert(0, os.environ["VILLA_INFERENCE_DIR"])
    from model_resnet3d_3d_decoder import RegressionModel, _extract_state_dict, _strip_known_prefixes
    ck = torch.load(os.environ["CANON_CKPT"], map_location="cpu", weights_only=False)
    sd = _strip_known_prefixes(_extract_state_dict(ck))
    m = RegressionModel(with_norm=any(k.startswith("normalization.") for k in sd))
    print("canonical load_state_dict:", m.load_state_dict(sd, strict=True))   # strict: a key mismatch must stop the run
    return m.eval()


def v8in():
    sys.path.insert(0, os.environ["V8IN_DIR"])
    from ink8um import InkDetector
    return InkDetector.from_pretrained(os.environ["V8IN_DIR"]).eval()
