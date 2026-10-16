#!/usr/bin/env python3
"""Run any Python script with conv3d_fast switched on, without editing it.
  python run_with_fast.py [--mode fp32_taps] [--stem] script.py [script args...]
  python run_with_fast.py [--mode fp32_taps] [--stem] -m package.module [args...]
(`conv3d_fast.py` must be importable: keep this folder's parent on PYTHONPATH or run from the package root.)"""
import os, runpy, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import conv3d_fast
args = sys.argv[1:]; mode, stem = "taps", False
while args and args[0] in ("--mode", "--stem"):
    if args[0] == "--mode": mode = args[1]; args = args[2:]
    else: stem = True; args = args[1:]
conv3d_fast.enable(mode=mode, stem=stem)
print(f"conv3d_fast enabled: mode={mode} stem={stem}", file=sys.stderr, flush=True)
if args[0] == "-m":
    sys.argv = args[1:]; runpy.run_module(args[1], run_name="__main__", alter_sys=True)
else:
    script = args[0]; sys.argv = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
