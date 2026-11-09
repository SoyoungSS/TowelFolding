#!/usr/bin/env python3
"""Check Newton imports and execute a small Warp kernel on the CUDA device."""

import argparse
from importlib.metadata import version
import sys

import newton
import numpy as np
import warp as wp
from pxr import Usd


@wp.kernel
def double_values(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    index = wp.tid()
    result[index] = 2.0 * values[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"Newton: {version('newton')} ({newton.__file__})", flush=True)
    print(f"Warp: {version('warp-lang')}; USD: {Usd.GetVersion()}", flush=True)
    wp.init()
    if not wp.is_cuda_available():
        raise SystemExit("CUDA unavailable. Check nvidia-smi in a host terminal with GPU access.")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise SystemExit("This check requires a CUDA device.")

    values = np.arange(16, dtype=np.float32)
    source = wp.array(values, dtype=float, device=device)
    result = wp.empty(16, dtype=float, device=device)
    wp.launch(double_values, dim=16, inputs=[source, result], device=device)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(result.numpy(), values * 2)
    print(f"PASS: Newton/USD imports and compiled CUDA kernel on {device.name}.")


if __name__ == "__main__":
    main()
