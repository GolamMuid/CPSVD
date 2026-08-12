#!/usr/bin/env python3
"""
check_and_fix_cuda_pytorch.py

1. Detects whether the machine has an NVIDIA GPU (via nvidia-smi).
2. Detects whether the installed PyTorch build has CUDA support.
3. If a GPU exists but PyTorch is CPU-only (or can't see the GPU), it
   reinstalls PyTorch using the CUDA wheel index matching the driver's
   supported CUDA version.

Safe by default: run with no flags to only DIAGNOSE.
Add --apply to actually uninstall/reinstall PyTorch.

Usage:
    python check_and_fix_cuda_pytorch.py            # diagnose only
    python check_and_fix_cuda_pytorch.py --apply    # diagnose + fix
"""

import argparse
import re
import subprocess
import sys


def run(cmd, check=False):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, check=check)


def has_nvidia_gpu():
    """
    Returns (bool, driver_cuda_version_str_or_None).

    Handles two distinct failure modes:
      1. nvidia-smi binary doesn't exist at all (FileNotFoundError) -- this is
         the normal case on machines with no NVIDIA GPU/drivers (e.g. Macs,
         AMD/Intel-only Linux boxes, or Windows without drivers installed).
      2. nvidia-smi exists but returns a non-zero exit code (e.g. driver
         installed but broken) -- treated the same way, as "no usable GPU".
    """
    try:
        result = run(["nvidia-smi"])
    except FileNotFoundError:
        return False, None
    except OSError:
        # Covers PermissionError and other OS-level launch failures too.
        return False, None

    if result.returncode != 0:
        return False, None

    match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    driver_cuda_version = match.group(1) if match else None
    return True, driver_cuda_version


def get_torch_cuda_status():
    """Returns dict with torch_installed, torch_version, torch_cuda_version, cuda_available."""
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
        }

    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,   # None means CPU-only build
        "cuda_available": torch.cuda.is_available(),
    }


def pick_wheel_tag(driver_cuda_version):
    """
    Maps the driver's max-supported CUDA version to a PyTorch wheel index tag.
    PyTorch wheels are only published for specific CUDA versions -- this picks
    the newest published tag that is <= the driver's supported version.
    Update PUBLISHED_TAGS if PyTorch adds/drops CUDA versions in the future.
    """
    PUBLISHED_TAGS = [
        ("12.6", "cu126"),
        ("12.4", "cu124"),
        ("12.1", "cu121"),
        ("11.8", "cu118"),
    ]

    if driver_cuda_version is None:
        return PUBLISHED_TAGS[0][1], True

    try:
        driver_ver = tuple(int(x) for x in driver_cuda_version.split("."))
    except ValueError:
        return PUBLISHED_TAGS[0][1], True

    for ver_str, tag in PUBLISHED_TAGS:
        tag_ver = tuple(int(x) for x in ver_str.split("."))
        if driver_ver >= tag_ver:
            return tag, False

    return PUBLISHED_TAGS[-1][1], True


def reinstall_pytorch_with_cuda(wheel_tag):
    print(f"\n>>> Uninstalling existing torch/torchvision/torchaudio ...")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    index_url = f"https://download.pytorch.org/whl/{wheel_tag}"
    print(f">>> Installing CUDA-enabled PyTorch ({wheel_tag}) from {index_url} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
         "--index-url", index_url],
        check=False
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Actually reinstall PyTorch if a mismatch is found. "
                              "Without this flag, the script only diagnoses.")
    args = parser.parse_args()

    print("=" * 70)
    print("CUDA / PyTorch Diagnostic")
    print("=" * 70)

    gpu_present, driver_cuda_version = has_nvidia_gpu()
    print(f"NVIDIA GPU detected      : {gpu_present}")
    print(f"Driver max CUDA version  : {driver_cuda_version or 'unknown'}")

    torch_status = get_torch_cuda_status()
    print(f"\nPyTorch installed        : {torch_status['torch_installed']}")
    print(f"PyTorch version          : {torch_status['torch_version']}")
    print(f"PyTorch built with CUDA  : {torch_status['torch_cuda_version'] or 'No (CPU-only build)'}")
    print(f"torch.cuda.is_available()   : {torch_status['cuda_available']}")

    print("\n" + "-" * 70)

    if not gpu_present:
        print("No NVIDIA GPU detected on this machine (or nvidia-smi is unavailable).")
        print("Nothing to fix -- CPU-only PyTorch is the correct setup here.")
        return

    if torch_status["cuda_available"]:
        print("GPU is present AND PyTorch already sees it. No action needed.")
        return

    print("MISMATCH DETECTED: an NVIDIA GPU is present, but the installed PyTorch")
    print("build cannot use CUDA (either it's a CPU-only build, or there's a")
    print("driver/runtime version issue).")

    wheel_tag, is_fallback = pick_wheel_tag(driver_cuda_version)
    if is_fallback:
        print(f"\nCould not confidently match the driver's CUDA version -- "
              f"defaulting to the {wheel_tag} wheel. Verify this is compatible "
              f"with your driver before proceeding.")
    else:
        print(f"\nSelected PyTorch CUDA wheel: {wheel_tag} (driver supports up to "
              f"CUDA {driver_cuda_version})")

    if not args.apply:
        print("\nRun again with --apply to uninstall the current PyTorch build")
        print(f"and install the CUDA-enabled build ({wheel_tag}).")
        return

    success = reinstall_pytorch_with_cuda(wheel_tag)
    if not success:
        print("\nReinstall command failed. Check the pip output above for details.")
        sys.exit(1)

    print("\n>>> Reinstall complete. Verifying ...")
    verify = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('torch:', torch.__version__); "
         "print('cuda build:', torch.version.cuda); "
         "print('cuda available:', torch.cuda.is_available())"],
        capture_output=True, text=True
    )
    print(verify.stdout)
    if verify.returncode != 0:
        print(verify.stderr)


if __name__ == "__main__":
    main()
