#!/usr/bin/env bash
# Prepare a fresh Lambda instance to record a run. Idempotent; safe to re-run.
#
# Lambda Stack ships torch, CUDA and triton in the system Python, so this installs the
# project against that interpreter rather than building a venv that would shadow a
# working torch with a wheel that may not match the driver.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== versions this run will be recorded under =="
python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"  torch      {torch.__version__}")
print(f"  cuda       {torch.version.cuda}")
try:
    import triton
    print(f"  triton     {triton.__version__}")
except ImportError:
    print("  triton     MISSING -- stop, the run cannot proceed")
print(f"  device     {p.name}")
print(f"  capability {p.major}.{p.minor}")
print(f"  sms        {p.multi_processor_count}")
PY

echo "== installing the project (torch and triton come from Lambda Stack) =="
python3 -m pip install --quiet -e ".[dev]"

echo "== verifying torch<->numpy interop survived the install =="
# The failure this guards is silent: pip upgrading numpy past torch's ABI leaves
# torch importable but unable to convert arrays, and the Triton backend calls
# torch.as_tensor(ndarray) on every launch. pyproject pins numpy<2; this proves it.
python3 - <<'PYCHECK'
import warnings
warnings.simplefilter("error")
import numpy as np
import torch
back = torch.as_tensor(np.ones((2, 3), dtype=np.float32), device="cuda").cpu().numpy()
assert back.shape == (2, 3), back.shape
print(f"  numpy {np.__version__} <-> torch {torch.__version__}: OK")
PYCHECK

echo "== CPU suite, to prove the checkout is sound before spending on device =="
python3 -m pytest -m "not gpu" -q

echo "== device tests =="
python3 -m pytest -m gpu -q

echo "bootstrap OK"
