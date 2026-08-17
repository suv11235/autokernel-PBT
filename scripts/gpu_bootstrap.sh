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

echo "== CPU suite, to prove the checkout is sound before spending on device =="
python3 -m pytest -m "not gpu" -q

echo "== device tests =="
python3 -m pytest -m gpu -q

echo "bootstrap OK"
