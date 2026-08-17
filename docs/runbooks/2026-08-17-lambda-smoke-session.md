# Runbook: Lambda smoke session

**Purpose.** Prove the Triton backend compiles, launches, and that the telemetry schema
survives contact with real hardware — *before* the real run. The schema is the one decision a
re-run cannot fix, so it gets one cheap rehearsal.

**Expected duration.** Under an hour, most of it bootstrap.

---

## 0. Before you launch anything

**Rent the cheapest available GPU.** A full-ladder run is ~28 KB of tensor payload across nine
trivial kernels, and the tolerance sweep adds little. Nothing here is compute-bound and no result
depends on the device being fast. What matters is that compute capability is *recorded* — it
changes register limits, occupancy and SASS introspection — not that it is high. An H100 for this
workload is pure bill.

**SSH key.** This machine has one Lambda key:

```
~/.ssh/lambda_hackathon        comment: lambda-hackathon
                               SHA256:vHNNyvZAbW29N4NGKl6u00C5QFidi24ueiRDusl7Nyw
```

and `~/.ssh/config` already defines `Host lambda` pointing at it. If the instance was launched
with a *different* key, either relaunch it selecting `lambda-hackathon`, or paste
`~/.ssh/lambda_hackathon.pub` into the instance's `~/.ssh/authorized_keys` using Lambda's web
terminal. That `.pub` file is public and safe to copy; the private half never needs to leave this
machine.

Confirm access before doing anything else:

```bash
ssh lambda 'nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv'
```

If that fails, stop. Everything below assumes a working shell.

## 1. Bootstrap

```bash
ssh lambda
git clone https://github.com/suv11235/autokernel-PBT.git
cd autokernel-PBT
./scripts/gpu_bootstrap.sh
```

**Record the version block it prints** — torch, CUDA, triton, device, capability, SM count.
Lambda Stack pins these from the instance image, so two runs on different images are not
automatically comparable, and this block is what makes the difference visible.

**If the device tests fail, stop here.** That is the smoke session having done its job. Capture
the output, terminate the instance, fix locally. Debugging on the clock is the one avoidable cost.

## 2. Record

```bash
python3 scripts/gpu_record.py --task relu            --out runs/gpu-relu
python3 scripts/gpu_record.py --task softmax         --out runs/gpu-softmax
python3 scripts/gpu_record.py --task layernorm       --out runs/gpu-layernorm
python3 scripts/gpu_record.py --task tolerance_sweep --out runs/gpu-tolerance
```

Each prints a status histogram. Anything other than all-`ok` is a finding, not a reason to retry
blindly — `compile_error` means the kernel never built, `launch_error` with `input_mutated` set
means a kernel wrote to its own input.

## 3. Check the telemetry actually populated

**This is the whole point of the session.**

```bash
python3 - <<'PY'
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.backends.telemetry import declared_keys
rows = ExecutionTable("runs/gpu-softmax").read()
t = rows[0].telemetry
print("rows:", len(rows))
for k in declared_keys():
    print(f"  {k:28} {t.get(k)}")
empty = [k for k in declared_keys() if t.get(k) is None]
print("\nfields that came back empty:", empty)
PY
```

**Any field in that empty list is a decision to make now, while the instance is still up.**

- Empty because *Triton moved it* → add another probe location to `_COMPILED_FIELDS` in
  `src/autokernel_pbt/props/backends/telemetry.py`, re-run step 2, confirm it populates. This is
  the cheap fix, and it is only cheap now.
- Empty because *the device does not report it* → a real absence. Leave it `MISSING` and note it.

`n_regs`, `shared_bytes` and `num_warps` are the ones that matter most: they are the ISSTA
taxonomy's device-only signals, and losing one costs another rented hour later.

## 4. Bring the runs home

Instance storage is ephemeral. **Nothing is saved until this step completes.**

On the instance:

```bash
tar czf runs.tar.gz runs/
```

From your laptop:

```bash
scp lambda:~/autokernel-PBT/runs.tar.gz .
tar xzf runs.tar.gz
```

Verify locally *before* terminating:

```bash
python3 -c "
from autokernel_pbt.props.table import ExecutionTable
for t in ('relu','softmax','layernorm','tolerance'):
    rows = ExecutionTable(f'runs/gpu-{t}').read()
    print(f'{t:12} {len(rows):3} rows, telemetry keys: {len(rows[0].telemetry)}')
"
```

## 5. Terminate the instance

Only after step 4 verifies locally. Check the Lambda console shows it terminated — a forgotten
instance bills until someone notices.

## 6. Score at home, on CPU

No device required. This is the property the whole architecture exists to buy: the executions are
a reusable dataset, scorable whenever, on anything.

Scoring the recorded runs through the four arms is the next piece of work. The executions are
already safe on disk and are not going stale.
