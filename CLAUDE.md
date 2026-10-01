# autokernel-PBT — working notes for agents

**PBT means *property-based testing*.** Not population-based training. An earlier draft used
the latter; that framing was removed in full.

The project compares **three oracle strategies** — reference, declarative, hybrid — over
**byte-identical replayed executions**, measuring bug-catching power, false-positive rate,
authoring cost, and cost-per-bug. Almost every convention below exists to keep one of those
numbers honest.

Design: `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`
Phase 1 plan: `docs/superpowers/plans/2026-08-14-property-layer-phase-1.md`

---

## Committing — read this before your first commit

**`git commit` is forbidden in this repo.** It injects a `Co-authored-by: Cursor` trailer that
lands on the contribution graph. Use:

```bash
git add <paths>
scripts/git_commit_clean.sh -m "subject" -m "body paragraph"
```

`git commit --amend` is forbidden for the same reason — to reword, use the `commit-tree` recipe
in `.cursor/skills/clean-git-commits/SKILL.md`.

**Verify the branch pointer moved after committing:**

```bash
git branch --show-current    # must not be empty
```

Both the helper and the skill's `commit-tree` reword recipe end in `git reset --soft`. The recipe
once used `--hard`, which discards uncommitted work, and combined with `git checkout <sha>`
silently detaches HEAD. That happened once and orphaned a commit that was only recoverable from
the reflog. Check the branch, not just the SHA the tool printed back.

Commit subjects are conventional (`feat:`, `fix:`, `docs:`, `spec:`, `test:`) with a short prose
body explaining **why**, not a changelog of what changed.

---

## Tests

```bash
pytest -m "not gpu" -q      # the whole suite; must be green and silent
```

- `filterwarnings = ["error"]` — **any warning is a failure.** Tests that legitimately provoke
  one opt out with `@pytest.mark.filterwarnings`, per-test, with a comment saying why.
- Markers: `gpu`, `integration`, `spec`.
- Each of `tests/spec/test_0004_*.py` through `test_0008_*.py` asserts every criterion in its
  feature's `acceptance.yaml` resolves to a **collectable** test node (0005 onward also reject a
  shared or file-only target). Adding a criterion that names a nonexistent test fails the suite.
  Features 0001 and 0002 have no such guard.
- Spec before code: a feature gets `specs/features/NNNN-*/spec.md` + `acceptance.yaml` first,
  and the spec test starts red. See `specs/README.md` and `docs/adr/0001-sdd-tdd.md`.

---

## Module contracts that cost review rounds to establish

Do not rediscover these.

| Contract | Why |
|---|---|
| `residual_ratio` takes an explicit `n=` | The default is the last-axis length, which is **wrong** for an already-reduced array. Pass the reduction length from the input. |
| Normalization is `max(log2(n), 1.0)` | Linear `n` is the bound for *sequential* accumulation; these backends reduce pairwise. Under linear `n` the reference arm missed bugs `np.allclose` catches. |
| `ExactDtypeError` is caught narrowly → `INCONCLUSIVE` | Letting it propagate aborts a run; mapping it to FAIL books a correct int-returning kernel as a caught bug. |
| Every `PropertyResult` carries exactly one of `case_id`/`group_id` **from an oracle** | `HybridOracle` concatenates arms; the split point is not recoverable from a flat list. `_result` raises otherwise. |
| Every **persisted score row** carries `group_id`, always; `case_id` refines it | The case group is the unit at which arms are comparable — per-result rates differ 0.778 vs 0.222 for the same 14 detections. The driver (`_keyed_by_group`) stamps it; `ScoreTable` refuses a row without it. |
| Both tables carry a `corpus_fingerprint`; pair them with `driver.read_run` | Case ids are a pure function of `(seed, index)`, so another run's `scores.parquet` joins perfectly and reports a rate about neither run. The fingerprint is `<uuid>-<case-id digest>`, side by side, so a re-record gets a new identity *and* every read recomputes the digest from its rows. Stamp and compare with the fingerprint from the **same** read (`read_with_fingerprint`), never a second `corpus_fingerprint()` call. The five recorded runs carry the older 64-hex form, which is accepted unverified. |
| Bad **data** → `INCONCLUSIVE`; bad **call** → raise | The line is whether a re-run costs hardware time. Offline scoring can be re-run for free. |
| `np.ascontiguousarray` is `ndmin=1` | It promotes 0-d to `(1,)` *before* safetensors sees it. safetensors round-trips `[]` faithfully — do not blame it. |
| Kernel inputs are read-only during execution | `readonly_inputs` turns silent corruption into a loud `launch_error`. Verified against 20 legitimate kernels; none affected. |
| The execution table is never observed torn | Index and payloads swap atomically. A crash may lose the table; it must never mix runs. Each payload is also stamped with its write's fingerprint (safetensors metadata) and refused under any other index -- the swap protects one writer from its own crash, not a reader from a second writer or a run directory copied home in pieces. |
| A payload's filename is `payload_filename(case_id)`, **never the raw id** | Partner ids contain `::`. Synced through OneDrive, every `…base::shift_rows.safetensors` was silently renamed `…base_shift_rows`, and the runs became unreadable; a case-insensitive disk also merged `Case-0` and `case-0`. The encoding is injective and lowercase. Legacy runs, verbatim names, are read by fallback **only** under a legacy index. Never keep a run directory on OneDrive. |
| Generated bytes must stay regenerable: **never change the stream key or draw order** | `default_rng([seed, index])`, tensors then relations, in declaration order. The recorded runs are regenerated from exactly this; `test_the_recorded_corpus_is_regenerated_byte_for_byte` pins it. Seeds are bounded to `[0, 2**32)`, because a wider one spills into the index word and collides. |
| `Status`/`Verdict` are `str`-mixin enums with `__str__ = str.__str__` | `format()` returns the value on py3.10/3.11 and the *name* on 3.12+, against a declared `>=3.10`. |
| `Case.dtype`/`Case.shape` describe the primary tensor `x` only | Helper tensors (`__perm__`) carry their own. Read each tensor's own attributes. |
| Per-task property bundles name their members **explicitly** | They were once built from the whole registry, which was correct only while every property held for softmax. `rows_have_zero_mean` holds for no softmax output at all. |
| Telemetry declares every key; unavailable ones carry `MISSING` | An omitted field makes "not captured" and "captured as zero" indistinguishable, and they mean opposite things about register pressure. |
| Triton introspection is **probed**, not asserted | Register and spill counts have moved between the compiled kernel and its metadata across releases. A version bump must degrade one field, not abort a paid run. That includes the device probe and any probed value the table cannot encode: both degrade to MISSING. |
| `kernel_source_hash` reads a JIT function's `.src` | `inspect.getsource` raises on a `JITFunction`, so until this was fixed every recorded Triton hash covered only the launcher and the kernel's *name*. Runs recorded before it are not hash-comparable with later ones. Unreadable source is MISSING, never a name hash. |
| The input-mutation check lives in the **launcher**, not the backend | `readonly_inputs` protects the host array, and the backend holds only host arrays — a check there is structurally incapable of firing. Only the launcher holds the device buffers. |
| `log2(n)` is safe but not tightest; the hybrid bound is `log2(tile) + n_tiles` | Measured on a multi-tile kernel: 3.5x drift vs `log2(n)`'s 5.9x. It needs the tile width, which telemetry records. Not adopted -- one kernel, one GPU. `n=1` is settled *against*: it looked flatter only in the single-tile regime. |
| The Triton tile width is **derived from the shape**, never fixed | Too large and every shape shares one compiled artifact, so all compiled telemetry is constant and carries no signal. Too small and `tl.arange(0, BLOCK)` drops the row's tail *silently* — measured, softmax rows summing to 1.51 instead of 1.0. |
| `compute-sanitizer` needs `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | PyTorch sub-allocates from large cached segments, so an intra-segment overrun is not an OOB at the CUDA level. Measured: a 1 MB overrun past a 1 KB buffer reports **0 errors** with caching on, and is caught with it off. |
| `pythonpath` in pyproject is a **pytest** setting | It makes `kernels/` importable for tests and does nothing for a standalone script. The device tests passed while `gpu_record.py` died on import. |
| A property's tolerance must model the **dominant** error mechanism | `rows_have_unit_variance` bounded float rounding and failed a correct reference: the real term is the reference's `eps`, `var/(var+eps)-1`, which grows as variance *shrinks* and is independent of `n`. **The bound still models rounding only.** The false alarm is avoided by layernorm's U(-10, 10) input (`tasks.py`), pinned by `test_layernorm_properties_pass_the_real_reference`; narrowing the input brings it back. |
| Mutant bodies are recorded **verbatim** | A mutant tidied by someone who has seen the property set is no longer blinded. Lint exceptions go in `pyproject.toml` per-file ignores, never in the body; the gate, not the editor, removes unusable candidates. |

---

## The review standard

Reviews here are adversarial by design, and it earned its cost — 20 defects were caught after
spec compliance had already passed.

- **Verify, do not trust the report.** Re-run the claim. Several implementer reports were
  accurate in substance and wrong in a specific number.
- **A passing test is a hypothesis.** Break the implementation and confirm the test dies. Four
  separate tests in this repo asserted a *label* rather than the behaviour the label named.
- **Every assertion must be the *unique* catcher for at least one saboteur.** "Every saboteur is
  caught" is too weak — a saboteur caught by an earlier assertion silently certifies a later one
  that never ran. Pair each saboteur with the exact expected message (`pytest.raises(..., match=)`)
  and verify by deleting each assertion in turn that precisely its own cases fail. This defect
  appeared three times and *relocated* each time it was fixed.
- **Guard every field a consumer reads, not the ones named "input".** The fairness criterion
  fingerprinted `case.tensors` but not `outputs` or `case.metadata()`; an arm that merely
  relabelled `case.relation` cost 14/14 detections with every tensor byte untouched.
- Disagreements get **measured**, not deferred. Two agents disagreeing usually means they built
  different harnesses — state your construction before claiming a number.

---

## Open obligations

Phase 1.5 discharged the original obligations for a driver (`props/driver.py`), the
kernel-identity and score tables, and the contract-built declarative arm; the list below was
renumbered afterwards. Feature 0006 discharged the
acceptance criteria for the declarative and hybrid arms, and wired `HybridOracle` and the
`allclose` arm into the driver. What remains:

1. **`elapsed_s` is recorded but not yet fair.** Arm order is randomized per run as of 0006
   (`driver.arm_order`), so the bias is no longer *systematic* — but a single run's value must
   still not be quoted as a cost-per-bug denominator. At ~0.5 ms per arm the measurement sits
   near the clock's noise floor; it needs repeated timing with a reported spread. Feature 0008, the
   metrics phase, shipped without it (`metrics/rates.py` has no timing), so no feature owns it yet.
2. **Partial abstention is undetectable.** The driver refuses an arm that is INCONCLUSIVE on
   *every* group, but an arm that abstains on some cannot be told from one that honestly could
   not judge them — abstention is a legitimate answer, so only the degenerate case is decidable.
   `RowsHaveUnitVariance` is now a deliberate instance of legitimate abstention.
3. **The ladder deflates absolute detection, in two places.** Degenerate shapes `(1,1)` and
   `(17,1)` make softmax identically 1.0, and layernorm's variance property abstains on the same
   rungs because a constant row normalizes to zero, not to unit variance. It deflates every arm
   equally, so arm-vs-arm stays unbiased, but any absolute rate is understated by that constant
   and the paper must say so. Measured end to end through the driver: **7/9 = 0.778** for every
   arm, on softmax against `unnormalized_softmax` and on layernorm against a
   centers-but-never-divides kernel. The generated report already states it
   (`metrics/report.py`, criterion `THE_REPORT_STATES_THE_DEFLATION`); the paper still must.
4. **Authoring cost for layernorm is `n = 1`**
   (`docs/measurements/2026-08-16-layernorm-authoring-cost.md`), with its threats pre-registered.
   Extend it before the number is reported.
5. ~~`pyproject.toml` pins no numpy upper bound.~~ **Discharged** by `fce213f`: `pyproject.toml`
   pins `numpy>=1.24,<2` and `scripts/gpu_bootstrap.sh` checks torch's numpy interop after
   install. Side effect: NumPy 1.x links OpenBLAS on macOS, not Accelerate, so CPU wall-clock
   numbers measured under NumPy 2.x do not reproduce in this project's venv (measured: 265 vs 1633
   steps in the same budget). `experiments/autoresearch-pbt/` therefore has its own venv; see its
   README.
6. `src/autokernel_pbt/harness/correctness.py` still carries the five-stage skeleton the parent design §11 says the
   property layer replaces. It is load-bearing for features 0001 and 0002 and their acceptance
   criteria, so retiring it means retiring a feature — a scope decision, not cleanup.
