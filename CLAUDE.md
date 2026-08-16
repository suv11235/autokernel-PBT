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

The helper uses `git reset --hard`, which combined with `git checkout <sha>` silently detaches
HEAD. That happened once and orphaned a commit that was only recoverable from the reflog. Check
the branch, not just the SHA the tool printed back.

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
- `tests/spec/test_0004_property_layer.py` asserts every criterion in
  `specs/features/0004-property-oracle-layer/acceptance.yaml` resolves to a **collectable** test
  node. Adding a criterion that names a nonexistent test fails the suite.
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
| Both tables carry a `corpus_fingerprint`; pair them with `driver.read_run` | Case ids are a pure function of `(seed, index)`, so another run's `scores.parquet` joins perfectly and reports a rate about neither run. The fingerprint is a per-write uuid plus the case-id set, so a re-record gets a new identity. |
| Bad **data** → `INCONCLUSIVE`; bad **call** → raise | The line is whether a re-run costs hardware time. Offline scoring can be re-run for free. |
| `np.ascontiguousarray` is `ndmin=1` | It promotes 0-d to `(1,)` *before* safetensors sees it. safetensors round-trips `[]` faithfully — do not blame it. |
| Kernel inputs are read-only during execution | `readonly_inputs` turns silent corruption into a loud `launch_error`. Verified against 20 legitimate kernels; none affected. |
| The execution table is never observed torn | Index and payloads swap atomically. A crash may lose the table; it must never mix runs. |
| `Status`/`Verdict` are `str`-mixin enums with `__str__ = str.__str__` | `format()` returns the value on py3.10/3.11 and the *name* on 3.12+, against a declared `>=3.10`. |
| `Case.dtype`/`Case.shape` describe the primary tensor `x` only | Helper tensors (`__perm__`) carry their own. Read each tensor's own attributes. |

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

Obligations 1, 2 and 4 were discharged by phase 1.5 (`props/driver.py`, the kernel-identity
and score tables, and the contract-built declarative arm). What remains:

1. **The declarative arm, hybrid arm, and contract loader have no acceptance criteria.** The
   existing criteria cover infrastructure only.
2. **`elapsed_s` is recorded but not yet fair.** It is order-biased — the arm that runs second
   inherits warm caches — and a single run's value must not be quoted as a cost-per-bug
   denominator. The metrics phase needs repeated timing with randomized arm order.
3. **Partial abstention is undetectable.** The driver refuses an arm that is INCONCLUSIVE on
   *every* group, but an arm that abstains on some cannot be told from one that honestly could
   not judge them — abstention is a legitimate answer, so only the degenerate case is decidable.
4. `HybridOracle` is not wired into `run_task`, which drives one task with two arms by design.
   Adding it means deciding how its deliberately conditional reference coverage interacts with
   the driver's whole-table coverage check.
5. Degenerate ladder shapes `(1,1)` and `(17,1)` make softmax identically 1.0, so ~22% of groups
   score any kernel clean. It deflates both arms equally, so arm-vs-arm stays unbiased, but the
   absolute detection rate is understated by that constant and the paper must say so. Measured
   end to end through the driver: 7/9 = 0.778 for both arms against `unnormalized_softmax`.
