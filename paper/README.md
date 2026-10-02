# AAAI workshop draft: *Skills, Properties, Proofs*

A 7-page (plus references) AAAI-format draft of this repository's work: the
property-based testing (PBT) gate, the skills that carried the work across agent
sessions, and the planned formal-proof layer.

## Build

```bash
make          # builds main.pdf (pdflatex + bibtex)
make check    # undefined refs, overfull boxes, BibTeX warnings, leftover \note{}s
make clean
```

## Using the AAAI author kit

The kit's style files are not committed. Without them, `main.tex` falls back to an
approximate two-column layout and the log warns that the output is not submittable.

1. Download the AAAI author kit for the target year from aaai.org. Use the workshop's
   own kit if it names one.
2. Copy its `.sty` and `.bst` (for example `aaai2026.sty` and `aaai2026.bst`) into this
   directory. `.gitignore` keeps them out of git.
3. If the kit's year is not 2026, change `\def\aaaistyle{aaai2026}` at the top of
   `main.tex`.
4. If BibTeX then reports a missing `\bibstyle`, add `\bibliographystyle{aaai2026}`
   (with the kit's name) before `\bibliography{references}`.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Preamble, title, section order |
| `numbers.tex` | **Every measured number the prose quotes**, each tagged with its source record |
| `sections/*.tex` | One file per section |
| `figures/harness.tex` | Figure 1 (TikZ) |
| `references.bib` | Bibliography; `% VERIFY` marks entries not checked against a primary source |

## Conventions

- **Prose never types a measured number.** It uses a macro from `numbers.tex`, and the
  macro's comment names the record under `docs/measurements/` it came from. Tables carry
  their source record in a comment above the table.
- **Every claim has a status.** `[M]` means measured and recorded, and `[P]` means a probe
  reproduced by a script but not yet recorded. Section 5 is entirely *planned*, and the
  paper says so.
- **Rates name their unit.** Every rate is per case group, and absolute detection is
  deflated by the ladder's single-column rungs (7/9 ceiling), as the paper states.
- Draft-only remarks use `\note{...}`, which prints in red. Set `\draftfalse` in
  `main.tex` and run `make check` before submitting.

## Before submission

- [ ] Record the symmetry probe (the six rolled, reversed, sorted and negated rows of
      Table 3) under `docs/measurements/`, and change its tag in `numbers.tex` from
      `[P]` to `[M]`.
- [ ] Resolve every `% VERIFY` in `references.bib` against a primary source.
- [ ] Re-run the prior-art search. The area produces about one competing preprint a
      month (see `reference/PBT-property-based-testing/NOTES.md` §7.3).
- [ ] Install the AAAI author kit, rebuild, and confirm the page count against the
      workshop's limit.
- [ ] Replace "Anonymous Submission" only if the workshop is not double-blind.
- [ ] If E1–E4 have run by then, replace Section 5's predictions with results. Do not
      edit the pre-registered criteria.
