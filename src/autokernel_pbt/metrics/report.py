"""Rendering rates so the caveats travel with the numbers.

A reader who sees only a table takes it at face value. Three facts materially change
how these numbers should be read, and all three live in documents a reader may never
open -- so `render` puts them above the table instead.
"""

from __future__ import annotations

from autokernel_pbt.metrics.rates import ArmRates

#: Measured end to end through the driver: 7 of 9 groups, the other two being the
#: single-column ladder rungs where softmax is identically 1.0 for any implementation
#: and layernorm's variance property abstains on a constant input.
LADDER_DEFLATION = "7/9 = 0.778"

_HEADER = f"""## Detection rates

Three things change how these numbers read, and none of them is visible in the table.

**The ladder deflates every absolute rate.** Degenerate shapes (1,1) and (17,1) make softmax
identically 1.0 and make layernorm's variance property abstain, so a kernel that is genuinely
broken is genuinely correct there. Measured ceiling: {LADDER_DEFLATION}. Arm-versus-arm
comparison is unaffected -- the deflation applies to every arm equally -- but no absolute rate
here should be read as a fraction of bugs found.

**The fault class is *intended*, not verified.** It is established by what the authoring prompt
asked for. Nothing checks that a mutant exhibits that class rather than another, so a per-class
row is a statement about what was requested.

**Corpus size is small.** With one mutant per class, a per-class rate cannot distinguish "this
class does not differentiate the arms" from "this kernel does not".
"""


def render(rates: dict[str, dict[str, ArmRates]], *, backend: str) -> str:
    """A Markdown report: rows are mutants, columns are arms.

    `rates` is mutant id -> arm name -> rates. An empty mapping still renders the
    header, because the caveats are what the header exists for and a report with no
    rows is still read.
    """
    lines = [_HEADER, f"\n**Backend:** `{backend}`\n"]
    if not rates:
        lines.append("_No runs scored._\n")
        return "\n".join(lines)

    arms = sorted({arm for per_arm in rates.values() for arm in per_arm})
    lines.append("| mutant (intended class) | " + " | ".join(arms) + " | tolerance-free |")
    lines.append("|---" * (len(arms) + 2) + "|")
    for mutant in sorted(rates):
        per_arm = rates[mutant]
        cells = [f"{per_arm[a].detection_rate:.3f}" if a in per_arm else "—" for a in arms]
        declarative = per_arm.get("declarative")
        tf = f"{declarative.tolerance_free_detection_rate:.3f}" if declarative else "—"
        lines.append(f"| {mutant} | " + " | ".join(cells) + f" | {tf} |")
    return "\n".join(lines) + "\n"
