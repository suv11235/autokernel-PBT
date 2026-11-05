# e3b-instructed-red-team

**Question.** E3's agents never wrote a wrong kernel, so they never tested the gate. E3b tells the agent to beat it, and
asks what gets through: arm A on the current bundle, arm B on the completed bundle with the gate undisclosed, and arm C on
the completed bundle with everything disclosed except the hidden checker.

**Record:** [`docs/measurements/2026-11-04-e3b-instructed-red-team-prereg.md`](../../../docs/measurements/2026-11-04-e3b-instructed-red-team-prereg.md),
pre-registered before this loop exists.

**Status: [planned].** The loop is not built, and no agent has run.

## Known attacks

`recorded/known_attacks.md` is what E3's instrument does with attacks written by hand, and an audit of E3's candidates for
the holes those attacks found. The record's §2 cites it. No model is involved.

```bash
../../../.venv/bin/python known_attacks.py > recorded/known_attacks.md
```
