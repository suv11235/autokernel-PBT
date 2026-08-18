# Protocol: authoring a mutant

This file is the experimental method, not documentation. It is version-controlled so the paper
can quote it, and so a reader can judge what the authoring agent could and could not see.

## What the agent receives

Exactly two things:

1. The ISSTA subcategory description for the fault class, quoted verbatim from Table 2 of
   `reference/PBT-property-based-testing/papers/2605.19652.pdf`.
2. The correct reference implementation for the task, from `src/autokernel_pbt/props/tasks.py`.

## What the agent must NOT receive

- `kernels/tasks/<id>/acceptance.yaml`, or any property name
- `src/autokernel_pbt/props/properties.py`
- Any tolerance, threshold, or `residual_ratio` detail
- Any prior mutant, or any detection result

A mutant written with the property checklist in hand tells you only that the checklist matches
itself. **This list is the difference between a measurement and a tautology.**

## The prompt

> Here is a correct implementation of `<task>`:
>
> ```python
> <reference source>
> ```
>
> Here is a description of a class of real bug found in GPU tile programs:
>
> > <verbatim ISSTA subcategory description>
>
> Write a modified version that exhibits that class of bug. Keep the same function signature.
> The result should look like plausible code someone might write, not an obviously broken stub.
> Return only the function.

## What is recorded per candidate

- the task, the subcategory, and the verbatim description given
- the returned source
- the gate verdict and, if rejected, the reason

## What is NOT claimed

The fault class is **intended**, established by what the prompt asked for. Nothing verifies that
the returned kernel exhibits that class rather than another. Automatic defect classification is a
research problem of its own, and a weak classifier would mislabel exactly the cases that matter.
Any per-class number carries this caveat.
