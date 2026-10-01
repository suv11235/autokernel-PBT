"""Kernel implementations: Triton ports, mutants and correct variants.

A REGULAR package, deliberately, not a namespace package. The import system prefers a
regular package to a namespace portion regardless of sys.path order, so while this
file was absent any installed distribution named `kernels` (HuggingFace ships one)
shadowed this directory no matter where the repo root sat on sys.path.

Nothing is imported here: the submodules import torch and triton at module scope, and
`kernels.mutants.taxonomy` must stay importable on a machine with neither.
"""
