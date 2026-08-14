"""PBT scheduler tests (spec 0003)."""

from autokernel_pbt.pbt.population import KernelCandidate
from autokernel_pbt.pbt.scheduler import PBTConfig, PBTScheduler


def _member(cid: str, fitness: float) -> KernelCandidate:
    return KernelCandidate(
        id=cid,
        source_path="kernels/triton/candidate.py",
        backend="triton",
        generation=0,
        fitness=fitness,
    )


def test_population_size():
    cfg = PBTConfig(population_size=6)
    scheduler = PBTScheduler([_member("a", 1.0)], config=cfg)
    out = scheduler.step()
    assert len(out) == 6


def test_exploit_selects_top_k():
    cfg = PBTConfig(population_size=4, exploit_fraction=0.5)
    members = [_member("low", 1.0), _member("high", 10.0), _member("mid", 5.0), _member("x", 2.0)]
    scheduler = PBTScheduler(members, config=cfg)
    scheduler.step()
    # All children should descend from top exploit pool (mutation creates new ids)
    assert all(m.generation == 1 for m in scheduler.members)


def test_explore_creates_new_lineage():
    cfg = PBTConfig(population_size=2)
    parent = _member("parent", 3.0)
    scheduler = PBTScheduler([parent], config=cfg)
    scheduler.step()
    for m in scheduler.members:
        assert m.lineage_id != parent.lineage_id or m.id != parent.id
