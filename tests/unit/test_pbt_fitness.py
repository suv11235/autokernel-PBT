"""PBT fitness tests (spec 0003)."""

from autokernel_pbt.pbt.fitness import FAILED_FITNESS, fitness_from_harness_result


def test_failed_candidate_low_fitness(load_fixture):
    result = load_fixture("harness_result_failed.json")
    assert fitness_from_harness_result(result) == FAILED_FITNESS


def test_success_candidate_uses_speedup(load_fixture):
    result = load_fixture("harness_result_success.json")
    assert fitness_from_harness_result(result) == 1.25
