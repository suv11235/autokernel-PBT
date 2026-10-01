"""Structure of the Triton ladder, checked on CPU against stubbed torch and triton.

Nothing here launches a kernel: tile derivation and the launcher's refusals all happen
before the first call into either library.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from autokernel_pbt.props.tasks import TASKS


@pytest.fixture
def ladder(device_stubs):
    return importlib.import_module("kernels.triton.ladder")


def test_block_for_rounds_up_to_a_power_of_two(ladder):
    assert [ladder.block_for(n) for n in (1, 7, 8, 129)] == [1, 8, 8, 256]


def test_a_row_wider_than_max_block_is_refused_as_a_bad_call(ladder):
    """`MAX_BLOCK` was documented as enforced and was not.

    The launcher's only guard compared the row to the tile, and the tile is derived
    from the row, so it could never fire. Refused at construction -- outside
    `Backend.run`, which would book a launcher's ValueError as a kernel's LAUNCH_ERROR
    -- because a misconfigured kernel costs nothing to fix and is not a finding.
    """
    with pytest.raises(ValueError, match=r"n_cols=16385 needs BLOCK=32768 > MAX_BLOCK=16384"):
        ladder.block_for(ladder.MAX_BLOCK + 1)


def test_the_launcher_refuses_a_tile_above_max_block(ladder):
    # Defence in depth for a hand-built constexprs dict that bypassed `block_for`.
    # The stub torch has no `as_tensor`, so a launcher that got past the guard dies
    # with AttributeError instead and fails this test.
    launch = ladder._launcher(object())
    with pytest.raises(ValueError, match=r"BLOCK=32768 exceeds MAX_BLOCK=16384"):
        launch(
            grid=(1,),
            constexprs={"BLOCK": 2 * ladder.MAX_BLOCK},
            record_compiled=lambda compiled: None,
            x=np.ones((1, 8), dtype=np.float32),
        )


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_every_shape_in_use_fits_under_the_cap(ladder, task_id):
    # The cap is inclusive: tolerance_sweep's widest row is exactly 16384.
    for shape in TASKS[task_id].domain.shapes:
        assert ladder.block_for(shape[-1]) <= ladder.MAX_BLOCK


def test_the_multi_tile_launcher_accepts_a_row_wider_than_max_block(ladder):
    # Rows wider than one tile are what `softmax_wide_kernel` is for -- the recorded
    # hybrid-tree measurement ran it to 262144 columns. Neither guard may apply to it.
    # Getting as far as the (stubbed-out) host-to-device copy is the proof it was not
    # refused.
    launch = ladder._wide_launcher(object())
    with pytest.raises(AttributeError, match="as_tensor"):
        launch(
            grid=(1,),
            constexprs=ladder.softmax_wide_kernel(1024).constexprs,
            record_compiled=lambda compiled: None,
            x=np.ones((1, 4 * ladder.MAX_BLOCK), dtype=np.float32),
        )
