"""Pure decision logic for the legacy generated-terrain curriculum."""

from __future__ import annotations


def terrain_curriculum_decisions(age_s, forward_m, failed):
    """Return ``move_up, move_down`` without reading reset-cleared episode state.

    ``forward_m`` is displacement projected onto the heading recorded when the
    episode began. Lateral motion and backward motion therefore cannot promote
    an environment. A promotion also requires at least three seconds of valid
    progress; an early physical failure demotes it.

    The arguments may be Python scalars, NumPy arrays, or PyTorch tensors. All
    comparisons are elementwise and preserve the input backend.
    """
    # ``failed == False`` works for Python booleans and tensor booleans, while
    # bitwise ``~True`` would produce the integer -2 for a scalar call.
    move_up = (failed == False) & (age_s >= 3.0) & (forward_m >= 2.0)
    move_down = failed & (age_s < 2.0)
    return move_up, move_down
