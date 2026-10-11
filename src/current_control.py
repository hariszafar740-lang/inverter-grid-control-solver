"""Positive-sequence P/Q current references and a radial current limiter.

The alpha-beta inputs use amplitude-invariant Clarke coordinates (peak phase
quantities, zero sequence omitted), so three-phase power is 3/2 times the
alpha-beta dot/cross products.  This is an algebraic, quasi-static control
block; it is not a PWM, semiconductor, or inner-current-loop model.
"""

from __future__ import annotations

import math

POWER_SCALE = 1.5


def current_reference_from_power(
    v_alpha_v: float,
    v_beta_v: float,
    p_ref_w: float,
    q_ref_var: float,
) -> tuple[float, float]:
    """Return alpha-beta current references for requested three-phase P and Q.

    The references solve

        P = 3/2 (v_alpha i_alpha + v_beta i_beta)
        Q = 3/2 (v_beta i_alpha - v_alpha i_beta).

    At zero grid voltage there is no voltage-aligned direction from which to
    construct a P/Q current command.  This reduced-order model sets the command
    to zero; actual fault-current behavior at zero voltage requires a specified
    inner controller and is outside this model.
    """
    values = (v_alpha_v, v_beta_v, p_ref_w, q_ref_var)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("voltage and P/Q references must be finite")

    v_alpha_v, v_beta_v, p_ref_w, q_ref_var = map(float, values)
    voltage_magnitude_v = math.hypot(v_alpha_v, v_beta_v)
    if voltage_magnitude_v == 0.0:
        return 0.0, 0.0

    # Use the unit-voltage components to avoid squaring a large voltage value.
    v_alpha_unit = v_alpha_v / voltage_magnitude_v
    v_beta_unit = v_beta_v / voltage_magnitude_v
    denominator = POWER_SCALE * voltage_magnitude_v
    i_alpha_ref_a = (
        v_alpha_unit * p_ref_w + v_beta_unit * q_ref_var
    ) / denominator
    i_beta_ref_a = (
        v_beta_unit * p_ref_w - v_alpha_unit * q_ref_var
    ) / denominator

    if not (math.isfinite(i_alpha_ref_a) and math.isfinite(i_beta_ref_a)):
        raise FloatingPointError("non-finite current reference generated")
    return i_alpha_ref_a, i_beta_ref_a


def apply_current_limit(
    i_alpha_ref_a: float,
    i_beta_ref_a: float,
    max_current_a: float,
) -> tuple[float, float, bool]:
    """Radially limit an alpha-beta current command to its peak magnitude.

    If limiting is active, both components are scaled by the same factor.  This
    preserves the current-vector angle and scales P and Q together; it does not
    implement active- or reactive-current priority logic.
    """
    values = (i_alpha_ref_a, i_beta_ref_a, max_current_a)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("current references and current limit must be finite")

    i_alpha_ref_a, i_beta_ref_a, max_current_a = map(float, values)
    if max_current_a <= 0.0:
        raise ValueError("max_current_a must be greater than zero")

    current_magnitude_a = math.hypot(i_alpha_ref_a, i_beta_ref_a)
    if not math.isfinite(current_magnitude_a):
        raise FloatingPointError("non-finite current magnitude")
    if current_magnitude_a <= max_current_a:
        return i_alpha_ref_a, i_beta_ref_a, False

    scale = max_current_a / current_magnitude_a
    i_alpha_a = i_alpha_ref_a * scale
    i_beta_a = i_beta_ref_a * scale
    if not (math.isfinite(i_alpha_a) and math.isfinite(i_beta_a)):
        raise FloatingPointError("non-finite limited current")
    return i_alpha_a, i_beta_a, True
