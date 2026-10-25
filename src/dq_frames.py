"""Amplitude-invariant Clarke (alpha-beta) and synchronous Park (dq) coordinate transforms.

All functions use the amplitude-invariant three-phase convention consistent across the
solver: balanced phase peak magnitude ``V_pk`` maps to ``|v_alphabeta| = V_pk`` and
``v_dq = [V_pk, 0]^T`` when aligned with ``theta``, so three-phase instantaneous active
and reactive power satisfy:

    P = 1.5 * (v_alpha * i_alpha + v_beta * i_beta) = 1.5 * (v_d * i_d + v_q * i_q)
    Q = 1.5 * (v_beta  * i_alpha - v_alpha * i_beta) = 1.5 * (v_q * i_d - v_d * i_q)
"""

from __future__ import annotations

import math

SQRT3 = math.sqrt(3.0)
INV_SQRT3 = 1.0 / SQRT3


def abc_to_alphabeta(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Amplitude-invariant Clarke transform ``(a, b, c) -> (alpha, beta, zero)``."""
    vals = (float(a), float(b), float(c))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("abc inputs must be finite numbers")
    va, vb, vc = vals
    alpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
    beta = (2.0 / 3.0) * (0.5 * SQRT3 * (vb - vc))
    zero = (va + vb + vc) / 3.0
    return alpha, beta, zero


def alphabeta_to_abc(alpha: float, beta: float, zero: float = 0.0) -> tuple[float, float, float]:
    """Inverse amplitude-invariant Clarke transform ``(alpha, beta, zero) -> (a, b, c)``."""
    vals = (float(alpha), float(beta), float(zero))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("alphabeta0 inputs must be finite numbers")
    al, be, z0 = vals
    va = al + z0
    vb = -0.5 * al + 0.5 * SQRT3 * be + z0
    vc = -0.5 * al - 0.5 * SQRT3 * be + z0
    return va, vb, vc


def alphabeta_to_dq(alpha: float, beta: float, theta_rad: float) -> tuple[float, float]:
    """Park rotation from stationary ``(alpha, beta)`` into frame at angle ``theta_rad``."""
    vals = (float(alpha), float(beta), float(theta_rad))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("alpha, beta, and theta_rad must be finite")
    al, be, th = vals
    c = math.cos(th)
    s = math.sin(th)
    return c * al + s * be, -s * al + c * be


def dq_to_alphabeta(d_val: float, q_val: float, theta_rad: float) -> tuple[float, float]:
    """Inverse Park rotation from synchronous ``(d, q)`` at ``theta_rad`` to ``(alpha, beta)``."""
    vals = (float(d_val), float(q_val), float(theta_rad))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("d_val, q_val, and theta_rad must be finite")
    dv, qv, th = vals
    c = math.cos(th)
    s = math.sin(th)
    return c * dv - s * qv, s * dv + c * qv


def rotate_dq_frame(d_src: float, q_src: float, delta_rad: float) -> tuple[float, float]:
    """Transform ``(d, q)`` from reference frame ``theta_1`` to ``theta_2 = theta_1 + delta_rad``."""
    return alphabeta_to_dq(d_src, q_src, delta_rad)


def compute_pq_dq(v_d: float, v_q: float, i_d: float, i_q: float) -> tuple[float, float]:
    """Three-phase active and reactive power ``(P_w, Q_var)`` in amplitude-invariant ``dq``."""
    vals = (float(v_d), float(v_q), float(i_d), float(i_q))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("dq voltage and current inputs must be finite")
    vd, vq, id_, iq_ = vals
    p_w = 1.5 * (vd * id_ + vq * iq_)
    q_var = 1.5 * (vq * id_ - vd * iq_)
    return p_w, q_var
