"""Three-phase instantaneous P/Q measurement and a discrete first-order LPF."""

from __future__ import annotations

import math


class PowerMeasurementLPF:
    """Measure three-phase P/Q from amplitude-invariant alpha-beta signals.

    The alpha-beta inputs are peak phase-equivalent quantities with zero
    sequence omitted.  Accordingly, the three-phase instantaneous powers are
    ``p = 3/2 (v_alpha*i_alpha + v_beta*i_beta)`` and
    ``q = 3/2 (v_beta*i_alpha - v_alpha*i_beta)``.

    The measurement filter uses forward Euler with ``alpha = 2*pi*f_c*dt``.
    Requiring ``0 < alpha <= 1`` keeps the discrete response stable and
    non-overshooting for a step input.  ``initial_p_w`` and ``initial_q_var``
    allow a benchmark to start at a declared steady operating point rather
    than creating an artificial filter startup transient.
    """

    POWER_SCALE = 1.5

    def __init__(
        self,
        cutoff_freq_hz: float = 10.0,
        dt: float = 1e-3,
        initial_p_w: float = 0.0,
        initial_q_var: float = 0.0,
    ) -> None:
        values = (cutoff_freq_hz, dt, initial_p_w, initial_q_var)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("filter parameters and initial states must be finite")

        self.cutoff_freq_hz = float(cutoff_freq_hz)
        self.wc = 2.0 * math.pi * self.cutoff_freq_hz
        self.dt = float(dt)
        if self.cutoff_freq_hz <= 0.0:
            raise ValueError("cutoff_freq_hz must be greater than zero")
        if self.dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        self.alpha = self.wc * self.dt
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                "2*pi*cutoff_freq_hz*dt must be in (0, 1] for a non-overshooting Euler LPF"
            )

        self.p_filtered = float(initial_p_w)
        self.q_filtered = float(initial_q_var)

    def compute_instantaneous_power(
        self,
        v_alpha: float,
        v_beta: float,
        i_alpha: float,
        i_beta: float,
    ) -> tuple[float, float]:
        """Calculate three-phase instantaneous P (W) and Q (var)."""
        values = (v_alpha, v_beta, i_alpha, i_beta)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("voltage and current components must be finite")

        v_alpha, v_beta, i_alpha, i_beta = map(float, values)
        p_inst = self.POWER_SCALE * (v_alpha * i_alpha + v_beta * i_beta)
        q_inst = self.POWER_SCALE * (v_beta * i_alpha - v_alpha * i_beta)
        if not (math.isfinite(p_inst) and math.isfinite(q_inst)):
            raise FloatingPointError("non-finite instantaneous power")
        return p_inst, q_inst

    def update(
        self,
        v_alpha: float,
        v_beta: float,
        i_alpha: float,
        i_beta: float,
    ) -> tuple[float, float]:
        """Update the discrete LPF from simultaneous voltage/current samples."""
        p_inst, q_inst = self.compute_instantaneous_power(
            v_alpha=v_alpha,
            v_beta=v_beta,
            i_alpha=i_alpha,
            i_beta=i_beta,
        )
        self.p_filtered += self.alpha * (p_inst - self.p_filtered)
        self.q_filtered += self.alpha * (q_inst - self.q_filtered)
        if not (math.isfinite(self.p_filtered) and math.isfinite(self.q_filtered)):
            raise FloatingPointError("non-finite filtered power")
        return self.p_filtered, self.q_filtered
