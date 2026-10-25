"""Grid-Forming Inverter Control & Thévenin Network Dynamic Solver."""

from typing import TYPE_CHECKING

from .current_control import apply_current_limit, current_reference_from_power
from .droop_control import InverterDroopController
from .power_measurement import PowerMeasurementLPF
from .thevenin_network import (
    ClosedLoopTransientResult,
    CoupledVSGTheveninModel,
    TheveninGridSpec,
    simulate_coupled_vsg_transient,
)
from .virtual_impedance import VirtualImpedanceController
from .virtual_inertia import VirtualInertiaEngine
from .visualizer import plot_closed_loop_thevenin_benchmark, plot_inverter_response

if TYPE_CHECKING:
    from .benchmark import (
        BenchmarkAcceptanceError,
        calculate_rocof_windows,
        run_fault_ride_through_benchmark,
        validate_frt_benchmark_results,
    )

__version__ = "0.2.0"

_BENCHMARK_EXPORTS = frozenset(
    {
        "BenchmarkAcceptanceError",
        "calculate_rocof_windows",
        "run_fault_ride_through_benchmark",
        "validate_frt_benchmark_results",
    }
)

__all__ = [
    "BenchmarkAcceptanceError",
    "ClosedLoopTransientResult",
    "CoupledVSGTheveninModel",
    "InverterDroopController",
    "PowerMeasurementLPF",
    "TheveninGridSpec",
    "VirtualImpedanceController",
    "VirtualInertiaEngine",
    "__version__",
    "apply_current_limit",
    "calculate_rocof_windows",
    "current_reference_from_power",
    "plot_closed_loop_thevenin_benchmark",
    "plot_inverter_response",
    "run_fault_ride_through_benchmark",
    "simulate_coupled_vsg_transient",
    "validate_frt_benchmark_results",
]


def __getattr__(name: str):
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
