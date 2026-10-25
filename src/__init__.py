"""Grid-Forming Inverter Control, Electromagnetic LCL Plant & Thévenin Network Solver."""

from typing import TYPE_CHECKING

from .cascaded_control import (
    CascadedPIConfig,
    ElectromagneticGFMModel,
    ElectromagneticTransientResult,
    simulate_electromagnetic_gfm_transient,
)
from .current_control import apply_current_limit, current_reference_from_power
from .dq_frames import (
    abc_to_alphabeta,
    alphabeta_to_abc,
    alphabeta_to_dq,
    compute_pq_dq,
    dq_to_alphabeta,
    rotate_dq_frame,
)
from .droop_control import InverterDroopController
from .lcl_plant import LCLFilterSpec, LCLGridPlant, simulate_lcl_damping_step
from .power_measurement import PowerMeasurementLPF
from .thevenin_network import (
    ClosedLoopTransientResult,
    CoupledVSGTheveninModel,
    TheveninGridSpec,
    simulate_coupled_vsg_transient,
)
from .virtual_impedance import VirtualImpedanceController
from .virtual_inertia import VirtualInertiaEngine
from .visualizer import (
    plot_closed_loop_thevenin_benchmark,
    plot_inverter_response,
    plot_lcl_cascaded_electromagnetic_benchmark,
)

if TYPE_CHECKING:
    from .benchmark import (
        BenchmarkAcceptanceError,
        calculate_rocof_windows,
        run_fault_ride_through_benchmark,
        validate_frt_benchmark_results,
    )

__version__ = "0.3.0"

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
    "CascadedPIConfig",
    "ClosedLoopTransientResult",
    "CoupledVSGTheveninModel",
    "ElectromagneticGFMModel",
    "ElectromagneticTransientResult",
    "InverterDroopController",
    "LCLFilterSpec",
    "LCLGridPlant",
    "PowerMeasurementLPF",
    "TheveninGridSpec",
    "VirtualImpedanceController",
    "VirtualInertiaEngine",
    "__version__",
    "abc_to_alphabeta",
    "alphabeta_to_abc",
    "alphabeta_to_dq",
    "apply_current_limit",
    "calculate_rocof_windows",
    "compute_pq_dq",
    "current_reference_from_power",
    "dq_to_alphabeta",
    "plot_closed_loop_thevenin_benchmark",
    "plot_inverter_response",
    "plot_lcl_cascaded_electromagnetic_benchmark",
    "rotate_dq_frame",
    "run_fault_ride_through_benchmark",
    "simulate_coupled_vsg_transient",
    "simulate_electromagnetic_gfm_transient",
    "simulate_lcl_damping_step",
    "validate_frt_benchmark_results",
]


def __getattr__(name: str):
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
