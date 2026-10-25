"""Grid-Forming & Grid-Following Inverter Control, Electromagnetic LCL Plant & FRT Solver."""

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
from .dvoc_control import (
    BlackStartExcitationResult,
    DispatchableVirtualOscillator,
    DVOCConfig,
)
from .em_frt_benchmark import (
    ElectromagneticFRTBenchmarkResult,
    ElectromagneticFRTTrajectory,
    ElectromagneticFRTVSGModel,
    ProtectionMode,
    run_electromagnetic_frt_benchmark,
    simulate_em_fault_sag,
    simulate_em_phase_jump,
)
from .gfl_pll import (
    ElectromagneticGFLModel,
    SRFPLLConfig,
    SynchronousReferenceFramePLL,
)
from .gfm_triad import (
    GFMMode,
    GFMTriadBenchmarkResult,
    GFMTriadMemberResult,
    MultiLoopDroopGFMModel,
    MultiLoopDVOCGFMModel,
    run_gfm_triad_benchmark,
    simulate_gfm_mode_transient,
)
from .lcl_plant import LCLFilterSpec, LCLGridPlant, simulate_lcl_damping_step
from .power_measurement import PowerMeasurementLPF
from .thevenin_network import (
    ClosedLoopTransientResult,
    CoupledVSGTheveninModel,
    TheveninGridSpec,
    simulate_coupled_vsg_transient,
)
from .transient_virtual_impedance import TransientVirtualImpedance, TVIConfig
from .virtual_impedance import VirtualImpedanceController
from .virtual_inertia import VirtualInertiaEngine
from .visualizer import (
    plot_closed_loop_thevenin_benchmark,
    plot_electromagnetic_frt_benchmark,
    plot_gfm_triad_benchmark,
    plot_inverter_response,
    plot_lcl_cascaded_electromagnetic_benchmark,
    plot_weak_grid_scr_benchmark,
)
from .weak_grid_benchmark import (
    SCRStabilityPoint,
    WeakGridArchitecture,
    WeakGridBenchmarkResult,
    WeakGridLineTripResult,
    estimate_critical_scr,
    run_weak_grid_scr_benchmark,
    simulate_weak_grid_line_trip,
    sweep_scr_stability,
)

if TYPE_CHECKING:
    from .benchmark import (
        BenchmarkAcceptanceError,
        calculate_rocof_windows,
        run_fault_ride_through_benchmark,
        validate_frt_benchmark_results,
    )

__version__ = "0.6.0"

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
    "BlackStartExcitationResult",
    "CascadedPIConfig",
    "ClosedLoopTransientResult",
    "CoupledVSGTheveninModel",
    "DVOCConfig",
    "DispatchableVirtualOscillator",
    "ElectromagneticFRTBenchmarkResult",
    "ElectromagneticFRTTrajectory",
    "ElectromagneticFRTVSGModel",
    "ElectromagneticGFLModel",
    "ElectromagneticGFMModel",
    "ElectromagneticTransientResult",
    "GFMMode",
    "GFMTriadBenchmarkResult",
    "GFMTriadMemberResult",
    "InverterDroopController",
    "LCLFilterSpec",
    "LCLGridPlant",
    "MultiLoopDVOCGFMModel",
    "MultiLoopDroopGFMModel",
    "PowerMeasurementLPF",
    "ProtectionMode",
    "SCRStabilityPoint",
    "SRFPLLConfig",
    "SynchronousReferenceFramePLL",
    "TVIConfig",
    "TheveninGridSpec",
    "TransientVirtualImpedance",
    "VirtualImpedanceController",
    "VirtualInertiaEngine",
    "WeakGridArchitecture",
    "WeakGridBenchmarkResult",
    "WeakGridLineTripResult",
    "__version__",
    "abc_to_alphabeta",
    "alphabeta_to_abc",
    "alphabeta_to_dq",
    "apply_current_limit",
    "calculate_rocof_windows",
    "compute_pq_dq",
    "current_reference_from_power",
    "dq_to_alphabeta",
    "estimate_critical_scr",
    "plot_closed_loop_thevenin_benchmark",
    "plot_electromagnetic_frt_benchmark",
    "plot_gfm_triad_benchmark",
    "plot_inverter_response",
    "plot_lcl_cascaded_electromagnetic_benchmark",
    "plot_weak_grid_scr_benchmark",
    "rotate_dq_frame",
    "run_electromagnetic_frt_benchmark",
    "run_fault_ride_through_benchmark",
    "run_gfm_triad_benchmark",
    "run_weak_grid_scr_benchmark",
    "simulate_coupled_vsg_transient",
    "simulate_electromagnetic_gfm_transient",
    "simulate_em_fault_sag",
    "simulate_em_phase_jump",
    "simulate_gfm_mode_transient",
    "simulate_lcl_damping_step",
    "simulate_weak_grid_line_trip",
    "sweep_scr_stability",
    "validate_frt_benchmark_results",
]


def __getattr__(name: str):
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
