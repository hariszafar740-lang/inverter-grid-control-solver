import matplotlib.pyplot as plt

def plot_inverter_response(time_array, f_vsm, p_filt, v_mag, save_path='results/inverter_transient_response.png'):
    """
    Renders 3-panel dynamic transient response for Grid-Forming Inverter:
    1. VSM Output Frequency Response f_vsm(t) vs Nominal 50 Hz Baseline
    2. Low-Pass Filtered Active Power P_filt(t)
    3. Virtual Impedance Modified Peak Voltage Magnitude |V_ref*|(t)
    """
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    # 1. Frequency Dynamics & Nadir
    axs[0].plot(time_array, f_vsm, color='#1f77b4', linewidth=1.6, label='VSM Frequency f(t)')
    axs[0].axhline(y=50.0, color='black', linestyle='--', alpha=0.7, label='Nominal Frequency (50 Hz)')
    axs[0].set_ylabel('Frequency (Hz)', fontsize=9, fontweight='bold')
    axs[0].set_title('Grid-Forming Inverter Dynamic Transient Response', fontsize=11, fontweight='bold')
    axs[0].grid(True, ls='--', alpha=0.5)
    axs[0].legend(loc='upper right')

    # 2. Filtered Active Power Step Dynamics
    axs[1].plot(time_array, [p / 1000.0 for p in p_filt], color='#2ca02c', linewidth=1.6, label='Active Power P_filt(t)')
    axs[1].set_ylabel('Power (kW)', fontsize=9, fontweight='bold')
    axs[1].grid(True, ls='--', alpha=0.5)
    axs[1].legend(loc='upper right')

    # 3. Virtual Impedance Voltage Magnitude Drop
    axs[2].plot(time_array, v_mag, color='#d62728', linewidth=1.6, label='Peak Voltage Magnitude |V_ref*|(t)')
    axs[2].set_xlabel('Time (Seconds)', fontsize=10, fontweight='bold')
    axs[2].set_ylabel('Voltage (V)', fontsize=9, fontweight='bold')
    axs[2].grid(True, ls='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
