import os
import numpy as np
from src.visualizer import plot_inverter_response

def test_plot_inverter_response_generation(tmp_path):
    time_array = np.linspace(0, 1.0, 100)
    f_vsm = np.ones(100) * 50.0
    p_filt = np.linspace(0, 10000, 100)
    v_mag = np.ones(100) * 325.27
    
    save_file = tmp_path / "test_inverter_plot.png"
    plot_inverter_response(time_array, f_vsm, p_filt, v_mag, save_path=str(save_file))
    assert os.path.exists(save_file)
