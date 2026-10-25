import numpy as np
import plotly.graph_objects as go
import matplotlib.lines as mlines    
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output

from IPython.display import Javascript, display
import gc

from scipy.signal import butter, filtfilt
from neuron import h
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

def plot_var_rgc_over_time(v_recorders, stim_amplitude, stim_frequency, t_stim, t_np, fig_show=True, save=False, return_fig=False):
    #original one
    plot_data = []
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2: continue
        
        xs = [h.x3d(i, sec=sec) for i in range(n3d)]
        ys = [h.y3d(i, sec=sec) for i in range(n3d)]
        zs = [h.z3d(i, sec=sec) for i in range(n3d)]
        
        # Access the dictionary using the exact same string key
        name_key = f"{sec.name()}(0.5)"
        
        if name_key in v_recorders:
            v_series = np.array(v_recorders[name_key])
            v_smooth = np.array(smoothened_membrane_potential(v_recorders[name_key], stim_frequency, h.dt))
            spike_rate = get_spike_rate(np.array(v_recorders[name_key]), t_stim)
            plot_data.append({
                'x': xs, 'y': ys, 'z': zs,
                'v_series': v_series,
                'spike_rate': spike_rate,
                'v_smooth': v_smooth,
                'name': sec.name()
            })

    step = int(2.0 / h.dt)
    time_indices = np.arange(0, len(t_np), step)

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.01, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Smoothened Potential (mV)", "Spike Rate (Hz)")
    )

    for branch in plot_data:
        actual_diam = h.Section(name=branch['name']).diam
        
        # We use a non-linear mapping so the soma is thick but not 'humongous'
        if "soma" in branch['name']:
            visual_width = 30#10  # Distinct but contained
        elif "initseg" in branch['name'] or "narrowr" in branch['name']:
            visual_width = 4   # Slightly thicker than dendrites
        else:
            visual_width = 2   # Fine lines for distal dendrites
        
        # Membrane Potential
        fig.add_trace(go.Scatter3d(
            x=branch['x'], y=branch['y'], z=branch['z'],
            mode='lines',
            line=dict(
                color=np.full(len(branch['x']), branch['v_smooth'][0]),
                colorscale='Magma', 
                cmin=-70, cmax=-30, 
                width=visual_width, 
                colorbar=dict(
                        title="mV",
                        x=0.45,       # Position in the middle of the two plots
                        thickness=15,
                        len=0.7
                        ),
                showscale=True if branch['name'] == 'soma' else False
            ),
            name=branch['name'],
            showlegend=False,
            ),
            row=1, col=1
        )

        # Spike Rate
        fig.add_trace(go.Scatter3d(
            x=branch['x'], y=branch['y'], z=branch['z'],
            mode='lines',
            line=dict(
                color=np.full(len(branch['x']), branch['spike_rate'][0]),
                colorscale='Magma',
                cmin = 0, cmax = 300,
                width=visual_width, 
                colorbar=dict(
                        title="mV",
                        x=1.02,       # Position in the middle of the two plots
                        thickness=15,
                        len=0.7
                        ),
                showscale=True if branch['name'] == 'soma' else False
            ),
            name=branch['name'],
            showlegend=False,
            ),
            row=1, col=2
        )

    # IMPORTANT: Each frame must contain the FULL data list for all traces
    frames = []
    for k in time_indices:
        t_curr = k * h.dt
        
        hfs_status = "<b>HFS: ON</b>" if 100 <= t_curr <= 900 else "HFS: OFF"
        pos_status = "<b>POS CC: ACTIVE</b>" if 300 <= t_curr <= 500 else "POS CC: inactive"
        neg_status = "<b>NEG CC: ACTIVE</b>" if 600 <= t_curr <= 800 else "NEG CC: inactive"
        
        # Build the data for this frame (Geometry + Color)
        frame_data = []
        for branch in plot_data:
            frame_data.append(go.Scatter3d(
                x=branch['x'], y=branch['y'], z=branch['z'],
                line=dict(color=np.full(len(branch['x']), branch['v_smooth'][k]))
            ))
            frame_data.append(go.Scatter3d(
                x=branch['x'], y=branch['y'], z=branch['z'],
                line=dict(color=np.full(len(branch['x']), branch['spike_rate'][k]))
            ))

        frames.append(go.Frame(
            data=frame_data,
            name=str(k),
            layout=go.Layout(annotations=[
                dict(x=0.03, y=0.97, xref="paper", yref="paper", text=f"Time: {t_curr:.1f} ms", showarrow=False, font=dict(size=16)),
                dict(x=0.03, y=0.90, xref="paper", yref="paper", text=hfs_status, showarrow=False, font=dict(color="red" if "ON" in hfs_status else "black")),
                dict(x=0.03, y=0.85, xref="paper", yref="paper", text=pos_status, showarrow=False, font=dict(color="orange" if "ACTIVE" in pos_status else "gray")),
                dict(x=0.03, y=0.80, xref="paper", yref="paper", text=neg_status, showarrow=False, font=dict(color="blue" if "ACTIVE" in neg_status else "gray"))
            ])
        ))

    fig.frames = frames

    slider_steps = []
    for i, k in enumerate(time_indices):
        t_current = k * h.dt  # Current time in ms
        
        # Determine which annotation strings to display
        hfs_text = "<b>HFS: ON</b>" if 100 <= t_current <= 900 else "HFS: OFF"
        pos_cc = "<b>POS CC: ACTIVE</b>" if 300 <= t_current <= 500 else ""
        neg_cc = "<b>NEG CC: ACTIVE</b>" if 600 <= t_current <= 800 else ""
        
        # Combine active status for the frame title
        status_label = f"{t_current:.1f}ms | {hfs_text} | {pos_cc} | {neg_cc}"
        
        step = {
            "args": [
                [str(k)],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }
            ],
            "label": f"{t_current:.0f} ms",
            "method": "animate"
        }
        slider_steps.append(step)

    fig.update_layout(
        height = 800,
        scene=dict(
            xaxis=dict(range=[-150, 150], title="X (µm)"),
            yaxis=dict(range=[-150, 150], title="Y (µm)"),
            zaxis=dict(range=[-50, 50], title="Z (µm)"),
            aspectmode='cube'
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 10, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "type": "buttons",
            "direction": "left",
            "showactive": False,
            "x": 0, "y": 0
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": slider_steps
        }],
        annotations=[
            # HFS Status (Top Left)
            dict(x=0.05, y=0.95, xref="paper", yref="paper", text="Stimulus Protocol:", showarrow=False, font=dict(size=14)),
            # We will use the 'currentvalue' of the slider to show the active stims
        ],
        margin=dict(l=0, r=0, b=0, t=50)
    )

    camera_top = dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0))
    scene_config = dict(xaxis=dict(range=[-150, 150]), yaxis=dict(range=[-150, 150]), 
                        zaxis=dict(range=[-50, 50]), aspectmode='cube', camera=camera_top)

    fig.update_layout(
        scene=scene_config, scene2=scene_config,
        height=600, width=1000,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
            ],
            "type": "buttons", "x": 0.05, "y": 0
        }],
        sliders=[{
            "steps": slider_steps, "active": 0, "x": 0.15, "len": 0.8,
            "currentvalue": {"prefix": "Time: ", "visible": True, "xanchor": "right"}
        }],
        margin=dict(l=10, r=10, b=100, t=100)
    )

    if fig_show:
        fig.show()
    if save:
        fig.write_html(f"RGC_Video{int(stim_amplitude)} uA {int(stim_frequency)}Hz.html", include_plotlyjs='cdn')
    return fig

def get_spike_rate(v_clean, t):
    peaks, _ = find_peaks(v_clean, height=-10, distance=1.0/h.dt)
    peaks_time = t[peaks]

    window_size = 50.0  # ms
    step_size = h.dt     # ms
    t_rate = np.arange(0, t[-1] + step_size, step_size)
    rate = np.zeros_like(t_rate)

    for i, t_point in enumerate(t_rate):
        t_start, t_end = t_point - window_size/2, t_point + window_size/2
        spike_count = np.sum((peaks_time >= t_start) & (peaks_time < t_end))
        rate[i] = (spike_count / window_size) * 1000
    return rate

def digital_artifact_removal(v_data, sampling_rate_khz, cutoff_khz=0.8):
    """
    Removes high-frequency stimulation artifacts.
    v_data: The recorded v_soma.to_python()
    sampling_rate_khz: 1/h.dt (e.g., if dt=0.01, rate is 100 kHz)
    cutoff_khz: The frequency above which signal is blocked (must be < stim freq)
    """
    nyquist = 0.5 * sampling_rate_khz
    normal_cutoff = cutoff_khz / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    
    return filtfilt(b, a, v_data)

def smoothened_membrane_potential(v_data, hfs_freq, dt_ms):
    """
    Returns a smoothened version of the membrane potential.
    Removes stimulation artefact using a low-pass filter.
    """

    # 1. Setup Parameters
    fs = 1000 / dt_ms         # Sampling rate in Hz (dt is in ms)
    fc = 50                   # Cutoff frequency in Hz
    order = 5                 # Filter order
    dt_sec = dt_ms / 1000.0   # Time step in seconds
    
    # 2. Butterworth Filter (Low-pass)
    # Wn is the normalized frequency: cutoff / Nyquist
    nyquist = fs / 2
    b, a = butter(order, fc / nyquist, btype='low')
    data_filtered = filtfilt(b, a, v_data)
    
    # 3. Moving Average (Smoothened Mean)
    # Window = (100 * HFS period) / dt
    av_window = int((1.0 / hfs_freq) / dt_sec * 20)
    
    if av_window < 1:
        av_window = 1
    
    if av_window > len(data_filtered):
        # Fallback if window is too large
        smooth_mean = np.full_like(data_filtered, np.mean(data_filtered))
    else:
        window = np.ones(av_window) / av_window
        smooth_mean = np.convolve(data_filtered, window, mode='same')
        
    return smooth_mean

def update_rx_3d(el_x, el_y, el_z):
    """
    Calculates transfer resistance (rx) for every segment.
    Uses segment position relative to 3D points for accuracy.
    """
    rho = 100e4#80e4#100e4 #300e4 # ohm-um
    
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2: continue # Skip if no 3D data
        
        # Get total 3D length to allow interpolation
        L3d = h.arc3d(n3d - 1, sec=sec)
        
        for seg in sec:
            # Find the distance along the section for this segment center (0 to L)
            target_arc = seg.x * L3d
            
            # Find the 3D point nearest to this arc length
            # For a 2-point axon, this safely picks the right spot
            x, y, z = 0, 0, 0
            for i in range(n3d):
                if h.arc3d(i, sec=sec) >= target_arc:
                    x = h.x3d(i, sec=sec)
                    y = h.y3d(i, sec=sec)
                    z = h.z3d(i, sec=sec)
                    break
            
            # Calculate distance to electrode
            r = np.sqrt((x - el_x)**2 + (y - el_y)**2 + (z - el_z)**2)
            if r < 1: r = 1 # Avoid singularity
            
            # Assign rx in MOhm
            seg.xtra.rx = (rho / (4 * np.pi * r)) * 1e-6

def plot_simulation_results(t, v, m, h_vec, cai, params):
    """
    Plots Raw Voltage, Firing Rate, and Gating Variables.
    'params' should be a dictionary containing stim_delay, stim_duration, 
    stim_amplitude, stim_frequency, and the iClamp objects.
    """
    
    v_clean = v_clean = digital_artifact_removal(v, 1/h.dt)
    mem_pot = smoothened_membrane_potential(v, params['freq'], h.dt)

    peaks, _ = find_peaks(v_clean, height=-10, distance=1.0/h.dt)
    peaks_time = t[peaks]

    window_size = 50.0  # ms
    step_size = 1.0     # ms 
    t_rate = np.arange(0, t[-1] + step_size, step_size)
    rate = np.zeros_like(t_rate)

    for i, t_point in enumerate(t_rate):
        t_start, t_end = t_point - window_size/2, t_point + window_size/2
        spike_count = np.sum((peaks_time >= t_start) & (peaks_time < t_end))
        rate[i] = (spike_count / window_size) * 1000

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3])

    fig.add_trace(go.Scatter(x=t, y=v, name="Raw Vm (mV)", line=dict(color='gray', width=1), opacity=0.5), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=v_clean, name="Artefact-Removed mV", 
                             line=dict(color='brown', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=mem_pot, name="Smoothened Membrane Potential (mV)", 
                             line=dict(color='pink', width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_rate, y=rate, name="Firing Rate (Hz)", 
                             line=dict(color='firebrick', width=2), fill='tozeroy'), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=m, name="m (Na act)", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=h_vec, name="h (Na inact)", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=cai, name="[Ca2+]i", line=dict(width=1)), row=3, col=1)

    for r in [1, 2, 3]:
        fig.add_vline(x=params['delay'], line=dict(color='grey', dash='dash'), row=r, col=1)
        fig.add_vline(x=params['delay'] + params['dur'], line=dict(color='grey', dash='dash'), row=r, col=1)
        if params.get('exc_amp', 0) != 0:
            fig.add_vline(x=params['exc_delay'], line=dict(color='green', dash='dot'), row=r, col=1)
            fig.add_vline(x=params['exc_delay'] + params['exc_dur'], line=dict(color='green', dash='dot'), row=r, col=1)
            fig.add_vline(x=params['exc_post_delay'] + params['exc_post_dur'], line=dict(color='green', dash='dot'), row=r, col=1)
        if params.get('inh_amp', 0) != 0:
            fig.add_vline(x=params['inh_delay'], line=dict(color='red', dash='dot'), row=r, col=1)
            fig.add_vline(x=params['inh_delay'] + params['inh_dur'], line=dict(color='red', dash='dot'), row=r, col=1)
            fig.add_vline(x=params['inh_post_delay'] + params['inh_post_dur'], line=dict(color='red', dash='dot'), row=r, col=1)

    fig.update_layout(height=850, title_text=f"HFS Simulation: {params['amp']}uA @ {params['freq']}Hz")
    fig.show()
    return fig, v_clean, mem_pot

def plot_simulation_results_segments(segments_data, t, params, show_fig=True, save=False, return_fig=False):
    """
    Plots Raw Voltage, Firing Rate, and Gating Variables.
    'params' should be a dictionary containing stim_delay, stim_duration, 
    stim_amplitude, stim_frequency, and the iClamp objects.
    """
    segments = ['soma', 'initseg', 'axon']
    segment_titles = ['Soma', 'AIS', 'Axon']
    
    # Added column_titles to label the top of each column automatically
    fig = make_subplots(rows=3, cols=3, shared_xaxes=True, 
                        vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3],
                        column_titles=segment_titles)
    
    for col in [1, 2, 3]:
        segment = segments[col-1]
        data = segments_data[segment]
        v = data['v']
        m_vec = data['m']
        n_vec = data['n']
        h_vec = data['h']
        cai = data['cai']
        
        v_clean = digital_artifact_removal(v, 1/h.dt)
        mem_pot = smoothened_membrane_potential(v, params['freq'], h.dt)

        peaks, _ = find_peaks(v_clean, height=-10, distance=1.0/h.dt)
        peaks_time = t[peaks]

        window_size = 50.0  # ms
        step_size = 1.0     # ms 
        t_rate = np.arange(0, t[-1] + step_size, step_size)
        rate = np.zeros_like(t_rate)

        for i, t_point in enumerate(t_rate):
            t_start, t_end = t_point - window_size/2, t_point + window_size/2
            spike_count = np.sum((peaks_time >= t_start) & (peaks_time < t_end))
            rate[i] = (spike_count / window_size) * 1000
            
        # Added legendgroup and legendgrouptitle_text to separate the legend entries
        fig.add_trace(go.Scatter(x=t, y=v, name="Raw Vm (mV)", 
                                 line=dict(color='gray', width=1), opacity=0.5,
                                 legendgroup=segment, legendgrouptitle_text=segment_titles[col-1],legendgrouptitle_font=dict(size=20, color='black')), 
                      row=1, col=col)

        fig.add_trace(go.Scatter(x=t, y=v_clean, name="Artefact-Removed mV", 
                                line=dict(color='brown', width=1),
                                legendgroup=segment), row=1, col=col)
                                
        fig.add_trace(go.Scatter(x=t, y=mem_pot, name="Smoothened Membrane Potential (mV)", 
                                line=dict(color='pink', width=1),
                                legendgroup=segment), row=1, col=col)

        fig.add_trace(go.Scatter(x=t_rate, y=rate, name="Firing Rate (Hz)", 
                                line=dict(color='firebrick', width=2), fill='tozeroy',
                                legendgroup=segment), row=2, col=col)

        fig.add_trace(go.Scatter(x=t, y=m_vec, name="m (Na act)", 
                                 line=dict(width=1, color='brown'),
                                 legendgroup=segment), row=3, col=col)
                                 
        fig.add_trace(go.Scatter(x=t, y=h_vec, name="h (Na inact)", 
                                 line=dict(width=1, color='green'),
                                 legendgroup=segment), row=3, col=col)
                                 
        fig.add_trace(go.Scatter(x=t, y=n_vec, name="n (K act)", 
                                 line=dict(width=1, color='blue'),
                                 legendgroup=segment), row=3, col=col)

        for r in [1, 2, 3]:
            fig.add_vline(x=params['delay'], line=dict(color='grey', dash='dash'), row=r, col=col)
            fig.add_vline(x=params['delay'] + params['dur'], line=dict(color='grey', dash='dash'), row=r, col=col)
            if params.get('exc_amp', 0) != 0:
                fig.add_vline(x=params['exc_delay'], line=dict(color='green', dash='dot'), row=r, col=col)
                fig.add_vline(x=params['exc_delay'] + params['exc_dur'], line=dict(color='green', dash='dot'), row=r, col=col)
                fig.add_vline(x=params['exc_post_delay'] + params['exc_post_dur'], line=dict(color='green', dash='dot'), row=r, col=col)
            if params.get('inh_amp', 0) != 0:
                fig.add_vline(x=params['inh_delay'], line=dict(color='red', dash='dot'), row=r, col=col)
                fig.add_vline(x=params['inh_delay'] + params['inh_dur'], line=dict(color='red', dash='dot'), row=r, col=col)
                fig.add_vline(x=params['inh_post_delay'] + params['inh_post_dur'], line=dict(color='red', dash='dot'), row=r, col=col)

    fig.update_layout(height=750, title_text=f"HFS Simulation: {params['amp']}uA @ {params['freq']}Hz")
    fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Firing Rate (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Gating Variables", row=3, col=1)

    if show_fig:
        fig.show()
    if save:
        fig.write_html(f"RGC_Segments_Over Time{int(params['amp'])}uA_{int(params['freq'])}Hz.html", include_plotlyjs='cdn')
    if return_fig:
        return fig
"""
def plot_final_rgc_with_electrode(el_x, el_y, el_z, elev=20, azim=45):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True, linestyle=':', alpha=0.4)

    # Updated color scheme
    colors = {
        'soma': '#2c3e50',
        'ais': '#ff69b4',   # Hot Pink for AIS
        'axon': '#c0392b',  # Red
        'dend': '#2980b9',  # Blue
        'elec': '#f1c40f'   # Gold Star
    }

    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2: continue
        
        x = [h.x3d(i, sec=sec) for i in range(n3d)]
        y = [h.y3d(i, sec=sec) for i in range(n3d)]
        z = [h.z3d(i, sec=sec) for i in range(n3d)]
        
        name = sec.name().lower()
        
        if 'soma' in name:
            col, lw = colors['soma'], 2.5
            ax.scatter(x[0], y[0], z[0], color=col, s=100, edgecolors='white', zorder=5)
        elif any(k in name for k in ['init', 'narrow']):
            col, lw = colors['ais'], 2.5 # Slightly thicker to make the pink pop
        elif 'axon' in name:
            col, lw = colors['axon'], 1.2
        else:
            col, lw = colors['dend'], 0.8
            
        ax.plot(x, y, z, color=col, linewidth=lw, alpha=0.9)

    # Plot the electrode
    ax.scatter(el_x, el_y, el_z, color=colors['elec'], s=250, marker='*', 
               edgecolors='black', linewidths=1.5, zorder=20)
    
    # Drop line
    ax.plot([el_x, el_x], [el_y, el_y], [min(0, el_z-50), el_z], 
            color='black', linestyle='--', linewidth=1, alpha=0.5)

    # --- Create the Legend ---
    handles = [
        mlines.Line2D([], [], color=colors['soma'], marker='o', linestyle='None', markersize=8, label='Soma'),
        mlines.Line2D([], [], color=colors['ais'], linewidth=2.5, label='AIS (Pink)'),
        mlines.Line2D([], [], color=colors['axon'], linewidth=1.2, label='Distal Axon'),
        mlines.Line2D([], [], color=colors['dend'], linewidth=0.8, label='Dendrites'),
        mlines.Line2D([], [], color=colors['elec'], marker='*', linestyle='None', 
                      markersize=15, markeredgecolor='black', label='Electrode')
    ]

    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, frameon=True)

    # Formatting
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (μm)')
    limit = 150
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-75, 75)
    ax.set_box_aspect((1, 1, 0.5)) 

    ax.set_title(f"RGC Morphology: AIS Highlighted Pink\nElectrode 25 um above the soma center", fontsize=12)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()

    fig.write_html("RGC_Morphology_with_Electrode.html", include_plotlyjs='cdn')

"""

def plot_final_rgc_with_electrode(el_x, el_y, el_z):
    fig = go.Figure()

    colors = {
        'soma': "#777809",
        'ais': "#023708",   # Hot Pink
        'axon': "#6f3730",  # Red
        'dend': '#2980b9',  # Blue
        'elec': '#f1c40f'   # Gold
    }

    # Track which labels we've already added to the legend
    added_to_legend = set()

    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2: continue
        
        x = [h.x3d(i, sec=sec) for i in range(n3d)]
        y = [h.y3d(i, sec=sec) for i in range(n3d)]
        z = [h.z3d(i, sec=sec) for i in range(n3d)]
        
        name = sec.name().lower()
        
        # Determine classification
        if 'soma' in name:
            label, col, lw = 'Soma', colors['soma'], 6
        elif any(k in name for k in ['init', 'narrow', 'initseg']):
            label, col, lw = 'AIS', colors['ais'], 5
        elif 'axon' in name:
            label, col, lw = 'Axon', colors['axon'], 3
        else:
            label, col, lw = 'Dendrites', colors['dend'], 1.5

        # Only show the legend entry if it's the first time we've seen this label
        show_leg = False
        if label not in added_to_legend:
            show_leg = True
            added_to_legend.add(label)

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=col, width=lw),
            name=label,
            legendgroup=label, # Grouping ensures clicking one toggles all
            showlegend=show_leg,
            hoverinfo='name'
        ))

    # Plot Electrode
    fig.add_trace(go.Scatter3d(
        x=[el_x], y=[el_y], z=[el_z],
        mode='markers',
        marker=dict(symbol='diamond', size=10, color=colors['elec'], 
                    line=dict(width=2, color='black')),
        name='Electrode'
    ))

    # Electrode Drop Line
    fig.add_trace(go.Scatter3d(
        x=[el_x, el_x], y=[el_y, el_y], z=[min(0, el_z-50), el_z],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))

    fig.update_layout(
        title="RGC Morphology with Electrode",
        scene=dict(
            xaxis=dict(title='X (μm)', range=[-250, 250]),
            yaxis=dict(title='Y (μm)', range=[-250, 250]),
            zaxis=dict(title='Z (μm)', range=[-75, 75]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        legend=dict(itemsizing='constant'), # Makes legend lines easier to see
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    fig.write_html("RGC_Morphology_with_Electrode.html", include_plotlyjs='cdn')

def plot_iv_curve(amps, v_steady):
    # Quick I-V Curve Diagnostic
    amps = np.linspace(-0.2, 0.2, 21) # -200 to +200 pA
    v_steady = []

    for a in amps:
        ic = h.IClamp(h.soma(0.5))
        ic.delay = 0
        ic.dur = 100
        ic.amp = a
        h.finitialize(-62.5)
        h.continuerun(100)
        v_steady.append(h.soma(0.5).v)

    plt.figure()
    plt.plot(amps*1000, v_steady, 'o-')
    plt.xlabel("Injected Current (pA)")
    plt.ylabel("Steady State Voltage (mV)")
    plt.title("RGC I-V Curve: Look for the 'Knee'")
    #plt.show()

def process_rgc_signal(v_data, t_data, hfs_freq, dt_ms):

    """
    Cleans RGC signals for analysis.
    Returns:
    - v_filtered: Best for spike counting (50Hz cutoff)
    - v_smooth: Best for mean membrane potential (moving average)
    """
    # 1. Setup Sampling Math
    fs = 1000.0 / dt_ms         # Sampling rate in Hz
    nyquist = fs / 2.0
    fc = 50.0                   # 50Hz cutoff to remove spikes/HFS
    
    # Safety check for Nyquist
    if fc >= nyquist:
        fc = nyquist * 0.8

    # 2. Butterworth Low-Pass Filter
    b, a = butter(5, fc / nyquist, btype='low')
    v_filtered = filtfilt(b, a, v_data)
    
    # 3. Moving Average (The MATLAB movmean equivalent)
    # Window = (100 * HFS period) / dt
    dt_sec = dt_ms / 1000.0
    av_window = int((1.0 / hfs_freq) / dt_sec * 100)
    
    if av_window < 1:
        av_window = 1
    
    # Use convolution for a fast, vectorized moving average
    window = np.ones(av_window) / av_window
    v_smooth = np.convolve(v_data, window, mode='same')
        
    return v_filtered, v_smooth

def plot_comprehensive_dashboard(t, v, m_vec, h_vec, n_vec, params, v_recorders, t_stim, segment='', show_fig=True, save=False, return_fig=False):
    # 1. GPU/MEMORY CLEANUP: Nuke old WebGL contexts before starting
    display(Javascript('document.querySelectorAll(".js-plotly-plot").forEach(p => p.delete && p.delete());'))
    gc.collect()

    # --- Data Processing ---
    v_clean = digital_artifact_removal(v, 1/h.dt)
    mem_pot = smoothened_membrane_potential(v, params['freq'], h.dt)

    peaks, _ = find_peaks(v_clean, height=-10, distance=1.0/h.dt)
    peaks_time = t[peaks]

    window_size = 50.0
    step_size = 1.0 
    t_rate = np.arange(0, t[-1] + step_size, step_size)
    rate = np.zeros_like(t_rate)

    for i, t_point in enumerate(t_rate):
        t_start, t_end = t_point - window_size/2, t_point + window_size/2
        spike_count = np.sum((peaks_time >= t_start) & (peaks_time < t_end))
        rate[i] = (spike_count / window_size) * 1000

    plot_data = []
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2: continue
        
        xs = [h.x3d(i, sec=sec) for i in range(n3d)]
        ys = [h.y3d(i, sec=sec) for i in range(n3d)]
        zs = [h.z3d(i, sec=sec) for i in range(n3d)]
        
        name_key = f"{sec.name()}(0.5)"
        
        if name_key in v_recorders:
            v_smooth = np.array(smoothened_membrane_potential(v_recorders[name_key], params['freq'], h.dt))
            spike_rate = get_spike_rate(np.array(v_recorders[name_key]), t_stim)
            plot_data.append({
                'x': xs, 'y': ys, 'z': zs,
                'spike_rate': spike_rate,
                'v_smooth': v_smooth,
                'name': sec.name()
            })

    # --- Group Branches for Trace Concatenation ---
    groups = {
        'soma': {'branches': [], 'width': 30},
        'initseg': {'branches': [], 'width': 4},
        'dend': {'branches': [], 'width': 2}
    }
    
    for branch in plot_data:
        if "soma" in branch['name']:
            groups['soma']['branches'].append(branch)
        elif "initseg" in branch['name']:
            groups['initseg']['branches'].append(branch)
        else:
            groups['dend']['branches'].append(branch)

    def get_concat_coords(branch_list):
        x, y, z = [], [], []
        for b in branch_list:
            x.extend(b['x'] + [None])
            y.extend(b['y'] + [None])
            z.extend(b['z'] + [None])
        return x, y, z

    def get_concat_colors(branch_list, time_idx, key):
        c = []
        for b in branch_list:
            # Repeat the color for all points + 1 for the 'None' separator
            c.extend([b[key][time_idx]] * (len(b['x']) + 1))
        return c

    # --- Setup the Dashboard ---
    fig = make_subplots(
        rows=3, cols=3,
        shared_xaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        column_widths=[0.35, 0.35, 0.3],
        specs=[
            [{'type': 'scene', 'rowspan': 3}, {'type': 'scene', 'rowspan': 3}, {'type': 'xy'}],
            [None, None, {'type': 'xy'}],
            [None, None, {'type': 'xy'}]
        ]
    )
    
    # --- 2D Raw / Smooth Vm (Now forced to lists) ---
    t_list = t.tolist()
    fig.add_trace(go.Scatter(x=t_list, y=v.tolist(), name="Raw Vm", 
                             line=dict(color='gray', width=1), opacity=0.3), row=1, col=3)
    fig.add_trace(go.Scatter(x=t_list, y=v_clean.tolist(), name="Cleaned", 
                             line=dict(color='brown', width=1)), row=1, col=3)
    fig.add_trace(go.Scatter(x=t_list, y=mem_pot.tolist(), name="Smooth Vm", 
                             line=dict(color='pink', width=1.5)), row=1, col=3)
    
    # Firing Rate
    fig.add_trace(go.Scatter(x=t_rate.tolist(), y=rate.tolist(), name="Rate", 
                             line=dict(color='firebrick', width=2), fill='tozeroy'), row=2, col=3)
    
    # Gating (Forced to lists)
    fig.add_trace(go.Scatter(x=t_list, y=m_vec.tolist(), name="m", line=dict(width=1)), row=3, col=3)
    fig.add_trace(go.Scatter(x=t_list, y=h_vec.tolist(), name="h", line=dict(width=1)), row=3, col=3)
    fig.add_trace(go.Scatter(x=t_list, y=n_vec.tolist(), name="n", line=dict(width=1)), row=3, col=3)

    # Static Protocol Lines (HFS and CC ranges)
    y_ranges = {1: [-80, 30], 2: [0, 250], 3: [0, 1]}
    protocol_times = [
        (params.get('delay'), params.get('dur'), 'grey'), # HFS
        (params.get('exc_delay'), params.get('exc_dur'), 'orange'),                            # POS CC (300-500)
        (params.get('inh_delay'), params.get('inh_dur'), 'blue')                               # NEG CC (600-800)
    ]
    for start, duration, color in protocol_times:
        for r in [1, 2, 3]:
            for x_pos in [start, start + duration]:
                fig.add_trace(go.Scatter(x=[x_pos, x_pos], y=y_ranges[r], 
                              mode="lines", line=dict(color=color, dash='dot', width=1), 
                              showlegend=False, hoverinfo='skip'), row=r, col=3)
    
    # --- The Animated Section Begins Here ---
    start_anim_idx = len(fig.data)

    # 3D Branches - Reduced to only 3 traces per subplot!
    for group_name, group_data in groups.items():
        if not group_data['branches']: continue
        
        cx, cy, cz = get_concat_coords(group_data['branches'])
        cv = get_concat_colors(group_data['branches'], 0, 'v_smooth')
        cr = get_concat_colors(group_data['branches'], 0, 'spike_rate')
        w = group_data['width']

        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='lines',
            line=dict(
                color=cv, colorscale='Magma', cmin=-70, cmax=-30, width=w,
                colorbar=dict(title="mV", x=0.3, thickness=15, len=0.7) if group_name == 'dend' else None,
            ),
            showlegend=False), row=1, col=1)
            
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='lines',
            line=dict(
                color=cr, colorscale='Magma', cmin=0, cmax=300, width=w,
                colorbar=dict(title="Hz", x=0.65, thickness=15, len=0.7) if group_name == 'dend' else None,
            ),
            showlegend=False), row=1, col=2)

    # The Moving "Time" Vertical Lines
    fig.add_trace(go.Scatter(x=[0, 0], y=y_ranges[1], mode="lines", line=dict(color="red", width=3), name="TimeMarker"), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0, 0], y=y_ranges[2], mode="lines", line=dict(color="red", width=3), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=[0, 0], y=y_ranges[3], mode="lines", line=dict(color="red", width=3), showlegend=False), row=3, col=3)

    # Floating Text Trace (Instead of Layout Annotations)
    fig.add_trace(go.Scatter(
        x=[0], y=[y_ranges[1][1]], 
        mode="text", text=["Time: 0.0 ms<br>HFS: OFF<br>POS CC: inactive<br>NEG CC: inactive"], 
        textposition="bottom right", 
        textfont=dict(size=14, color="black"), 
        showlegend=False
    ), row=1, col=3)

    anim_trace_indices = list(range(start_anim_idx, len(fig.data)))

    step = int(5.0 / h.dt)
    time_indices = np.arange(0, len(t), step)
    frames = []

    for k in time_indices:
        t_curr = k * h.dt
        frame_data = []
        
        # Color updates
        for group_name, group_data in groups.items():
            if not group_data['branches']: continue
            
            cv = get_concat_colors(group_data['branches'], k, 'v_smooth')
            cr = get_concat_colors(group_data['branches'], k, 'spike_rate')
            
            frame_data.append(go.Scatter3d(line=dict(color=cv)))
            frame_data.append(go.Scatter3d(line=dict(color=cr)))
            
        # Vertical Line movement
        frame_data.extend([go.Scatter(x=[t_curr, t_curr])] * 3)

        # Floating Text Update (With inline HTML for colors)
        hfs_s = "<span style='color:red'><b>HFS: ON</b></span>" if 100 <= t_curr <= 900 else "HFS: OFF"
        pos_s = "<span style='color:orange'><b>POS CC: ACTIVE</b></span>" if 300 <= t_curr <= 500 else "<span style='color:gray'>POS CC: inactive</span>"
        neg_s = "<span style='color:blue'><b>NEG CC: ACTIVE</b></span>" if 600 <= t_curr <= 800 else "<span style='color:gray'>NEG CC: inactive</span>"
        
        status_text = f"Time: {t_curr:.1f} ms<br>{hfs_s}<br>{pos_s}<br>{neg_s}"
        
        # Move the text along with the vertical line in row 1
        frame_data.append(go.Scatter(x=[0], text=[status_text]))

        frames.append(go.Frame(data=frame_data, name=str(k), traces=anim_trace_indices))

    fig.frames = frames

    slider_steps = []
    for k in time_indices:
        t_current = k * h.dt
        is_tick = round(t_current) % 100 == 0 # 100 ms ticks

        step_dict = {
            # Note redraw: False is essential here!
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": f"{t_current:.0f}" if is_tick else "",
            "method": "animate"
        }
        slider_steps.append(step_dict)

    camera_top = dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0))
    scene_config = dict(xaxis=dict(range=[-150, 150], title="X"), yaxis=dict(range=[-150, 150], title="Y"), 
                        zaxis=dict(range=[-50, 50], title="Z"), aspectmode='cube', camera=camera_top)

    fig.update_layout(
        title=f"HFS Dashboard: {params.get('amp', 0)}uA @ {params.get('freq', 0)}Hz",
        height=750, width=1600,
        scene=scene_config, scene2=scene_config,
        updatemenus=[{
            "buttons": [
                {"args": [None, {
                    "frame": {"duration": 5, "redraw": True}, # Must be True for 3D
                    "fromcurrent": True,
                    "active": -1,
                    "transition": {"duration": 0}
                }], "label": "Play", "method": "animate"},

                {"args": [[None], {
                    "frame": {"duration": 5, "redraw": True}, 
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }], "label": "Pause", "method": "animate"},
            ],
            "type": "buttons", "direction": "right", "x": 0.05, "y": -0.05
        }],
        sliders=[{
            "active": 0, "yanchor": "top", "xanchor": "left", "x": 0.15, "y": -0.05, "len": 0.8,
            "currentvalue": {"prefix": "Time: ", "visible": True, "xanchor": "right", "offset": 15},
            "pad": {"b": 40, "t": 20},
            "tickcolor": "#444", "ticklen": 8, "tickwidth": 1, "minorticklen": 0,
            "steps": slider_steps
        }],
        margin=dict(l=20, r=20, b=120, t=80)
    )
    fig.layout.sliders[0].active = 0
    fig.layout.updatemenus[0].buttons[1]["args"][1]["fromcurrent"] = False
    
    if show_fig:
        fig.show()

    config = {'displayModeBar': True, 'responsive': True}
    if save:
        fig.write_html(
            f"HFS_Dashboard {segment} {params.get('amp', 0)}uA @ {params.get('freq', 0)}Hz.html", 
            include_plotlyjs='cdn',
            full_html=True,
            auto_play=False,
            config=config
            )
    if return_fig:
        return fig

def browse_simulation_results(fig_dict, save=False, filename="CollectionOfData"):
    # Sort keys numerically if possible, otherwise use strings
    try:
        sorted_keys = sorted(fig_dict.keys(), key=lambda x: float(x))
    except ValueError:
        sorted_keys = sorted(fig_dict.keys())

    # Calculate the middle index for the default value
    middle_index = len(sorted_keys) // 2
    default_key = sorted_keys[middle_index]

    # Create the dropdown widget
    dropdown = widgets.Dropdown(
        options=sorted_keys,
        value=default_key,
        description='Amplitude:',
        style={'description_width': 'initial'}
    )

    # Create an output container for the figure
    output = widgets.Output()

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            with output:
                display(Javascript('document.querySelectorAll(".js-plotly-plot").forEach(p => p.delete && p.delete());'))
                gc.collect()

                clear_output(wait=True)
                # Show the selected figure
                fig_dict[change['new']].show()

    # Observe changes in the dropdown
    dropdown.observe(on_change)

    # Display the initial (middle) figure
    with output:
        fig_dict[default_key].show()

    # Render the UI
    display(widgets.VBox([dropdown, output]))
    if save:
        save_fig_dict_html(fig_dict, output_html=f"{filename}.html")


import json
import plotly.utils

def save_fig_dict_html(figs, output_html="combined.html"):
    keys = list(figs.keys())
    middle_i = len(keys) // 2
    default_key = keys[middle_i]

    # 1. Convert all figures to JSON strings (this is memory efficient)
    fig_json_data = {}
    for key in keys:
        # We use the Plotly JSON encoder to handle numpy arrays/dates
        fig_json_data[key] = json.dumps(figs[key], cls=plotly.utils.PlotlyJSONEncoder)

    options = [f'<option value="{k}" {"selected" if k == default_key else ""}>{k}</option>' for k in keys]

    html = f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
            .controls {{ margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            #plot-stage {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="controls">
            <label><b>Select Amplitude: </b></label>
            <select id="fig-selector" onchange="loadFigure(this.value)" style="font-size:16px; padding:5px;">
                {''.join(options)}
            </select>
        </div>

        <div id="plot-stage" style="width:100%; height:850px;"></div>

        <script>
            // Store all figure data in a JS object
            const allFigures = {json.dumps(fig_json_data)};

            function loadFigure(key) {{
                const stage = document.getElementById('plot-stage');
                
                // 1. Purge the old WebGL context completely to free up memory
                Plotly.purge(stage);

                // 2. Parse the JSON for the new figure
                const figData = JSON.parse(allFigures[key]);

                // 3. Render the figure fresh
                Plotly.newPlot(stage, figData.data, figData.layout, {{responsive: true, displayModeBar: true}}).then(function(gd) {{
                    // Add the frames back in for animation
                    Plotly.addFrames(gd, figData.frames);
                    
                    // Stop auto-play immediately
                    Plotly.animate(gd, null, {{
                        mode: "immediate",
                        frame: {{duration: 0, redraw: true}},
                        transition: {{duration: 0}}
                    }});
                }});
            }}

            // Load the initial figure on startup
            window.onload = function() {{
                loadFigure("{default_key}");
            }};
        </script>
    </body>
    </html>
    """

    with open(output_html, "w") as f:
        f.write(html)