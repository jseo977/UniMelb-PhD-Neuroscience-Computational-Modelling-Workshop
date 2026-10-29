"""
Helper functions for the UniMelb PhD Neuroscience computational modelling workshop
(Hodgkin–Huxley modelling with the Sheasby & Fohlmeister 1999 retinal ganglion cell).

Contents
  1. Single-compartment model (Activities 1–3)   build_soma_sc, run_soma_sc, find_spikes, spike_half_width
  2. Running simulations                          run_from_rest, Recorder
  3. Stimuli                                      iclamp, pulse_wave, extracellular, stop_extracellular,
                                                  reset_stimuli, update_rx_3d
  4. Spikes and thresholds                        spike_times, find_threshold, scan_threshold
  5. Optional-extra exercises                     fires_with_pulse, second_pulse_fires,
                                                  fires_extracellular, segments, point_above
  6. Morphology                                   seg_xyz, dendrite_path_sites, plot_final_rgc_with_electrode
  7. Interactive dashboard                        plot_interactive_dashboard
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from neuron import h

h.load_file('stdrun.hoc')          # provides h.continuerun

PEAK_HEIGHT = -20.0                # mV : default spike-detection threshold


# ══════════════════════════════════════════════════════════════════════════════
# 1. SINGLE-COMPARTMENT MODEL (Activities 1–3)
# ══════════════════════════════════════════════════════════════════════════════
def build_soma_sc():
    """A single cylindrical compartment with the size and ion channels of the soma of ctt3219b."""
    soma_sc = h.Section(name='soma_sc')
    soma_sc.L    = 27.69     # µm (gives the same membrane area as the soma, ~1216 µm²)
    soma_sc.diam = 13.98     # µm
    soma_sc.nseg = 1
    soma_sc.Ra   = 110       # ohm·cm
    soma_sc.cm   = 1         # µF/cm²

    # Passive leak
    # The full model uses g_pas = 8e-6 S/cm², but on an isolated soma that gives an input
    # resistance of ~10 GΩ (−10 pA -> ~−100 mV). In the full cell, the dendrites and axon
    # carry most of that current away. A larger leak stands in for them, giving an
    # input resistance of ~200 MΩ, which is physiological for an RGC.
    soma_sc.insert('pas')
    soma_sc.g_pas = 0.0004    # S/cm²
    soma_sc.e_pas = -62.5     # mV

    # Active channels (spike.mod) and calcium pump (capump.mod)
    soma_sc.insert('spike')
    soma_sc.insert('cad')
    soma_sc.gnabar_spike = 0.080     # S/cm² : Na
    soma_sc.gkbar_spike  = 0.018     # S/cm² : delayed-rectifier K
    soma_sc.gabar_spike  = 0.054     # S/cm² : A-type K
    soma_sc.gcabar_spike = 0.0015    # S/cm² : Ca
    soma_sc.gkcbar_spike = 0.000065  # S/cm² : Ca-activated K
    soma_sc.ena = 35.0               # mV
    soma_sc.ek  = -75.0              # mV
    soma_sc(0.5).depth_cad = soma_sc.diam / 2   # µm : Ca shell depth
    return soma_sc


def run_soma_sc(soma_sc, amp_pa, dur, delay=10, tstop=50, dt=0.05, celsius=33):
    """Inject a current pulse (amp_pa in pA, dur in ms) into the single compartment and return
    the recordings as arrays: t, v, m, h, n (gating variables) and i (injected current, pA)."""
    clamp = h.IClamp(soma_sc(0.5))
    clamp.delay, clamp.dur, clamp.amp = delay, dur, amp_pa / 1000       # IClamp works in nA
    rec = {'t': h.Vector().record(h._ref_t),
           'v': h.Vector().record(soma_sc(0.5)._ref_v),
           'm': h.Vector().record(soma_sc(0.5)._ref_m_spike),
           'h': h.Vector().record(soma_sc(0.5)._ref_h_spike),
           'n': h.Vector().record(soma_sc(0.5)._ref_n_spike),
           'i': h.Vector().record(clamp._ref_i)}
    h.dt, h.tstop, h.celsius = dt, tstop, celsius
    h.finitialize(-62.5)
    soma_sc.cai = 0.0001                       # reset calcium to its resting concentration
    h.fcurrent()
    h.frecord_init()
    h.continuerun(tstop)
    out = {k: np.array(x) for k, x in rec.items()}
    out['i'] *= 1000                           # nA -> pA
    return out


def find_spikes(v, height=PEAK_HEIGHT, prominence=10):
    """Indices of action-potential peaks (above height, ignoring small ripples)."""
    peaks, _ = find_peaks(v, height=height, prominence=prominence)
    return peaks


def spike_half_width(t, v, i_pk, dvdt_thres=10.0):
    """Width (ms) at half-amplitude, measured from the spike's threshold (dV/dt > 10 V/s) to its peak."""
    dvdt = np.gradient(v, t)
    i_th = i_pk
    while i_th > 0 and dvdt[i_th - 1] > dvdt_thres:
        i_th -= 1
    half = v[i_th] + (v[i_pk] - v[i_th]) / 2
    lo, hi = i_pk, i_pk
    while lo > 0 and v[lo - 1] >= half:
        lo -= 1
    while hi < len(v) - 1 and v[hi + 1] >= half:
        hi += 1
    return t[hi] - t[lo]


# ══════════════════════════════════════════════════════════════════════════════
# 2. RUNNING SIMULATIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_from_rest(tstop, dt=0.025, v_init=-62.5):
    """Initialise every section at rest (calcium reset too) and run to tstop (ms)."""
    h.dt, h.tstop = dt, tstop
    h.finitialize(v_init)
    for sec in h.allsec():
        if h.ismembrane('ca_ion', sec=sec):
            sec.cai = 0.0001                   # resting calcium, so repeated runs agree
    h.fcurrent()
    h.frecord_init()
    h.continuerun(tstop)


class Recorder:
    """Records time and Vm at named segments.  rec = Recorder({'soma': h.soma(0.5), ...})
    rec.run(tstop, dt) runs from rest and returns (t, {name: Vm}) as arrays."""
    def __init__(self, sites):
        self.t = h.Vector().record(h._ref_t)
        self.v = {}
        for name, seg in sites.items():
            self.add(name, seg)

    def add(self, name, seg):
        self.v[name] = h.Vector().record(seg._ref_v)

    def run(self, tstop, dt=0.025):
        run_from_rest(tstop, dt)
        return np.array(self.t), {k: np.array(x) for k, x in self.v.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 3. STIMULI
# ══════════════════════════════════════════════════════════════════════════════
def iclamp(seg, amp, delay, dur):
    """A current clamp at seg: amp in nA, delay and dur in ms. Keep the returned object alive."""
    c = h.IClamp(seg)
    c.amp, c.delay, c.dur = amp, delay, dur
    return c


def pulse_wave(t_on, phase_dur, amp, shape, tstop, dt=0.025):
    """Extracellular pulse waveform (µA): 'cathodic' (negative), 'anodal' (positive) or
    'biphasic' (cathodic, then anodal of the same size and length)."""
    t_w = np.arange(0, tstop + dt, dt)
    i_w = np.zeros_like(t_w)
    first = (t_w >= t_on) & (t_w < t_on + phase_dur)
    second = (t_w >= t_on + phase_dur) & (t_w < t_on + 2 * phase_dur)
    if shape == 'cathodic':
        i_w[first] = -amp
    elif shape == 'anodal':
        i_w[first] = amp
    elif shape == 'biphasic':
        i_w[first], i_w[second] = -amp, amp
    else:
        raise ValueError("shape must be 'cathodic', 'anodal' or 'biphasic'")
    return t_w, i_w


_XSTIM = {}                                    # keeps the extracellular stimulus vectors alive


def extracellular(t_wave, i_wave):
    """Play a current waveform (µA; negative = cathodic) through the extracellular electrode."""
    stop_extracellular()
    _XSTIM['t'], _XSTIM['i'] = h.Vector(t_wave), h.Vector(i_wave)
    _XSTIM['i'].play(h._ref_is_xtra, _XSTIM['t'])


def stop_extracellular():
    """Stop any extracellular stimulus started with extracellular()."""
    if 'i' in _XSTIM:
        _XSTIM['i'].play_remove()
    _XSTIM.clear()
    h.is_xtra = 0


def reset_stimuli(el_home):
    """Stop extracellular stimulation, set every current clamp in the model to 0 nA, and put
    the electrode back at el_home = (x, y, z) µm. (Re-run Activity 4's stimulus cell to
    restore its clamp.)"""
    stop_extracellular()
    for clamp in h.List('IClamp'):
        clamp.amp = 0
    update_rx_3d(*el_home)


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


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPIKES AND THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
def spike_times(t, v, t_from=0.0, t_to=np.inf, height=PEAK_HEIGHT):
    """Times (ms) of spike peaks above height, between t_from and t_to."""
    p, _ = find_peaks(v, height=height)
    tp = t[p]
    return tp[(tp >= t_from) & (tp <= t_to)]


def find_threshold(fires, lo, hi, rel_tol=0.02):
    """Smallest amplitude between lo and hi for which fires(amp) is True, to within rel_tol
    (geometric bisection). Returns nan if hi does not fire, lo if lo already fires."""
    if not fires(hi):
        return np.nan
    if fires(lo):
        return lo
    while hi / lo > 1 + rel_tol:
        mid = np.sqrt(lo * hi)
        if fires(mid):
            hi = mid
        else:
            lo = mid
    return hi


def scan_threshold(fires, amps, rel_tol=0.02):
    """Like find_threshold, but first steps up through amps (ascending) to the first that fires.
    Use when a response can fail again at higher amplitudes (e.g. extracellular block)."""
    prev = None
    for a in amps:
        if fires(a):
            return a if prev is None else find_threshold(fires, prev, a, rel_tol)
        prev = a
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTIONAL-EXTRA EXERCISES
# ══════════════════════════════════════════════════════════════════════════════
def fires_with_pulse(rec, amp, dur, site='soma', t_on=5, after=15, key='soma', dt=0.025):
    """Does one pulse fire a spike at rec's site `key`?  site='soma': current clamp at the soma
    (amp in nA); site='electrode': cathodic pulse from the extracellular electrode (amp in µA)."""
    tstop = t_on + dur + after
    clamp = None
    if site == 'soma':
        clamp = iclamp(h.soma(0.5), amp, t_on, dur)
    elif site == 'electrode':
        extracellular(*pulse_wave(t_on, dur, amp, 'cathodic', tstop, dt))
    else:
        raise ValueError("site must be 'soma' or 'electrode'")
    t, v = rec.run(tstop, dt)
    stop_extracellular()
    del clamp
    return len(spike_times(t, v[key], t_on)) > 0


def second_pulse_fires(rec, amp1, amp2, interval, first_spike, pulse=0.5, t1=5, after=20, dt=0.025):
    """Two current pulses at the soma (nA), interval ms apart: does the second one fire a spike
    of its own (after both the second pulse and the first spike, at first_spike ms)?"""
    c1 = iclamp(h.soma(0.5), amp1, t1, pulse)
    c2 = iclamp(h.soma(0.5), amp2, t1 + interval, pulse)
    t, v = rec.run(t1 + interval + after, dt)
    del c1, c2
    return len(spike_times(t, v['soma'], max(t1 + interval, first_spike + 0.5))) > 0


def fires_extracellular(rec, amp, shape, phase_dur=0.5, t_on=5, after=20, key='axon_far', dt=0.025):
    """Does an extracellular pulse (amp in µA, shape as in pulse_wave) send a spike to rec's site `key`?"""
    tstop = t_on + after
    extracellular(*pulse_wave(t_on, phase_dur, amp, shape, tstop, dt))
    t, v = rec.run(tstop, dt)
    stop_extracellular()
    return len(spike_times(t, v[key], t_on)) > 0


def segments(where='all'):
    """All segments in the cell ('all'), or in the sections whose names start with where (e.g. 'dend')."""
    secs = list(h.allsec()) if where == 'all' else [s for s in h.allsec() if s.name().startswith(where)]
    return [seg for sec in secs for seg in sec]


def point_above(seg, height):
    """(x, y, z) µm of the point height µm above seg, e.g. to place the electrode over it."""
    x, y, z = seg_xyz(seg)
    return (x, y, z + height)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MORPHOLOGY
# ══════════════════════════════════════════════════════════════════════════════
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


def seg_xyz(seg):
    """3D coordinates (µm) of a segment, interpolated along its section's 3D points."""
    sec = seg.sec
    n = int(h.n3d(sec=sec))
    arc = np.array([h.arc3d(i, sec=sec) for i in range(n)])
    pos = seg.x * sec.L
    return tuple(float(np.interp(pos, arc, [f(i, sec=sec) for i in range(n)]))
                 for f in (h.x3d, h.y3d, h.z3d))


def _soma_ellipsoid(sec, n_around=32, n_along=20):
    """Vertices and triangles of a smooth ellipsoid with the soma's length, diameter and
    orientation (long axis = first to last 3D point)."""
    n = int(h.n3d(sec=sec))
    pts = np.array([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)] for i in range(n)])
    p0, p1 = pts[0], pts[-1]
    centre = (p0 + p1) / 2
    axis = p1 - p0
    u = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else np.array([1.0, 0, 0])
    ref = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(u, ref); v /= np.linalg.norm(v)
    w = np.cross(u, v)
    a, b = max(sec.L, sec.diam) / 2, sec.diam / 2
    th = np.linspace(0, np.pi, n_along)                       # along the long axis
    ph = np.linspace(0, 2 * np.pi, n_around, endpoint=False)  # around it
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    xyz = (centre[None, None, :]
           + (a * np.cos(TH))[..., None] * u
           + (b * np.sin(TH) * np.cos(PH))[..., None] * v
           + (b * np.sin(TH) * np.sin(PH))[..., None] * w).reshape(-1, 3)
    I, J, K = [], [], []
    for i in range(n_along - 1):
        for j in range(n_around):
            q0, q1 = i * n_around + j, i * n_around + (j + 1) % n_around
            q2, q3 = q0 + n_around, q1 + n_around
            I += [q0, q1]; J += [q2, q2]; K += [q1, q3]
    return xyz, I, J, K


def dendrite_path_sites(fractions=(0.25, 0.55, 0.9), prefix='dend'):
    """
    Segments along the longest dendritic path (soma -> furthest dendritic tip), at the
    given fractions of that path's length. Returns [(segment, distance from soma in µm)].
    """
    soma = h.soma(0.5)
    dends = [s for s in h.allsec() if s.name().startswith(prefix)]
    tip = max(dends, key=lambda s: h.distance(soma, s(1)))
    path, sec = [], tip
    while sec is not None and sec.name().startswith(prefix):     # walk back to the soma
        path.append(sec)
        parent = sec.parentseg()
        sec = parent.sec if parent is not None else None
    segs = [seg for s in reversed(path) for seg in s]
    dists = np.array([h.distance(soma, seg) for seg in segs])
    out = []
    for f in fractions:
        i = int(np.argmin(np.abs(dists - f * h.distance(soma, tip(1)))))
        out.append((segs[i], float(dists[i])))
    return out


_SITE_COLOURS = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
_GATE_COLOURS = {'m': '#17becf', 'h': '#9467bd', 'n': '#8c564b'}


def _default_site_colours(labels):
    # Fixed colours by name: soma black, AIS orange, axon green, dendrites light -> dark blue
    import matplotlib.pyplot as _plt
    dend = [l for l in labels if 'dend' in l.lower()]
    blues = _plt.cm.Blues(np.linspace(0.4, 0.95, max(len(dend), 1)))
    out, other = {}, iter(_SITE_COLOURS * 3)
    for l in labels:
        low = l.lower()
        if 'soma' in low:
            out[l] = '#000000'
        elif 'ais' in low or 'initseg' in low:
            out[l] = '#e67e22'
        elif 'axon' in low:
            out[l] = '#2ca02c'
        elif l in dend:
            r, g, b, _ = blues[dend.index(l)]
            out[l] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        else:
            out[l] = next(other)
    return out


def plot_interactive_dashboard(t, v_recorders, sites, panels=None, gates=None, protocol=(),
                               frame_ms=5.0, v_range=None, view_um=None,
                               title="Interactive dashboard", panel_height=150,
                               show_fig=True, save_html=None):
    """
    t            : time vector (ms), h.Vector or array
    v_recorders  : {section name: recorded Vm at the middle of that section}, used to colour
                   the 3D cell. Keys may be 'dend1[3]' or 'dend1[3](0.5)'.
    sites        : recording sites shown as traces and as markers on the cell:
                   [(label, segment, recorded Vm), ...] or (label, segment, Vm, colour).
                   Without a colour: soma black, AIS orange, axon green, dendrites
                   light -> dark blue in the order given.
    panels       : how to group the sites into trace panels: [(panel title, [site labels]), ...]
                   Default: one panel per site.
    gates        : optional {'m': ..., 'h': ..., 'n': ...} soma gates, drawn as a last panel
    protocol     : [(label, t_start, t_end, colour), ...] shaded on the traces and shown as
                   ON/off in the status text, e.g. [('Stimulus', 5, 10, 'red')]
    frame_ms     : time between animation frames (smaller = smoother but larger figure)
    v_range      : (min, max) mV of the colour scale (default -80 to 30 mV)
    view_um      : half-width (µm) of the 3D view. Default: fit the soma and dendrites (the
                   much longer axon runs off the edge of the view).
    save_html    : file name to also save the dashboard as a standalone HTML page
    """
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    protocol = list(protocol)

    # --- Recording sites ------------------------------------------------------
    site_info = {}
    auto_colour = _default_site_colours([s[0] for s in sites])
    for s in sites:
        label, seg, vec = s[:3]
        colour = s[3] if len(s) > 3 else auto_colour[label]
        site_info[label] = dict(seg=seg, v=np.asarray(vec, dtype=float), colour=colour)
    if panels is None:
        panels = [(f"{label} Vm", [label]) for label in site_info]
    panels = [(ptitle, list(labels)) for ptitle, labels in panels]
    if gates:
        panels.append(("Soma gates", None))
    n_rows = len(panels)

    # --- Per-section Vm at the animation frames --------------------------------
    frame_idx = np.arange(0, len(t), max(1, int(round(frame_ms / dt))))
    t_frames = t[frame_idx]

    groups = {'initseg': dict(width=6, secs=[]), 'other': dict(width=3, secs=[])}
    somas = []
    for sec in h.allsec():
        name = sec.name()
        vec = v_recorders.get(name, v_recorders.get(f"{name}(0.5)"))
        n3d = int(h.n3d(sec=sec))
        if vec is None or n3d < 2:
            continue
        v_raw = np.asarray(vec, dtype=float)
        v_col = v_raw
        if 'soma' in name:                    # drawn as a shaded ellipsoid, not a line
            xyz, I, J, K = _soma_ellipsoid(sec)
            somas.append(dict(xyz=xyz, I=I, J=J, K=K, v=v_col[frame_idx], name=name))
            continue
        key = 'initseg' if 'initseg' in name else 'other'
        groups[key]['secs'].append(dict(
            x=[h.x3d(i, sec=sec) for i in range(n3d)] + [None],
            y=[h.y3d(i, sec=sec) for i in range(n3d)] + [None],
            z=[h.z3d(i, sec=sec) for i in range(n3d)] + [None],
            v=v_col[frame_idx], is_axon=name.startswith('axon')))
    groups = {k: g for k, g in groups.items() if g['secs']}
    if not groups and not somas:
        raise ValueError("No recorded sections with 3D points were found in v_recorders.")
    for g in groups.values():
        for c in 'xyz':
            g[c] = [p for s in g['secs'] for p in s[c]]
        g['rep'] = np.array([len(s['x']) for s in g['secs']])
        g['V'] = np.array([s['v'] for s in g['secs']])          # sections x frames

    def _colours(g, k):
        return np.round(np.repeat(g['V'][:, k], g['rep']), 2).tolist()

    vmin, vmax = v_range if v_range else (-80, 30)

    # 3D view: fit everything except the axon (or a fixed half-width if view_um is given)
    if view_um:
        ranges = [[-view_um, view_um], [-view_um, view_um], [-view_um / 3, view_um / 3]]
    else:
        pts = [(x, y, z) for g in groups.values() for s in g['secs'] if not s['is_axon']
               for x, y, z in zip(s['x'], s['y'], s['z']) if x is not None]
        pts = np.array(pts + [tuple(p) for sm in somas for p in sm['xyz']], dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.08 * (hi - lo).max()
        ranges = [[float(a - pad), float(b + pad)] for a, b in zip(lo, hi)]
    spans = np.array([r[1] - r[0] for r in ranges])
    aspect = dict(zip('xyz', (spans / spans.max()).tolist()))

    # --- Figure layout: 3D cell (left) + one trace panel per row (right) --------
    specs = [[{'type': 'scene', 'rowspan': n_rows}, {'type': 'xy'}]] + \
            [[None, {'type': 'xy'}] for _ in range(n_rows - 1)]
    fig = make_subplots(rows=n_rows, cols=2, shared_xaxes=True, specs=specs,
                        column_widths=[0.46, 0.54], horizontal_spacing=0.12,
                        vertical_spacing=min(0.06, 0.3 / n_rows),
                        subplot_titles=["Membrane potential"] + [p[0] for p in panels])

    tl = t.tolist()
    y_ranges = {}
    for r, (ptitle, labels) in enumerate(panels, start=1):
        if labels is None:          # gates panel
            for k, lab in (('m', 'm (Na act.)'), ('h', 'h (Na inact.)'), ('n', 'n (K act.)')):
                if k in gates:
                    fig.add_trace(go.Scatter(x=tl, y=np.asarray(gates[k]).tolist(), name=lab,
                                             legendgroup='gates',
                                             line=dict(width=1.5, color=_GATE_COLOURS[k])), row=r, col=2)
            y_ranges[r] = [0, 1]
            fig.update_yaxes(title_text='Gate', row=r, col=2)
            continue
        vs = [site_info[l]['v'] for l in labels]
        for l, v in zip(labels, vs):
            fig.add_trace(go.Scatter(x=tl, y=v.tolist(), name=l, showlegend=len(labels) > 1,
                                     line=dict(color=site_info[l]['colour'], width=1.5)), row=r, col=2)
        lo, hi = min(v.min() for v in vs), max(v.max() for v in vs)
        pad = max(2.0, 0.08 * (hi - lo))
        y_ranges[r] = [float(lo - pad), float(hi + pad)]
        fig.update_yaxes(title_text='Vm (mV)', row=r, col=2)
    for r in range(1, n_rows + 1):
        fig.update_yaxes(range=y_ranges[r], row=r, col=2)
        for label, t0, t1, colour in protocol:
            fig.add_vrect(x0=t0, x1=t1, fillcolor=colour, opacity=0.15, line_width=0,
                          layer='below', row=r, col=2)
    fig.update_xaxes(title_text='Time (ms)', row=n_rows, col=2)

    # Recording-site markers on the cell (sites outside the 3D view are listed in the hover only)
    for l, info in site_info.items():
        if somas and 'soma' in info['seg'].sec.name():
            continue                          # the soma itself shows where this trace is from
        x, y, z = seg_xyz(info['seg'])
        if ranges[0][0] <= x <= ranges[0][1] and ranges[1][0] <= y <= ranges[1][1]:
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z + 2], mode='markers', name=l, showlegend=False,
                                       marker=dict(size=7, color=info['colour'], symbol='diamond'),
                                       hovertext=[l], hoverinfo='text'), row=1, col=1)

    # --- Animated traces: 3D cell, time markers, status text ------------------
    start_anim = len(fig.data)
    for sm in somas:
        fig.add_trace(go.Mesh3d(
            x=sm['xyz'][:, 0], y=sm['xyz'][:, 1], z=sm['xyz'][:, 2], i=sm['I'], j=sm['J'], k=sm['K'],
            intensity=np.full(len(sm['xyz']), round(float(sm['v'][0]), 2)), intensitymode='vertex',
            colorscale='Magma', cmin=vmin, cmax=vmax, showscale=not groups, flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.75, specular=0.35, roughness=0.4, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=1000),
            colorbar=dict(title="mV", x=0.41, thickness=15, len=0.6),
            hovertext=sm['name'], hoverinfo='text', name=sm['name'], showlegend=False), row=1, col=1)
    last = list(groups)[-1] if groups else None
    for gname, g in groups.items():
        fig.add_trace(go.Scatter3d(
            x=g['x'], y=g['y'], z=g['z'], mode='lines', hoverinfo='skip', showlegend=False,
            line=dict(color=_colours(g, 0), colorscale='Magma', cmin=vmin, cmax=vmax, width=g['width'],
                      colorbar=dict(title="mV", x=0.41, thickness=15, len=0.6) if gname == last else None)),
            row=1, col=1)
    for r in range(1, n_rows + 1):
        fig.add_trace(go.Scatter(x=[t[0], t[0]], y=y_ranges[r], mode='lines', hoverinfo='skip',
                                 line=dict(color='red', width=2), showlegend=False), row=r, col=2)

    t_fmt = '.1f' if frame_ms >= 0.5 else '.2f'

    def _status(tc):
        lines = [f"<b>t = {tc:{t_fmt}} ms</b>"]
        for label, t0, t1, colour in protocol:
            on = t0 <= tc < t1
            lines.append(f"<span style='color:{colour if on else 'gray'}'>"
                         f"{'<b>' if on else ''}{label}: {'ON' if on else 'off'}{'</b>' if on else ''}</span>")
        return "<br>".join(lines)

    fig.add_trace(go.Scatter(x=[t[0]], y=[y_ranges[1][1]], mode='text', text=[_status(t[0])],
                             textposition='bottom right', textfont=dict(size=12),
                             showlegend=False, hoverinfo='skip'), row=1, col=2)
    anim_idx = list(range(start_anim, len(fig.data)))

    frames = []
    for k, tc in enumerate(t_frames):
        data = [go.Mesh3d(intensity=np.full(len(sm['xyz']), round(float(sm['v'][k]), 2)))
                for sm in somas]
        data += [go.Scatter3d(line=dict(color=_colours(g, k))) for g in groups.values()]
        data += [go.Scatter(x=[tc, tc])] * n_rows
        data.append(go.Scatter(text=[_status(tc)]))
        frames.append(go.Frame(data=data, name=f"{tc:.4f}", traces=anim_idx))
    fig.frames = frames

    # Slider ticks at 'nice' intervals (about 10 labels)
    raw = (t[-1] - t[0]) / 10
    mag = 10 ** np.floor(np.log10(raw))
    tick_every = min((m * mag for m in (1, 2, 5, 10)), key=lambda v: abs(v - raw))

    def _tick(tc):
        q = tc / tick_every
        return f"{round(q) * tick_every:g}" if abs(q - round(q)) * tick_every < dt / 2 else ""

    steps = [dict(method='animate', label=_tick(tc),
                  args=[[f"{tc:.4f}"], dict(mode='immediate', frame=dict(duration=0, redraw=True),
                                            transition=dict(duration=0))])
             for tc in t_frames]

    camera_top = dict(eye=dict(x=0, y=0, z=1.6), up=dict(x=0, y=1, z=0))
    fig.update_layout(
        title=title, height=panel_height * n_rows + 220,
        scene=dict(aspectmode='manual', aspectratio=aspect, camera=camera_top,
                   xaxis=dict(title='x (µm)', range=ranges[0]), yaxis=dict(title='y (µm)', range=ranges[1]),
                   zaxis=dict(title='', range=ranges[2], showticklabels=False)),
        legend=dict(x=1.01, y=0.5, yanchor='middle'),
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=-0.03, xanchor='left',
                          yanchor='top', buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label='❚❚ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate',
                                    transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.12, y=-0.01, len=0.86, steps=steps,
                      currentvalue=dict(visible=False), pad=dict(t=30, b=10))],
        margin=dict(l=10, r=10, t=80, b=90))

    if save_html:
        fig.write_html(save_html, include_plotlyjs='cdn', auto_play=False)
    if show_fig:
        fig.show()
    return fig
