"""
Helper functions for the UniMelb PhD Neuroscience Computational Neuroscience Workshop
(Hodgkin–Huxley modelling with the Sheasby & Fohlmeister 1999 retinal ganglion cell).

Contents
  1. Spikes                                      find_spikes, spike_half_width
  2. Plotting                                    make_figure, plot_hh_panels
  3. Full cell (Activities 4–5)                  load_rgc_model, update_rx_3d, plot_final_rgc_with_electrode,
                                                 seg_xyz, record_segments, dendrite_path_sites,
                                                 spike_origin, furthest_in_view
  4. Interactive dashboard                       plot_interactive_dashboard
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from neuron import h

h.load_file('stdrun.hoc')          # provides h.continuerun

PEAK_HEIGHT = -20.0                # mV : default spike-detection threshold
GATE_COLOURS = {'m': '#2a78d6', 'h': '#eb6834', 'n': '#1baf7a'}
_SITE_COLOURS = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
_GATE_COLOURS = {'m': '#17becf', 'h': '#9467bd', 'n': '#8c564b'}

# ══════════════════════════════════════════════════════════════════════════════
# 1. SPIKES
# ══════════════════════════════════════════════════════════════════════════════
def find_spikes(v, height=PEAK_HEIGHT, prominence=10):
    """Indices of action-potential peaks (above height, ignoring small ripples)."""
    peaks, _ = find_peaks(v, height=height, prominence=prominence)
    return peaks


def spike_half_width(t, v, i_pk, dvdt_thres=10.0):
    """Width (ms) at half-amplitude, measured from the spike's threshold (dV/dt > 10 V/s) to its peak."""
    dvdt = np.gradient(v, t)
    i0 = max(0, int(np.searchsorted(t, t[i_pk] - 2.0)))          # look back up to 2 ms
    i_th = i0 + int(np.argmax(dvdt[i0:i_pk + 1]))                # steepest point of the upstroke
    while i_th > 0 and dvdt[i_th - 1] > dvdt_thres:               # back to where dV/dt crosses threshold
        i_th -= 1
    half = v[i_th] + (v[i_pk] - v[i_th]) / 2
    lo, hi = i_pk, i_pk
    while lo > 0 and v[lo - 1] >= half:
        lo -= 1
    while hi < len(v) - 1 and v[hi + 1] >= half:
        hi += 1
    return t[hi] - t[lo]


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def make_figure(nrows=1, ncols=1, close=True, **kwargs):
    """plt.subplots(nrows, ncols, **kwargs) with the interactive toolbar moved to the right, so it
    does not cover the axis labels. Closes earlier figures first unless close=False (use that for
    a second figure in the same cell). Returns (fig, axes)."""
    if close:
        plt.close('all')
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    try:
        fig.canvas.toolbar_position = 'right'   # only exists for interactive (widget) figures
        fig.canvas.header_visible = False
    except Exception:
        pass
    return fig, axes


def plot_hh_panels(t, i_stim, v_traces, m, h_gate, n, gnabar, gkbar, extra_panels=(),
                   title=None, stim_label='IClamp (pA)'):
    """
    The standard figure for every activity, all on one time axis: the injected current, Vm,
    the gating variables (m, h, n) and the Na⁺ and K⁺ conductances.

    t, i_stim    : time (ms) and injected current (pA)
    v_traces     : Vm traces for the Vm panel: [(label, Vm, colour), ...]
    m, h_gate, n : gating variables, recorded where the conductances are calculated
    gnabar, gkbar: maximal conductances there (S/cm²), so gNa = gnabar·m³h and gK = gkbar·n⁴
    extra_panels : more Vm panels below the first, e.g. along the dendrite and the axon:
                   [(y label, [(label, Vm, colour), ...]), ...]
    """
    t, m, h_gate, n = (np.asarray(x, dtype=float) for x in (t, m, h_gate, n))
    n_rows = 4 + len(extra_panels)
    fig, axes = make_figure(n_rows, 1, figsize=(9.5, 2 + 1.8 * n_rows), sharex=True,
                            gridspec_kw={'height_ratios': [1] + [2] * (n_rows - 1)})
    axes[0].plot(t, np.asarray(i_stim, dtype=float), color='k')
    axes[0].set_ylabel(stim_label)
    for ax, (ylabel, traces) in zip(axes[1:], [('Vm (mV)', v_traces)] + list(extra_panels)):
        for label, v, colour in traces:
            ax.plot(t, np.asarray(v, dtype=float), color=colour, lw=1.2, label=label)
        ax.set_ylabel(ylabel)
        if len(traces) > 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax_gate, ax_g = axes[-2], axes[-1]
    ax_gate.plot(t, m, label='m (Na act.)', color=GATE_COLOURS['m'])
    ax_gate.plot(t, h_gate, label='h (Na inact.)', color=GATE_COLOURS['h'])
    ax_gate.plot(t, n, label='n (K act.)', color=GATE_COLOURS['n'])
    ax_gate.set_ylabel('Gate')
    ax_gate.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax_g.plot(t, gnabar * m**3 * h_gate * 1000, label='gNa', color=GATE_COLOURS['m'])     # mS/cm²
    ax_g.plot(t, gkbar * n**4 * 1000, label='gK (delayed rect.)', color=GATE_COLOURS['n'])
    ax_g.set_ylabel('g (mS/cm²)')
    ax_g.set_xlabel('Time (ms)')
    ax_g.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    if title:
        axes[0].set_title(title)
    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. FULL CELL: MODEL, ELECTRODE AND MORPHOLOGY
# ══════════════════════════════════════════════════════════════════════════════
def load_rgc_model(morph_dir="neuron_morphology"):
    """Load the Sheasby & Fohlmeister (1999) RGC (ctt3219b) from morph_dir, flip it so the soma faces
    up, centre it on the soma, and insert the passive and extracellular (xtra) mechanisms.
    Does nothing if the model is already loaded."""
    if hasattr(h, 'soma'):
        print("Model already loaded.")
        return
    h.load_file(f"{morph_dir}/neuron_morphology_ctt3219b.asc")
    h.load_file(f"{morph_dir}/neuron_morphology_ctt3219b.hoc")

    def points(sec):
        return [(h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec), h.diam3d(i, sec=sec))
                for i in range(int(h.n3d(sec=sec)))]

    # Reverse the z coordinates to flip the morphology (soma facing up). The soma's 3D points are
    # also put in reverse order: the soma is connected to the AIS by its x = 1 end, and NEURON
    # attaches a section's first 3D point to its parent, so without this the AIS (which has no
    # 3D points of its own) would be drawn running back through the soma.
    for sec in h.allsec():
        pts = points(sec)
        if len(pts) < 2:
            continue
        if sec == h.soma:
            pts = pts[::-1]
        sec.pt3dclear()
        for x, y, z, d in pts:
            sec.pt3dadd(x, y, -z, d)
    h.define_shape()

    # Centre the cell on the soma, and insert the mechanisms
    cx, cy, cz = np.mean([p[:3] for p in points(h.soma)], axis=0)
    for sec in h.allsec():
        sec.insert('extracellular')                    # extracellular field ...
        sec.insert('xtra')                             # ... from a point electrode (xtra.mod)
        sec.insert('pas')
        pts = points(sec)
        if len(pts) < 2:
            continue
        sec.pt3dclear()
        for x, y, z, d in pts:
            sec.pt3dadd(x - cx, y - cy, z - cz, d)
        for seg in sec:                                # connect xtra to the extracellular mechanism
            h.setpointer(seg._ref_e_extracellular, 'ex', seg.xtra)
            h.setpointer(seg._ref_i_membrane, 'im', seg.xtra)

    h.ko0_k_ion = 5.0      # mM, extracellular K
    h.ki0_k_ion = 145.0    # mM, intracellular K
    h.define_shape()


def update_rx_3d(el_x, el_y, el_z, rho=100e4):
    """Place a point electrode at (el_x, el_y, el_z) µm: set each segment's transfer resistance
    (xtra.rx, MΩ) for a point source in a uniform medium of resistivity rho (Ω·µm)."""
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2:
            continue                                   # no 3D points
        L3d = h.arc3d(n3d - 1, sec=sec)
        for seg in sec:
            # first 3D point at or beyond the segment's centre
            i = next(k for k in range(n3d) if h.arc3d(k, sec=sec) >= seg.x * L3d)
            x, y, z = h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)
            r = max(np.sqrt((x - el_x)**2 + (y - el_y)**2 + (z - el_z)**2), 1.0)   # µm, avoid r = 0
            seg.xtra.rx = rho / (4 * np.pi * r) * 1e-6     # Ω -> MΩ


def plot_final_rgc_with_electrode(el_x, el_y, el_z, view_um=140):
    """3D plot of the cell (soma, AIS, axon, dendrites) and the electrode at (el_x, el_y, el_z) µm,
    in a view view_um µm either side of the soma (the axon runs off the edge)."""
    colours = {'Soma': '#555555', 'AIS': '#e67e22', 'Thin segment': '#82e0aa', 'Axon': '#1baf7a',
               'Dendrites': '#2a78d6'}
    widths = {'Soma': 8, 'AIS': 6, 'Thin segment': 4, 'Axon': 4, 'Dendrites': 2}
    fig = go.Figure()
    shown = set()                                      # one legend entry per part of the cell
    for sec in h.allsec():
        n3d = int(h.n3d(sec=sec))
        if n3d < 2:
            continue
        name = sec.name().lower()
        if 'soma' in name:                             # the soma: a shaded ellipsoid, as in the dashboards
            xyz, I, J, K = _soma_ellipsoid(sec)
            fig.add_trace(go.Mesh3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=I, j=J, k=K, color=colours['Soma'],
                flatshading=False, lighting=dict(ambient=0.45, diffuse=0.75, specular=0.35, roughness=0.4,
                                                 fresnel=0.2),
                lightposition=dict(x=100, y=200, z=1000), name='Soma', showlegend=True,
                hovertext=sec.name(), hoverinfo='text'))
            continue
        elif 'init' in name:
            label = 'AIS'
        elif 'narrow' in name:
            label = 'Thin segment'
        elif 'axon' in name:
            label = 'Axon'
        else:
            label = 'Dendrites'
        fig.add_trace(go.Scatter3d(
            x=[h.x3d(i, sec=sec) for i in range(n3d)],
            y=[h.y3d(i, sec=sec) for i in range(n3d)],
            z=[h.z3d(i, sec=sec) for i in range(n3d)],
            mode='lines', line=dict(color=colours[label], width=widths[label]),
            name=label, legendgroup=label, showlegend=label not in shown, hoverinfo='name'))
        shown.add(label)

    fig.add_trace(go.Scatter3d(                        # electrode, with a line down to the cell
        x=[el_x], y=[el_y], z=[el_z], mode='markers', name='Electrode',
        marker=dict(symbol='diamond', size=10, color='#f1c40f', line=dict(width=2, color='black'))))
    fig.add_trace(go.Scatter3d(
        x=[el_x, el_x], y=[el_y, el_y], z=[min(0, el_z - 50), el_z], mode='lines',
        line=dict(color='black', width=2, dash='dash'), showlegend=False, hoverinfo='none'))

    fig.update_layout(
        title="RGC morphology with electrode",
        scene=dict(xaxis=dict(title='X (µm)', range=[-view_um, view_um]),
                   yaxis=dict(title='Y (µm)', range=[-view_um, view_um]),
                   zaxis=dict(title='Z (µm)', range=[-0.4 * view_um, 0.4 * view_um]),
                   aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.4),
                   camera=dict(eye=dict(x=0.85, y=0.85, z=0.65))),    # closer than plotly's default view
        legend=dict(itemsizing='constant'), height=650,
        margin=dict(l=0, r=0, b=0, t=40))
    fig.show()
    return fig


def _x_at_points(sec):
    """Position along the section (0-1, as in sec(x)) of each of its 3D points.
    In this morphology some sections (the AIS and the thin segment) have their 3D points stored
    in the opposite direction to how they are connected, so the end of the 3D drawing that
    touches the section's x = 0 neighbour is found first."""
    n = int(h.n3d(sec=sec))
    xs = np.array([h.arc3d(i, sec=sec) for i in range(n)]) / max(h.arc3d(n - 1, sec=sec), 1e-9)
    # sections joined to this one's x = 0 end: its parent (if attached by that end) and its children there
    nbrs = [c for c in sec.children() if c.parentseg().x == 0]
    if sec.parentseg() is not None and sec.orientation() == 0:
        nbrs.append(sec.parentseg().sec)
    pts = np.array([[f(i, sec=s) for f in (h.x3d, h.y3d, h.z3d)]
                    for s in nbrs for i in range(int(h.n3d(sec=s)))])
    if len(pts):
        first = np.array([h.x3d(0, sec=sec), h.y3d(0, sec=sec), h.z3d(0, sec=sec)])
        last = np.array([h.x3d(n - 1, sec=sec), h.y3d(n - 1, sec=sec), h.z3d(n - 1, sec=sec)])
        if np.linalg.norm(pts - last, axis=1).min() < np.linalg.norm(pts - first, axis=1).min():
            xs = 1 - xs                            # 3D points run from x = 1 to x = 0
    return xs


def seg_xyz(seg):
    """3D coordinates (µm) of a segment, interpolated along its section's 3D points."""
    sec = seg.sec
    n = int(h.n3d(sec=sec))
    px = _x_at_points(sec)
    order = np.argsort(px)
    return tuple(float(np.interp(seg.x, px[order], np.array([f(i, sec=sec) for i in range(n)])[order]))
                 for f in (h.x3d, h.y3d, h.z3d))


def spike_origin(t, seg_recorders):
    """The segment, among those in seg_recorders ({section name: [Vm per segment]}), where the
    action potential peaks first. Returns (segment, its recorded Vm)."""
    t = np.asarray(t)
    by_name = {sec.name(): sec for sec in h.allsec()}
    pairs = [(seg, vec) for name, vecs in seg_recorders.items() for seg, vec in zip(by_name[name], vecs)]
    def peak_time(vec):                            # refined between time steps
        v = np.asarray(vec)
        i = int(np.clip(np.argmax(v), 1, len(v) - 2))
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        return t[i] + (t[1] - t[0]) * 0.5 * (v[i - 1] - v[i + 1]) / (den if den != 0 else -1e-9)
    return min(pairs, key=lambda p: peak_time(p[1]))


def furthest_in_view(seg_recorders, view_centre, view_um, margin=5):
    """The segment, among those in seg_recorders, that is furthest from the soma (along the cell)
    while still inside a 3D view of half-width view_um around view_centre (with a margin in µm).
    Returns (segment, its recorded Vm)."""
    by_name = {sec.name(): sec for sec in h.allsec()}
    lim = np.array([view_um, view_um, view_um / 3]) - margin
    inside = [(seg, vec) for name, vecs in seg_recorders.items() for seg, vec in zip(by_name[name], vecs)
              if np.all(np.abs(np.array(seg_xyz(seg)) - np.array(view_centre)) <= lim)]
    return max(inside, key=lambda p: h.distance(h.soma(0.5), p[0]))


def record_segments(sections):
    """Record Vm in every segment of each section: returns {section name: [h.Vector per segment]},
    for plot_interactive_dashboard(seg_recorders=...)."""
    return {sec.name(): [h.Vector().record(seg._ref_v) for seg in sec] for sec in sections}


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
                               frame_ms=5.0, v_range=None, view_um=None, view_centre=(0, 0, 0),
                               seg_recorders=None, t_window=None, frame_delay=40, colour_gamma=1.0,
                               mark_soma=False,
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
    v_range      : (min, max) mV of the colour scale (default: -80 mV to the peak Vm in the soma, or in
                   the seg_recorders sections). Anything higher is drawn in the top colour.
    view_um      : half-width (µm) of the 3D view, centred on view_centre. Default: fit the soma
                   and dendrites (the much longer axon runs off the edge of the view).
    view_centre  : (x, y, z) µm at the centre of the 3D view when view_um is given
    seg_recorders: optional {section name: [recorded Vm of each of its segments, in order]}.
                   These sections are coloured segment by segment (instead of by their middle),
                   so a spike can be seen moving along them.
    mark_soma    : True to also draw the marker of a site in the soma (above the soma)
    colour_gamma : 1 = a linear colour scale. Larger values (e.g. 3) spend more of the colours
                   near the top of the scale, so the action potential's peak stands out from
                   Vm values a few tens of mV below it.
    save_html    : file name to also save the dashboard as a standalone HTML page
    """
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    keep = np.ones(len(t), bool) if t_window is None else (t >= t_window[0]) & (t <= t_window[1])
    cut = lambda vec: np.asarray(vec, dtype=float)[keep]          # restrict to the time window
    t = t[keep]
    protocol = list(protocol)

    # --- Recording sites ------------------------------------------------------
    site_info = {}
    auto_colour = _default_site_colours([s[0] for s in sites])
    for s in sites:
        label, seg, vec = s[:3]
        colour = s[3] if len(s) > 3 else auto_colour[label]
        site_info[label] = dict(seg=seg, v=cut(vec), colour=colour)
    if panels is None:
        panels = [(f"{label} Vm", [label]) for label in site_info]
    panels = [(ptitle, list(labels)) for ptitle, labels in panels]
    if gates:
        panels.append(("Soma gates", None))
    n_rows = len(panels)

    # --- Vm at every 3D point of the cell, at the animation frames --------------
    frame_idx = np.arange(0, len(t), max(1, int(round(frame_ms / dt))))
    t_frames = t[frame_idx]
    seg_recorders = seg_recorders or {}

    groups = {'detail': dict(width=10, secs=[]), 'initseg': dict(width=6, secs=[]),
              'other': dict(width=3, secs=[])}
    somas = []
    for sec in h.allsec():
        name = sec.name()
        n3d = int(h.n3d(sec=sec))
        vec = v_recorders.get(name, v_recorders.get(f"{name}(0.5)"))
        if n3d < 2 or (vec is None and name not in seg_recorders):
            continue
        if name in seg_recorders:                 # one recording per segment: interpolate along the section
            seg_x = [seg.x for seg in sec]
            Vs = np.array([cut(v)[frame_idx] for v in seg_recorders[name]])     # segments x frames
            px = _x_at_points(sec)
            V_pts = np.array([np.interp(px, seg_x, Vs[:, k]) for k in range(len(frame_idx))]).T
            v_mid = Vs[len(seg_x) // 2]
        else:
            v_mid = cut(vec)[frame_idx]
            V_pts = np.repeat(v_mid[None, :], n3d, axis=0)
        if 'soma' in name:                        # drawn as a shaded ellipsoid, not a line
            xyz, I, J, K = _soma_ellipsoid(sec)
            somas.append(dict(xyz=xyz, I=I, J=J, K=K, v=v_mid, name=name))
            continue
        key = 'detail' if name in seg_recorders else 'initseg' if 'initseg' in name else 'other'
        groups[key]['secs'].append(dict(
            x=[h.x3d(i, sec=sec) for i in range(n3d)] + [None],
            y=[h.y3d(i, sec=sec) for i in range(n3d)] + [None],
            z=[h.z3d(i, sec=sec) for i in range(n3d)] + [None],
            V=np.vstack([V_pts, V_pts[-1:]]),     # one row per point, incl. the gap between sections
            is_axon=name.startswith('axon')))
    groups = {k: g for k, g in groups.items() if g['secs']}
    if not groups and not somas:
        raise ValueError("No recorded sections with 3D points were found in v_recorders.")
    for g in groups.values():
        for c in 'xyz':
            g[c] = [p for s in g['secs'] for p in s[c]]
        g['V'] = np.vstack([s['V'] for s in g['secs']])               # points x frames

    def _colours(g, k):
        return np.round(g['V'][:, k], 2).tolist()

    if v_range:
        vmin, vmax = v_range
    else:   # -80 mV to the peak of the action potential at the soma (or in the seg_recorders sections)
        if seg_recorders:
            top = max(cut(v).max() for vecs in seg_recorders.values() for v in vecs)
        elif somas:
            top = max(sm['v'].max() for sm in somas)
        else:
            top = max(g['V'].max() for g in groups.values())
        vmin, vmax = -80, float(np.ceil(top))
    # Colour scale: Magma, optionally stretched near the top (colour_gamma > 1)
    cmap = plt.get_cmap('magma')
    cscale = [[float(x), '#%02x%02x%02x' % tuple(int(255 * c) for c in cmap(x ** colour_gamma)[:3])]
              for x in np.linspace(0, 1, 41)]

    # 3D view: fit everything except the axon (or a fixed half-width if view_um is given)
    if view_um:
        cx, cy, cz = view_centre
        ranges = [[cx - view_um, cx + view_um], [cy - view_um, cy + view_um],
                  [cz - view_um / 3, cz + view_um / 3]]
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
    y_ranges, in_legend = {}, set()
    for r, (ptitle, labels) in enumerate(panels, start=1):
        if labels is None:          # gates panel
            for k, lab in (('m', 'm (Na act.)'), ('h', 'h (Na inact.)'), ('n', 'n (K act.)')):
                if k in gates:
                    fig.add_trace(go.Scatter(x=tl, y=cut(gates[k]).tolist(), name=lab,
                                             legendgroup='gates',
                                             line=dict(width=1.5, color=_GATE_COLOURS[k])), row=r, col=2)
            y_ranges[r] = [0, 1]
            fig.update_yaxes(title_text='Gate', row=r, col=2)
            continue
        vs = [site_info[l]['v'] for l in labels]
        for l, v in zip(labels, vs):
            fig.add_trace(go.Scatter(x=tl, y=v.tolist(), name=l.strip(), legendgroup=l.strip(),
                                     showlegend=len(labels) > 1 and l.strip() not in in_legend,
                                     line=dict(color=site_info[l]['colour'], width=1.5)), row=r, col=2)
            if len(labels) > 1:
                in_legend.add(l.strip())
        lo, hi = min(v.min() for v in vs), max(v.max() for v in vs)
        pad = max(2.0, 0.08 * (hi - lo))
        y_ranges[r] = [float(lo - pad), float(hi + pad)]
        fig.update_yaxes(title_text='Vm (mV)', row=r, col=2)
    for r in range(1, n_rows + 1):
        fig.update_yaxes(range=y_ranges[r], row=r, col=2)
        for label, t0, t1, colour in protocol:
            fig.add_vrect(x0=t0, x1=t1, fillcolor=colour, opacity=0.15, line_width=0,
                          layer='below', row=r, col=2)
    fig.update_xaxes(range=[float(t[0]), float(t[-1])], col=2)
    fig.update_xaxes(title_text='Time (ms)', row=n_rows, col=2)

    # Recording-site markers on the cell (sites outside the 3D view are listed in the hover only)
    for l, info in site_info.items():
        in_soma = 'soma' in info['seg'].sec.name()
        if somas and in_soma and not mark_soma:
            continue                          # the soma itself shows where this trace is from
        x, y, z = seg_xyz(info['seg'])
        if in_soma:
            z += h.soma.diam / 2 + 1          # above the soma, so it is not hidden inside it
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
            colorscale=cscale, cmin=vmin, cmax=vmax, showscale=not groups, flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.75, specular=0.35, roughness=0.4, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=1000),
            colorbar=dict(title="mV", x=0.41, thickness=15, len=0.6),
            hovertext=sm['name'], hoverinfo='text', name=sm['name'], showlegend=False), row=1, col=1)
    last = list(groups)[-1] if groups else None
    for gname, g in groups.items():
        fig.add_trace(go.Scatter3d(
            x=g['x'], y=g['y'], z=g['z'], mode='lines', hoverinfo='skip', showlegend=False,
            line=dict(color=_colours(g, 0), colorscale=cscale, cmin=vmin, cmax=vmax, width=g['width'],
                      colorbar=dict(title="mV", x=0.41, thickness=15, len=0.6) if gname == last else None)),
            row=1, col=1)
    for r in range(1, n_rows + 1):
        fig.add_trace(go.Scatter(x=[t[0], t[0]], y=y_ranges[r], mode='lines', hoverinfo='skip',
                                 line=dict(color='red', width=2), showlegend=False), row=r, col=2)

    t_fmt = '.1f' if frame_ms >= 0.5 else '.2f' if frame_ms >= 0.05 else '.3f'

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
        frames.append(go.Frame(data=data, name=f"f{k}", traces=anim_idx))
    fig.frames = frames

    # Slider ticks at 'nice' intervals (about 10 labels)
    raw = (t[-1] - t[0]) / 10
    mag = 10 ** np.floor(np.log10(raw))
    tick_every = min((m * mag for m in (1, 2, 5, 10)), key=lambda v: abs(v - raw))

    def _tick(tc):
        q = tc / tick_every
        return f"{round(q) * tick_every:g}" if abs(q - round(q)) * tick_every < dt / 2 else ""

    steps = [dict(method='animate', label=_tick(tc),
                  args=[[f"f{k}"], dict(mode='immediate', frame=dict(duration=0, redraw=True),
                                        transition=dict(duration=0))])
             for k, tc in enumerate(t_frames)]

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
                 args=[None, dict(frame=dict(duration=frame_delay, redraw=True), fromcurrent=True,
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