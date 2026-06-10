"""Rendering module for the crack-filling coverage planners.

Turns the trajectory output of the Sensor-based Complete Coverage (SCC) and
Online SCC planners into publication-quality figures and animated GIFs.  Three
entry points cover the main use cases:

- **Static summary figure** — ``plot_full_path`` draws the complete robot
  trajectory over the crack skeleton with start/end markers and key metrics.
- **Animated GIF** — ``render_animation`` replays the planner step by step,
  driven by an event stream captured via ``VizOptions``.  Three styles are
  available (``"plain"``, ``"publish"``, ``"color"``).
- **Live interactive figure** — ``SCCLiveDebug`` provides a per-stage
  debug view for the offline SCC planner; ``_LiveRenderer`` does the same for
  the online planner when ``VizOptions(mode="live")`` is used.

All figures use a workspace coordinate system with the origin at the top-left
(y increases downward), matching the image convention used internally by the
planners.  When ``style="publish"`` or ``"color"``, axes are displayed in metres
(1 px = 2 mm); otherwise they are in pixels.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib

matplotlib.use("Agg")          # headless: render to file, no display
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon, Rectangle

XLIM = (0, 3050)
YLIM = (2898, 0)               # y is image row -> origin at top (imshow convention)
PATH_COLOR = "#D95319"
SENSOR_R = 342                 # s  (sensor range radius, px)
NOZZLE_R = 44                  # a  (footprint/nozzle radius, px)
_HEAD_PX = 75                  # arrowhead triangle size (px, scaled by M at draw time)

# publish styling
_M_PER_PX = 0.002              # 1 px = 2 mm -> metres (axes in m, 3050px -> 6.1 m)
_CREAM = "#FBF0D5"            # workspace fill (color: outside-cells complement)
_ORANGE = "#D95319"          # workspace border / callout arrows
_REEB_PUBLISH = "#4DBEEE"    # single Reeb-edge colour for plain publish
_PALETTE = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30",
                  "#4DBEEE", "#A2142F"]   # cell colour cycle


@dataclass
class VizOptions:
    """Configuration for the planner's visualization output.

    Pass an instance to the planner (e.g. ``run_online_scc(_viz=VizOptions(...))``)
    to enable rendering.  The planner records per-iteration and per-pose events
    into ``.events`` during the run; calling ``finalize()`` converts those events
    into figures and/or a GIF.  ``finalize()`` is called automatically at the end
    of the planner run.

    Parameters
    ----------
    mode : str
        Controls what output is produced:

        ``"pos"``
            No rendering; print the robot's current position to stdout as the
            planner runs.
        ``"show"``
            Render the animation and open the result in the OS default viewer;
            no files are saved.
        ``"save"``
            Render the animation, write per-step PNGs and the final GIF to disk,
            and write a single final-path PNG; do not open a viewer (headless).
        ``"show+save"``
            Render, save files to disk, and open the result in the OS viewer.
        ``"live"``
            Draw frames on screen as the planner runs; requires an interactive
            matplotlib backend (TkAgg or QtAgg).
        ``"final"``
            Write only the single final-path summary PNG; no GIF is produced.

    style : str
        Controls how much detail is rendered:

        ``"plain"``
            Minimal: cell outlines, detected cracks, path line, and robot disks.
        ``"publish"``
            Detailed figure with axes in metres: cream workspace background, the
            full crack skeleton (undetected cracks shown light, detected cracks
            dark), the Reeb graph with edge labels, critical points, and a
            step-by-step callout sequence in the intro frames.
        ``"color"``
            Same as ``"publish"`` with each Morse cell filled and outlined in a
            distinct colour; the workspace outside the cells is filled cream.

    step : int
        Number of poses per animation frame (higher = shorter GIF, default 1).
    fps : int
        Frames per second for the output GIF (default 15).
    out_gif : str
        Output path for the animated GIF.
    final_png : str or None
        Output path for the single final-path PNG written in every mode.  If
        ``None``, the path is derived from ``out_gif`` by replacing the extension.
    frame_dir : str or None
        If set, individual PNG frames are also written to this directory.
    pos_every : int
        In ``mode="pos"``, print the robot position every this many poses.
    title : str
        Figure title prefix (e.g. ``"OnlineSCC — myCrack8_100_1"``).
    dpi : int or None
        Resolution for per-frame PNGs.  ``None`` selects 200 for hi-res styles,
        110 for ``"publish"``/``"color"`` previews, and 90 for plain.
    figsize : tuple or None
        Matplotlib figure size in inches.  ``None`` picks a size automatically.
    gif_max_px : int
        Cap the longer side of the GIF (in pixels); full-resolution PNGs are
        unaffected.  Default 1100.
    smooth_step_m : float
        If greater than zero, tween the robot disk between consecutive poses with
        sub-steps spaced this many metres apart (smooth motion); 0 = off.
    events : list
        Event log populated by the planner during the run.  Normally left empty
        at construction and filled automatically.
    """
    mode: str = "save"
    style: str = "plain"
    step: int = 1
    fps: int = 15
    out_gif: str = "Results/GIF/onlinescc.gif"
    final_png: str | None = None    # single final-path PNG (the deliverable, written in EVERY mode)
    frame_dir: str | None = None
    pos_every: int = 10
    title: str = "OnlineSCC"          # callers compose "{algo} — {map}"
    dpi: int | None = None          # None -> 170 publish / 90 plain (per-frame PNG res)
    figsize: tuple | None = None    # None -> auto per style
    gif_max_px: int = 1100          # cap the GIF's larger side (PNG frames stay full-res)
    smooth_step_m: float = 0.0      # >0 -> tween robot disk between poses (m); 0 = off
    events: list = field(default_factory=list)

    _MODES = ("pos", "show", "save", "show+save", "live", "final")
    _STYLES = ("plain", "publish", "color")

    def __post_init__(self):
        if self.mode not in self._MODES:
            raise ValueError(f"VizOptions.mode must be one of {self._MODES}, got {self.mode!r}")
        if self.style not in self._STYLES:
            raise ValueError(f"VizOptions.style must be one of {self._STYLES}, got {self.style!r}")
        # use publication resolution when saving a detailed style; previews stay lighter.
        self._hires = self.mode in ("save", "show+save", "final") and self.style in ("publish", "color")
        self._live = None                                # lazy _LiveRenderer (mode="live")

    # --- planner-side hooks (called by run_online_scc) ---
    def capture(self, event):
        self.events.append(event)
        if self.mode == "live":                          # draw on screen AS the run proceeds
            if self._live is None:
                self._live = _LiveRenderer(style=self.style, title=self.title, every=self.step)
            self._live.feed(event)

    def on_pose(self, it, i, curPt):
        if self.mode == "pos" and i % max(1, self.pos_every) == 0:
            print(f"  [viz] iter {it} pose {i}: pos=({curPt[0]:.1f}, {curPt[1]:.1f})", flush=True)

    def finalize(self):
        """Render the accumulated events and write output files according to ``mode``.

        Always writes the single final-path summary PNG (to ``final_png``, or a
        path derived from ``out_gif``).  In ``"save"`` / ``"show+save"`` modes,
        also produces the animated GIF and optional per-frame PNGs.  In
        ``"live"`` mode, holds the interactive window open.

        Returns
        -------
        (gif_path, n_frames) : tuple, or None
            Path to the written GIF and number of frames, or ``None`` when no GIF
            was produced (``"pos"``, ``"live"``, or empty event log).
        """
        import os
        import tempfile
        if self.mode == "pos" or not self.events:
            return None
        # Always write the single final-path PNG (hi-res for detailed styles).
        # GIFs (save modes) still go to out_gif; this PNG goes to final_png.
        fpng = self.final_png or (self.out_gif.rsplit(".", 1)[0] + ".png"
                                  if self.out_gif.endswith(".gif") else self.out_gif)
        if fpng:
            render_animation(self.events, out_gif=fpng, step=self.step, fps=self.fps,
                             title=self.title, style=self.style, dpi=self.dpi, figsize=self.figsize,
                             gif_max_px=self.gif_max_px, hires=self.style in ("publish", "color"),
                             final_only=True, out_png=fpng)
            print(f"  [viz] final-path frame -> {fpng}", flush=True)
        if self.mode == "live":                          # interactive already drawn -> hold the window
            if self._live is not None:
                self._live.hold()
            return None
        if self.mode == "final":                         # the final PNG above is the whole output
            return fpng, 1
        save = self.mode in ("save", "show+save")
        show = self.mode in ("show", "show+save")
        gif = self.out_gif if save else os.path.join(tempfile.gettempdir(), "onlinescc_preview.gif")
        fdir = self.frame_dir if save else None
        out, nf = render_animation(self.events, out_gif=gif, step=self.step, fps=self.fps,
                                   frame_dir=fdir, title=self.title, style=self.style,
                                   dpi=self.dpi, figsize=self.figsize, gif_max_px=self.gif_max_px,
                                   smooth_step_m=self.smooth_step_m, hires=self._hires)
        if show:
            _open_file(out)
        return out, nf


def _open_file(path):
    """Open a file in the OS default viewer (best-effort; no-op/print if unavailable)."""
    import os
    import subprocess
    import sys
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                       # noqa: S606 (Windows default app)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:                            # headless / no viewer
        print(f"  [viz] rendered {path} (could not open viewer: {e})", flush=True)


def _setup_axes(ax, title=None):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if title:
        ax.set_title(title)


def _draw_cracks(ax, skeleton):
    """Scatter the crack skeleton (row, col) as light points."""
    if skeleton is not None:
        m, n = np.nonzero(np.asarray(skeleton))
        ax.plot(n, m, ".", color="0.55", ms=0.6, zorder=1)


def _clean_path(PE):
    """Drop the [0,0] seed row and any NaN separators -> plain (N,2)."""
    PE = np.atleast_2d(np.asarray(PE, dtype=float))[:, :2]
    if PE.shape[0] and np.all(PE[0] == 0):
        PE = PE[1:]
    return PE


def plot_full_path(PE, skeleton=None, out_png=None, res=None,
                   title="OnlineSCC coverage + crack-fill path"):
    """Produce a static figure of the complete robot trajectory over the crack skeleton.

    Draws the full path as a continuous line with a start marker (▼) and end
    marker (▲).  If ``res`` is provided, key metrics are appended to the title.

    Parameters
    ----------
    PE : array-like, shape (N, 2)
        Path vertices as (x, y) pairs.  An initial ``[0, 0]`` seed row and NaN
        separators are stripped automatically.
    skeleton : 2-D array-like or None
        Binary crack-skeleton image.  If provided, skeleton pixels are drawn as
        light grey dots behind the path.
    out_png : str or None
        If set, save the figure to this path and close it; otherwise return the
        ``Figure`` object.
    res : array-like or None
        Optional result vector ``[numItr, density, time, pathLen_ft, areaCover]``.
        When provided, ``numItr``, ``pathLen`` (ft), and ``areaCover`` are shown
        in the figure subtitle.
    title : str
        Figure title.

    Returns
    -------
    str or Figure
        The path to the saved PNG when ``out_png`` is set, otherwise the
        matplotlib ``Figure`` object.
    """
    path = _clean_path(PE)
    fig, ax = plt.subplots(figsize=(9, 8.6))
    _draw_cracks(ax, skeleton)
    # NaN rows naturally break the line between disconnected iterations
    ax.plot(path[:, 0], path[:, 1], "-", color=PATH_COLOR, lw=1.0, zorder=2,
            label=f"path ({path.shape[0]} pts)")
    ax.plot(path[0, 0], path[0, 1], "v", color="#0072BD", ms=12,
            markerfacecolor="#0072BD", zorder=3, label="start")
    ax.plot(path[-1, 0], path[-1, 1], "^", color="#77AC30", ms=12,
            markerfacecolor="#77AC30", zorder=3, label="end")
    sub = title
    if res is not None:
        r = np.atleast_1d(np.asarray(res, dtype=float)).ravel()
        sub += f"\nnumItr={int(r[0])}  oscclen={r[3]:.1f} ft  areaCover={r[4]:.3f}"
    _setup_axes(ax, sub)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        return out_png
    return fig


def animate_full_path(PE, skeleton=None, out_gif="path.gif", step=2, fps=20,
                      show_footprint=True, dpi=90,
                      title="OnlineSCC: Coverage + Crack-Fill"):
    """Produce an animated GIF of the robot traversing the full trajectory.

    Each frame extends the cumulative path line by ``step`` poses and moves the
    sensor and nozzle footprint disks to the robot's current position.

    Parameters
    ----------
    PE : array-like, shape (N, 2)
        Path vertices as (x, y) pairs.
    skeleton : 2-D array-like or None
        Binary crack-skeleton image drawn as background dots.
    out_gif : str
        Output GIF file path.
    step : int
        Number of poses to advance per frame; larger values produce shorter GIFs.
    fps : int
        Frames per second in the output GIF.
    show_footprint : bool
        If True, draw the sensor disk (yellow) and nozzle footprint disk (orange)
        at the robot's current position.
    dpi : int
        Figure resolution in dots per inch.
    title : str
        Figure title.

    Returns
    -------
    (out_gif, n_frames) : tuple[str, int]
    """
    import imageio.v2 as imageio

    path = _clean_path(PE)
    N = path.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7.7))
    _draw_cracks(ax, skeleton)
    _setup_axes(ax, title)
    (line,) = ax.plot([], [], "-", color=PATH_COLOR, lw=1.0, zorder=2)
    (head,) = ax.plot([], [], "o", color=PATH_COLOR, ms=4, zorder=4)
    ax.plot(path[0, 0], path[0, 1], "v", color="#0072BD", ms=10,
            markerfacecolor="#0072BD", zorder=3)
    sensor = Circle((path[0, 0], path[0, 1]), SENSOR_R, fill=True, fc="#FFD60A",
                    ec="#E0A800", alpha=0.18, zorder=2) if show_footprint else None
    nozzle = Circle((path[0, 0], path[0, 1]), NOZZLE_R, fill=True, fc="#D95319",
                    ec="#D95319", alpha=0.5, zorder=4) if show_footprint else None
    if sensor is not None:
        ax.add_patch(sensor); ax.add_patch(nozzle)
    fig.tight_layout()

    frames = []
    idxs = list(range(1, N + 1, max(1, step)))
    if idxs[-1] != N:
        idxs.append(N)
    for k in idxs:
        line.set_data(path[:k, 0], path[:k, 1])
        cur = path[k - 1]
        head.set_data([cur[0]], [cur[1]])
        if sensor is not None:
            sensor.center = (cur[0], cur[1]); nozzle.center = (cur[0], cur[1])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
    plt.close(fig)
    # imageio deprecated `fps`; per-frame `duration` is in milliseconds.
    imageio.mimsave(out_gif, frames, duration=1000.0 / fps, loop=0)
    return out_gif, len(frames)


def render_animation(viz, out_gif="Results/GIF/onlinescc.gif", step=1, fps=15,
                     dpi=None, title="OnlineSCC: Coverage + Crack-Fill",
                     frame_dir=None, figsize=None, style="plain", units=None,
                     intro_pause=14, decomp_pause=10, final_pause=24, gif_max_px=1100,
                     want_tags=None, smooth_step_m=0.0, hires=False,
                     final_only=False, out_png=None):
    """Render the planner's run as an animated GIF from a captured event stream.

    Replays the sequence of ``"init"``, ``"iter"``, and ``"pose"`` events
    recorded by ``VizOptions`` during a planner run, producing one output frame
    per ``step`` poses.  All map geometry and workspace bounds are read from the
    event stream; no external data files are required.

    Parameters
    ----------
    viz : list of dict
        Event stream captured by ``VizOptions.events``.
    out_gif : str
        Output path for the animated GIF.
    step : int
        Poses per frame.  Higher values produce shorter GIFs.
    fps : int
        Frames per second in the output GIF.
    dpi : int or None
        Per-frame PNG resolution.  ``None`` selects 200 (hi-res), 110 (publish
        preview), or 90 (plain) based on ``style`` and ``hires``.
    title : str
        Figure title prefix.
    frame_dir : str or None
        If set, individual full-resolution PNG frames are also written here.
    figsize : tuple or None
        Matplotlib figure size in inches.  ``None`` selects an automatic size.
    style : str
        Rendering style — one of:

        ``"plain"``
            Minimal: cell outlines, detected cracks, path line, robot disks.
        ``"publish"``
            Detailed figure with axes in metres.  Shows the full crack skeleton
            (undetected cracks light, detected cracks dark), the Reeb graph
            with edge and critical-point labels, connecting and wall-follow
            edges, the planned path with a direction arrowhead, and sensor /
            nozzle disks.  The first frames play a callout intro sequence
            (Start Point, Footprint Range, Sensor Range, Scanned Cracks,
            Unknown Cracks) before the first cell decomposition.  A "Final
            Path" frame closes the animation.
        ``"color"``
            Same as ``"publish"`` with each Morse cell filled and outlined in
            a distinct colour; the workspace outside the cells is filled cream.

    units : str or None
        Axis units string (``"m"`` or ``"px"``).  ``None`` selects ``"m"`` for
        publish/color styles and ``"px"`` for plain.
    intro_pause : int
        Number of frames each intro callout card is held (default 14).
    decomp_pause : int
        Number of frames the decomposition card is held (default 10).
    final_pause : int
        Number of frames the final-path card is held (default 24).
    gif_max_px : int
        Cap the longer side of the GIF in pixels; full-resolution PNGs are
        unaffected.  Default 1100.
    want_tags : set or None
        If set, render only the frames whose tag is in this set and return
        immediately.  ``None`` renders all frames.
    smooth_step_m : float
        If greater than zero, insert interpolated sub-frames between consecutive
        poses so the robot disk glides smoothly; spacing is in metres.
    hires : bool
        If True, use publication resolution (dpi 200, ~2000 px frames).
    final_only : bool
        If True, render only the final "Final Path" summary frame; write it as
        a PNG to ``out_png`` and return immediately without producing a GIF.
    out_png : str or None
        Output path for a single PNG frame when ``final_only=True``.

    Returns
    -------
    (out_gif, n_frames) : tuple[str, int]
        Path to the written GIF and number of frames, or ``(png_path, 1)`` when
        ``final_only=True``.
    """
    import os
    import imageio.v2 as imageio

    publish = style in ("publish", "color")
    color_cells = style == "color"
    if units is None:
        units = "m" if publish else "px"
    M = _M_PER_PX if units == "m" else 1.0           # coordinate scale (px->units)
    # hires -> publication res (~2000px PNGs); else lighter (styled 110, plain 90 dpi).
    if dpi is None:
        dpi = 200 if hires else (110 if publish else 90)
    if figsize is None:
        figsize = (10.0, 9.6) if publish else (8.2, 7.9)
    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)

    # ----- accumulated render state -----
    skel = None                                       # full crack skeleton (col,row)
    skel_idx = {}                                      # (row,col) -> skeleton row index
    det = np.zeros(0, dtype=bool)                      # detected-mask over skeleton rows
    det_set = set()                                    # (row,col) sensed so far
    path_pts = []                                      # all actual poses (x,y) -> final frame
    done_path = []                                      # actual poses of completed iterations
    cur_poses = []                                      # actual poses of the current iteration
    planned = np.empty((0, 2))                          # current iter's full planned path
    cells, critP, reeb = [], np.empty((0, 2)), []
    reebwall, reebCell, wall_fol, Path = [], np.empty(0, int), [], []
    is_scc = False                                     # SCC (offline) run -> SCC intro/overlays
    scc_nodes = np.empty((0, 2))                        # crack-graph nodes (x,y)
    scc_edges = np.empty((0, 2), int)                  # crack-graph edges (node-index pairs)
    scc_adds = np.empty((0, 2, 2))                     # Chinese Postman matched edges (x,y pairs)
    cur_curPt = np.array([0.0, 0.0])                    # robot position entering current iteration
    shown_callouts = set()                               # explanatory callouts shown once each
    last_pos = None                                      # last emitted robot position (for tweening)
    smooth_px = (smooth_step_m / _M_PER_PX) if smooth_step_m else 0.0   # tween step (px)
    cur_it, bounds = 0, (3050, 2898)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    frames, pose_n, frame_n = [], 0, 0

    remaining = set(want_tags) if want_tags is not None else None
    if final_only:                                    # render ONLY the "Final Path" frame
        remaining = {"final"}
    stop_render = False

    def _grab(repeat=1, tag=None):
        nonlocal frame_n, stop_render
        if remaining is not None and tag not in remaining:
            frame_n += repeat                       # skip the expensive draw entirely
            return
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()   # full-res frame
        gframe = _downscale(buf, gif_max_px)                          # GIF: capped size
        for _ in range(repeat):
            frames.append(gframe)
            if frame_dir:
                name = f"{_safe_tag(tag)}.png" if remaining is not None else f"{frame_n}.png"
                imageio.imwrite(os.path.join(frame_dir, name), buf)
            frame_n += 1
        if remaining is not None and tag in remaining:
            remaining.discard(tag)
            if not remaining:
                stop_render = True

    def _wanted(tag):
        return remaining is None or tag in remaining

    def _cell_color(k):
        return _PALETTE[int(k) % len(_PALETTE)]

    def _base_axes(subtitle):
        # Pure axis setup only. The workspace fill is a separate element (_draw_workspace)
        # so the "Final Path" frame can opt out of the cream fill.
        ax.clear()
        cb, rb = bounds
        # axis fits the workspace exactly -- no padding gap to the (axes-box) border
        ax.set_xlim(0, cb * M); ax.set_ylim(rb * M, 0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x ({units})"); ax.set_ylabel(f"y ({units})")
        if subtitle:
            ax.set_title(subtitle, fontsize=10)

    def _draw_workspace():
        # No fill and no outline here. The color style paints the cream complement
        # around the cells in _draw_cells; publish stays on white.
        if publish:
            cb, rb = bounds
            ax.add_patch(Rectangle((0, 0), cb * M, rb * M, fc="none", ec="none",
                                   lw=0, zorder=0))

    def _draw_cracks(thin_color="0.78"):       # undetected cracks shown lighter
        if not publish:                                   # plain: only revealed (sensed)
            if det_set:
                r, c = np.array(sorted(det_set)).T
                ax.plot(c * M, r * M, ".", color="0.12", ms=0.9, zorder=2)
            return
        if skel is None:
            return
        c, r = skel[:, 0], skel[:, 1]                     # publish: full thin + detected thick
        ax.plot(c * M, r * M, ".", color=thin_color, ms=0.35, zorder=1)
        if det.any():
            ax.plot(c[det] * M, r[det] * M, ".", color="0.0", ms=1.7, zorder=2)

    def _draw_cells():
        if color_cells:
            # Fill the workspace outside the current cells with cream, then draw colour cells on top.
            from shapely.geometry import box as _sbox
            cb, rb = bounds
            u = _cells_to_polygon(cells)
            comp = _sbox(0, 0, cb, rb)
            comp = comp.difference(u) if u is not None else comp
            _fill_geom(ax, comp, M, _CREAM, 0.3)
            for k, cell in enumerate(cells):
                xy = np.atleast_2d(cell)
                col = _cell_color(k)
                ax.fill(xy[:, 0] * M, xy[:, 1] * M, facecolor=col, alpha=0.18, zorder=1)
                ax.plot(xy[:, 0] * M, xy[:, 1] * M, "-", color=col, lw=1.6, zorder=2)
            return
        for k, cell in enumerate(cells):
            xy = np.atleast_2d(cell)
            if publish:
                ax.plot(xy[:, 0] * M, xy[:, 1] * M, "-", color="0.55", lw=1.0, zorder=2)
            else:
                ax.plot(xy[:, 0] * M, xy[:, 1] * M, "-", color="0.78", lw=0.8, zorder=1)

    def _reeb_edge_color(j):
        # In color style each Reeb edge takes its cell's colour; publish uses a single light-blue.
        if color_cells and j < len(reebCell):
            return _cell_color(reebCell[j])
        return _REEB_PUBLISH

    def _node_xy(i):
        return critP[int(i)] if 0 <= int(i) < len(critP) else None

    def _draw_reeb(callouts=False):
        if not publish:
            return
        reeb_cb = conn_cb = wall_cb = None       # remembered points for the callouts
        # --- Reeb edges (solid) + E# labels ---
        for j, e in enumerate(reeb):
            e = np.atleast_2d(np.asarray(e, float))
            xs, ys = (_spline3(e[:, 0], e[:, 1]) if e.shape[0] >= 3 else (e[:, 0], e[:, 1]))
            ax.plot(xs * M, ys * M, "-", color=_reeb_edge_color(j), lw=2.0, zorder=4)
            mid = e[len(e) // 2]
            ax.text((mid[0] + 20) * M, (mid[1] + 70) * M, f"E{j + 1}",
                    color=_reeb_edge_color(j), fontsize=8, zorder=7)
            if reeb_cb is None:
                reeb_cb = mid
        # --- wall-follow edges: dashed "doubled" edge from the reebwall curve,
        # in the edge's colour, labelled E'#. wall_fol is Path-parallel (one trailing 0);
        # its leading entries align 1:1 with the reeb edges. ---
        for k in range(len(wall_fol)):
            if wall_fol[k] == 1 and k < len(reebwall):
                e = np.atleast_2d(np.asarray(reebwall[k], float))
                xs, ys = (_spline3(e[:, 0], e[:, 1]) if e.shape[0] >= 3 else (e[:, 0], e[:, 1]))
                ax.plot(xs * M, ys * M, "--", color=_reeb_edge_color(k), lw=2.0, zorder=4)
                mid = e[len(e) // 2]                    # E'# label (doubled wall-follow edge)
                ax.text((mid[0] - 70) * M, (mid[1] - 60) * M, f"E'{k + 1}",
                        color=_reeb_edge_color(k), fontsize=8, zorder=7)
                if wall_cb is None:
                    wall_cb = mid
        # --- Reeb connecting edges: always blue dash-dot ---
        def _conn(p, q):
            nonlocal conn_cb
            if p is None or q is None:
                return
            ax.plot([p[0] * M, q[0] * M], [p[1] * M, q[1] * M], "-.",
                    color="#0072BD", lw=2.2, zorder=4)
            if conn_cb is None:
                conn_cb = (np.asarray(p) + np.asarray(q)) / 2
        if cur_it > 1 and len(Path):                  # robot -> first cell's start node
            _conn(cur_curPt, _node_xy(Path[0][0]))
        for row in Path:                              # cell-connection jumps (cellOrd < 0)
            if int(row[2]) < 0:
                _conn(_node_xy(row[0]), _node_xy(row[1]))
        # --- critical points (red dots) + C# labels ---
        if len(critP):
            ax.plot(critP[:, 0] * M, critP[:, 1] * M, "r.", ms=12, zorder=6)
            for j, p in enumerate(critP):
                ax.text((p[0] + 20) * M, (p[1] + 70) * M, f"C{j + 1}",
                        color="r", fontsize=8, zorder=7)
        # --- explanatory callouts: arrow to the element, shown once each (first decomposition
        # where it appears). Text is placed close to the feature but clear of drawn elements
        # (cell edges, Reeb / connecting / wall-follow edges, path, critical points).
        if callouts:
            sstep = 0.012 * max(bounds)                   # sampling step, relative to map size
            obs = []                                     # drawn-element point cloud (#2)
            for c in cells:
                obs += _sample_polyline(np.atleast_2d(c)[:, :2], step_px=sstep)
            for e in reeb:
                obs += _sample_polyline(e, step_px=sstep)
            for k in range(len(wall_fol)):
                if wall_fol[k] == 1 and k < len(reebwall):
                    obs += _sample_polyline(reebwall[k], step_px=sstep)
            if cur_it > 1 and len(Path):
                obs += _sample_polyline([cur_curPt, _node_xy(Path[0][0])], step_px=sstep)
            for row in Path:
                if int(row[2]) < 0 and _node_xy(row[0]) is not None and _node_xy(row[1]) is not None:
                    obs += _sample_polyline([_node_xy(row[0]), _node_xy(row[1])], step_px=sstep)
            if done_path:
                obs += _sample_polyline(done_path, step_px=sstep)
            if len(planned):
                obs += _sample_polyline(planned, step_px=sstep)
            if len(critP):
                obs += [np.asarray(p, float) for p in critP]
            obs = np.asarray(obs, float) if obs else None

            def _explain(text, point):
                p = np.asarray(point, float)
                avoid = [(cur_curPt[0], cur_curPt[1], SENSOR_R)] if cur_it > 1 else []
                a = _free_anchor(p, obs, bounds, avoid=avoid, text=text)   # #2 off drawn elements
                _callout(ax, text, (p[0] * M, p[1] * M), (a[0] * M, a[1] * M))
            if "reeb" not in shown_callouts and reeb_cb is not None:
                _explain("Reeb Edge", reeb_cb); shown_callouts.add("reeb")
            if "conn" not in shown_callouts and conn_cb is not None:
                _explain("Reeb\nConnecting Edge", conn_cb); shown_callouts.add("conn")
            if "wall" not in shown_callouts and wall_cb is not None:
                _explain("Doubled Edges\nRepresents a Wall-Follow", wall_cb)
                shown_callouts.add("wall")

    def _draw_path(arrow=True, show_planned=True):
        # During poses: show the full planned path for the current iteration plus the
        # cumulative path of completed iterations. The decomposition frame shows only
        # completed iterations (show_planned=False).
        disp = list(done_path)
        if show_planned and len(planned):
            disp = disp + [tuple(p) for p in planned]
        if disp:
            d = np.asarray(disp, float)
            ax.plot(d[:, 0] * M, d[:, 1] * M, "--", color=PATH_COLOR,
                    lw=1.8 if publish else 1.3, zorder=5)
        start_xy = path_pts[0] if path_pts else (disp[0] if disp else None)
        if start_xy is not None:
            ax.plot(start_xy[0] * M, start_xy[1] * M, "v", color="#0072BD", ms=11,
                    markerfacecolor="#0072BD", zorder=8)
        if arrow and show_planned and len(planned) >= 2:   # tail-less arrowhead at end of planned path
            tip = planned[-1]; prev = planned[max(0, len(planned) - 3)]
            _arrowhead(ax, (prev[0] * M, prev[1] * M), (tip[0] * M, tip[1] * M),
                       PATH_COLOR, _HEAD_PX * M)

    def _draw_robot(cur):
        if cur is None:
            return
        ax.add_patch(Circle((cur[0] * M, cur[1] * M), SENSOR_R * M, fc="#FCE9A6",
                            ec="0.0" if publish else "#E0A800",
                            alpha=0.30 if publish else 0.18, lw=1.0, zorder=3))
        ax.add_patch(Circle((cur[0] * M, cur[1] * M), NOZZLE_R * math.sqrt(2) * M,
                            fc="#F4A582" if publish else "#D95319",
                            ec="#C0392B" if publish else "#D95319", alpha=0.85 if publish else 0.45,
                            zorder=6))

    # ---- SCC-specific overlays (offline planner) ----
    def _draw_scc_graph(stars=True):
        if scc_edges.size and scc_nodes.size:
            for u, v in scc_edges:
                ax.plot([scc_nodes[u, 0] * M, scc_nodes[v, 0] * M],
                        [scc_nodes[u, 1] * M, scc_nodes[v, 1] * M], "--",
                        color="#2ca02c", lw=1.8, zorder=4)
        if stars and scc_nodes.size:                       # crack endpoint markers
            ax.plot(scc_nodes[:, 0] * M, scc_nodes[:, 1] * M, "*", color="r",
                    ms=11, mec="0.2", mew=0.4, zorder=7)

    def _draw_scc_adds(arrows=True):
        # Chinese Postman matched edges: blue dash-dot + arrowhead
        for seg in scc_adds:
            p, q = np.asarray(seg[0], float), np.asarray(seg[1], float)
            ax.plot([p[0] * M, q[0] * M], [p[1] * M, q[1] * M], "-.",
                    color="#0072BD", lw=2.0, zorder=5)
            if arrows:
                _arrowhead(ax, (p[0] * M, p[1] * M), (q[0] * M, q[1] * M),
                           "#0072BD", _HEAD_PX * 0.6 * M)

    def _scc_intro():
        """Build-up intro sequence for the SCC planner: cracks, then crack graph, then cells/Reeb/matched edges."""
        _base_axes(f"{title}\nKnown Cracks"); _draw_workspace(); _draw_cracks()
        _grab(intro_pause, tag="intro:cracks")
        _base_axes(f"{title}\nCrack Graph (Visibility)"); _draw_workspace(); _draw_cracks()
        _draw_scc_graph(stars=True)
        _grab(intro_pause, tag="intro:graph")
        _base_axes(f"{title}\nCell Decomposition + Reeb Graph + Crack Routing")
        _draw_workspace(); _draw_cracks(); _draw_cells(); _draw_reeb(callouts=False)
        _draw_scc_graph(stars=True); _draw_scc_adds()
        _grab(decomp_pause, tag="decomp:1")

    def _scc_emit_pose(cur):
        _base_axes(f"{title}\nCoverage Sweep | pose {pose_n}")
        _draw_workspace(); _draw_cracks(); _draw_cells()
        if len(planned):                                   # remaining/plan: black dashed
            ax.plot(planned[:, 0] * M, planned[:, 1] * M, "--", color="0.1", lw=1.1, zorder=4)
            sj = min(4, len(planned) - 1)                  # start: direction-pointing arrowhead
            s0 = 2.0 * planned[0] - planned[sj]
            _arrowhead(ax, (s0[0] * M, s0[1] * M), (planned[0, 0] * M, planned[0, 1] * M),
                       "#0072BD", _HEAD_PX * M)
        if len(path_pts) >= 2:                             # covered so far: red solid
            p = np.asarray(path_pts, float)
            ax.plot(p[:, 0] * M, p[:, 1] * M, "-", color="#D62728", lw=1.8, zorder=5)
        _draw_robot(cur)
        _grab(tag=f"pose:{cur_it}:{pose_n}")

    def _emit_decomp():
        _base_axes(f"{title}\nIter {cur_it}: Cell Decomposition + Reeb Graph")
        _draw_workspace(); _draw_cracks(); _draw_cells()
        _draw_path(arrow=False, show_planned=False)     # completed path only at decomposition frame
        _draw_robot(cur_curPt if cur_it > 1 else start_cur)
        _draw_reeb(callouts=True)
        _grab(decomp_pause if publish else 1, tag=f"decomp:{cur_it}")

    def _emit_pose(cur):
        _base_axes(f"{title}\niter {cur_it} | pose {pose_n}")
        _draw_workspace(); _draw_cracks()
        if color_cells:
            _draw_cells()
        _draw_path(arrow=True); _draw_robot(cur)
        _grab(tag=f"pose:{cur_it}:{pose_n}")

    # ----- intro callout sequence (publish only), one element at a time -----
    def _intro(start_cur, scanned_idx):
        cb, rb = bounds
        sx, sy = float(start_cur[0]), float(start_cur[1])

        def _frame(callback, sub, tag):
            _base_axes(f"{title}\n{sub}")
            _draw_workspace(); _draw_cracks()
            callback()
            _grab(intro_pause, tag=tag)

        def _start_marker():
            ax.plot(sx * M, sy * M, "v", color="#0072BD", ms=12,
                    markerfacecolor="#0072BD", zorder=8)

        def _footprint():
            ax.add_patch(Circle((sx * M, sy * M), NOZZLE_R * math.sqrt(2) * M, fc="#F4A582",
                                ec="#C0392B", alpha=0.85, zorder=6))
            _start_marker()

        def _sensor():
            ax.add_patch(Circle((sx * M, sy * M), SENSOR_R * M, fc="#FCE9A6", ec="0.0",
                                alpha=0.30, lw=1.0, zorder=3))
            _footprint()

        # 1) Start Point: text label beside the marker, no arrow
        def _legend_start():
            _start_marker()
            _label_beside(ax, "Start Point", (sx * M, sy * M), 0.035 * cb * M)
        _frame(_legend_start, "legend: start point", "intro:start")

        # 2) Footprint Range: text beside the nozzle disk, no arrow
        def _legend_footprint():
            _footprint()
            _label_beside(ax, "Footprint Range", (sx * M, sy * M),
                          (NOZZLE_R * math.sqrt(2) + 60) * M)
        _frame(_legend_footprint, "legend: footprint range", "intro:footprint")

        # 3) Sensor Range: text beside the sensor disk, no arrow
        def _legend_sensor():
            _sensor()
            _label_beside(ax, "Sensor Range", (sx * M, sy * M), (SENSOR_R + 50) * M)
        _frame(_legend_sensor, "legend: sensor range", "intro:sensor")

        # 4) Scanned Cracks: detected cracks first appear here; arrow to a crack inside the sensor
        for i in scanned_idx:
            if i < len(det):
                det[i] = True
        def _legend_scanned():
            _sensor()
            if len(scanned_idx) and skel is not None:
                p = skel[scanned_idx[len(scanned_idx) // 2]]
                a = _free_anchor(p, None, bounds, avoid=[(sx, sy, SENSOR_R)],
                                 text="Scanned Cracks\n(in Sensor Range)", prox=0.5, cap_frac=0.05)
                _callout(ax, "Scanned Cracks\n(in Sensor Range)", (p[0] * M, p[1] * M),
                         (a[0] * M, a[1] * M))
        _frame(_legend_scanned, "legend: scanned cracks", "intro:scanned")

        # 5) Unknown Cracks: arrows to far unscanned cracks, text in free space
        def _legend_unknown():
            # Central label in the clear top strip with two arrows fanning to spread unknown cracks.
            _sensor()
            if skel is None or not len(skel):
                return
            txt = "Unknown Cracks\n(in the WorkSpace)"
            out = skel[np.hypot(skel[:, 0] - sx, skel[:, 1] - sy) > SENSOR_R]   # unknown
            if not len(out):
                out = skel
            # text anchor: a crack-free spot in the upper-centre region
            a = _free_anchor((0.55 * cb, 0.10 * rb), out, bounds, avoid=[(sx, sy, SENSOR_R)],
                             text=txt, prox=0.06, cap_frac=0.06)
            # two spread crack targets: nearest to the anchor + nearest one well away from it
            d = np.hypot(out[:, 0] - a[0], out[:, 1] - a[1])
            p1 = out[int(np.argmin(d))]
            spread = np.hypot(out[:, 0] - p1[0], out[:, 1] - p1[1]) > 0.25 * cb
            targets = [p1, out[np.flatnonzero(spread)[
                int(np.argmin(d[spread]))]]] if spread.any() else [p1]
            for p in targets:
                ax.annotate("", xy=(p[0] * M, p[1] * M), xytext=(a[0] * M, a[1] * M),
                            arrowprops=dict(arrowstyle="->", color=_ORANGE, lw=1.6,
                                            mutation_scale=18), zorder=9)
            ax.text(a[0] * M, a[1] * M, txt, fontsize=10, ha="left", va="center", zorder=9)
        _frame(_legend_unknown, "legend: unknown cracks", "intro:unknown")

    # ----- "Final Path" frame: full path + start/end -----
    def _final():
        _base_axes(None)                              # title goes inside the frame
        _draw_workspace()
        cb, rb = bounds
        ax.text(0.5 * cb * M, 0.035 * rb * M, "Final Path", fontsize=13, fontweight="bold",
                ha="center", va="top", zorder=10)
        if skel is not None:
            ax.plot(skel[:, 0] * M, skel[:, 1] * M, ".", color=_ORANGE, ms=0.5, zorder=1)
        if len(path_pts) >= 2:
            p = np.asarray(path_pts, float)
            ax.plot(p[:, 0] * M, p[:, 1] * M, "--", color="0.1", lw=1.2, zorder=4)
            # one arrowhead at the MIDDLE of each straight edge, pointing end-to-end
            # (simplify the dense path into edges so the direction is unambiguous)
            from shapely.geometry import LineString
            verts = np.asarray(LineString(p).simplify(40.0).coords)
            for v0, v1 in zip(verts[:-1], verts[1:]):
                if float(np.hypot(*(v1 - v0))) < 160.0:     # only label edges longer than ~0.3 m
                    continue
                mid = 0.5 * (v0 + v1)
                _arrowhead(ax, (v0[0] * M, v0[1] * M), (mid[0] * M, mid[1] * M),
                           "0.1", _HEAD_PX * 0.5 * M)
            # Start/end shown as direction-pointing arrowheads aligned with the robot's heading.
            sj = min(4, len(p) - 1)
            s0 = 2.0 * p[0] - p[sj]                        # virtual point behind the start
            _arrowhead(ax, (s0[0] * M, s0[1] * M), (p[0, 0] * M, p[0, 1] * M),
                       "#0072BD", _HEAD_PX * M)            # start: tip at p[0], initial heading
            ek = max(0, len(p) - 5)
            _arrowhead(ax, (p[ek, 0] * M, p[ek, 1] * M), (p[-1, 0] * M, p[-1, 1] * M),
                       "#77AC30", _HEAD_PX * M)            # end: tip at p[-1], final heading
            off = 0.025 * bounds[0] * M
            _label_beside(ax, "Start Point", (p[0, 0] * M, p[0, 1] * M), off, -off)
            _label_beside(ax, "End Point", (p[-1, 0] * M, p[-1, 1] * M), off, off)
        _grab(final_pause, tag="final")

    # ----- find the first pose (start position) and the cracks within sensor range -----
    start_cur, scanned_idx = None, []
    for e in viz:
        if e.get("kind") == "pose":
            start_cur = np.asarray(e["curPt"], float)
            break

    intro_done = False
    want_intro = remaining is None or any(t.startswith("intro:") for t in remaining)
    for e in viz:
        if stop_render:                                       # all wanted frames done
            break
        kind = e.get("kind")
        if kind == "init":
            sk = np.atleast_2d(np.asarray(e["skeleton"]))      # (row,col)
            skel = np.column_stack([sk[:, 1], sk[:, 0]]).astype(float) if len(sk) else None
            det = np.zeros(len(sk), dtype=bool) if len(sk) else np.zeros(0, bool)
            skel_idx = {(int(r), int(c)): k for k, (r, c) in enumerate(sk)}
            bounds = (int(e.get("colBW", 3050)), int(e.get("rowBW", 2898)))
            is_scc = bool(e.get("planner") == "SCC")           # offline SCC -> SCC intro/overlays
            if publish and skel is not None and start_cur is not None:
                scanned_idx = list(np.flatnonzero(
                    np.hypot(skel[:, 0] - start_cur[0], skel[:, 1] - start_cur[1]) <= SENSOR_R))
        elif kind == "iter":
            done_path.extend(cur_poses); cur_poses = []        # close out previous iteration
            cells = e.get("cells", [])
            critP = np.atleast_2d(e.get("critP", np.empty((0, 2))))
            reeb = e.get("reeb", [])
            reebwall = e.get("reebwall", [])
            reebCell = np.asarray(e.get("reebCell", np.empty(0, int)))
            wall_fol = list(e.get("wall_fol", []))
            Path = e.get("Path", [])
            cur_curPt = np.asarray(e.get("curPt", cur_curPt), float)
            planned = np.atleast_2d(np.asarray(e.get("raw_subXY", np.empty((0, 2))), float))[:, :2]
            cur_it = e.get("it", cur_it)
            last_pos = None                                    # no tweening across decomposition
            _sn = e.get("scc_crackNodes")
            scc_nodes = np.atleast_2d(np.asarray(_sn, float)) if (_sn is not None and np.size(_sn)) else np.empty((0, 2))
            _se = e.get("scc_crackEdges")
            scc_edges = np.atleast_2d(np.asarray(_se, int)) if (_se is not None and np.size(_se)) else np.empty((0, 2), int)
            _sa = e.get("scc_adds")
            scc_adds = np.asarray(_sa, float) if (_sa is not None and np.size(_sa)) else np.empty((0, 2, 2))
            if is_scc and publish:                             # offline SCC: build-up intro + decomp:1
                if not intro_done:
                    if not final_only:
                        _scc_intro()
                    intro_done = True
                    if stop_render:
                        break
            else:
                if publish and not intro_done:                 # show callouts before the first Reeb frame
                    if want_intro:
                        _intro(start_cur if start_cur is not None else cur_curPt, scanned_idx)
                    else:
                        for i in scanned_idx:                  # keep detection state without rendering
                            if i < len(det):
                                det[i] = True
                    intro_done = True
                    if stop_render:
                        break
                if _wanted(f"decomp:{cur_it}"):
                    _emit_decomp()
        else:                                                  # pose event
            s = e.get("sensed")
            if s is not None and len(s):
                for rc in s:
                    key = (int(rc[0]), int(rc[1]))
                    det_set.add(key)
                    k = skel_idx.get(key)              # mark crack as detected in the overlay
                    if k is not None:
                        det[k] = True
            cur = np.asarray(e["curPt"], float)
            if "subXY" in e:                          # planned path updated mid-iteration
                planned = np.atleast_2d(np.asarray(e["subXY"], float))[:, :2]
            path_pts.append((float(cur[0]), float(cur[1])))
            cur_poses.append((float(cur[0]), float(cur[1])))
            pose_n += 1
            # Render every step-th coverage pose, but every crack-fill pose so the nozzle
            # footprint reaching each crack endpoint is never skipped.
            if (pose_n % max(1, step) == 0 or e.get("fill")) and _wanted(f"pose:{cur_it}:{pose_n}"):
                # Optional smooth tweening: glide the robot disk between the last emitted pose
                # and this one at smooth_px spacing for natural motion.
                _pose_fn = _scc_emit_pose if is_scc else _emit_pose
                if smooth_px and remaining is None and last_pos is not None:
                    seg = cur - last_pos
                    nsub = int(float(np.hypot(*seg)) // smooth_px)
                    for j in range(1, nsub + 1):
                        _pose_fn(last_pos + seg * (j / (nsub + 1)))
                _pose_fn(cur)
                last_pos = np.asarray(cur, float).copy()
    if final_only:                                            # one "Final Path" PNG, no GIF
        if path_pts:
            _final()
        png = out_png or (out_gif if out_gif.endswith(".png")
                          else out_gif.rsplit(".", 1)[0] + ".png")
        os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return png, 1
    if publish and path_pts and _wanted("final"):
        _final()
    elif path_pts:                                             # plain: keep a last frame
        done_path.extend(cur_poses); planned = np.empty((0, 2))
        _emit_pose(path_pts[-1])
    plt.close(fig)
    if frames:
        imageio.mimsave(out_gif, frames, duration=1000.0 / fps, loop=0)
    return out_gif, len(frames)


def _downscale(buf, max_px):
    """Downscale an RGB frame so its larger side does not exceed ``max_px``.

    Applied only to GIF frames; full-resolution PNG frames are written before
    downscaling.  Returns the input buffer unchanged if it is already within
    bounds or if ``max_px`` is falsy.
    """
    if not max_px:
        return buf
    h, w = buf.shape[:2]
    big = max(h, w)
    if big <= max_px:
        return buf
    try:
        from PIL import Image
        scale = max_px / big
        return np.asarray(Image.fromarray(buf).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS))
    except Exception:
        return buf


def _safe_tag(tag):
    """Filesystem-safe stem for a frame tag (single-frame mode)."""
    return (tag or "frame").replace(":", "_")


def _sample_polyline(poly, step_px=45):
    """Densify a polyline (x,y) into points ~step_px apart (NaN-separated loops ok)."""
    poly = np.atleast_2d(np.asarray(poly, float))
    pts = []
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        if not (np.all(np.isfinite(a[:2])) and np.all(np.isfinite(b[:2]))):
            continue
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(1, int(d / step_px))
        for tt in np.linspace(0.0, 1.0, n + 1):
            pts.append(a[:2] + (b[:2] - a[:2]) * tt)
    return pts


def _free_anchor(target, obstacles, bounds, avoid=(), text="", prox=0.015, cap_frac=None):
    """Find a callout-text anchor position in open space near ``target``.

    Chooses the canvas position that maximises clearance from the drawn-element
    point cloud (``obstacles``) and the exclusion disks (``avoid``), with a mild
    bias toward proximity to ``target`` so arrows stay short.  The text-box
    footprint is estimated from the string length and number of lines.
    """
    cb, rb = bounds
    t = np.asarray(target, float)
    obs = np.asarray(obstacles, float) if (obstacles is not None and len(obstacles)) else None
    # text-box extents from the actual text (ha=left, va=center). Sizes are RELATIVE to
    # the workspace (char ~0.011*cb, half-line ~0.016*rb) so they adapt to maps of any
    # pixel scale (the figure size/font are fixed -> data-px per char scales with bounds).
    lines = (text or "").split("\n") or [""]
    tw = max(max((len(s) for s in lines), default=6) * 0.011 * cb, 0.06 * cb)
    th = max(len(lines) * 0.016 * rb, 0.03 * rb)
    foot = [(fx * tw, fy * th) for fx in (0.0, 0.25, 0.5, 0.75, 1.0) for fy in (-1.0, 0.0, 1.0)]
    w = prox                                              # proximity bias: small -> clearance dominates;
    #          large -> stays close to the feature (short arrow, fewer real obstacles)
    cap = cap_frac * max(cb, rb) if cap_frac else None     # cap clearance so proximity breaks ties

    def clearance(x, y):
        c = 1e18
        for dx, dy in foot:
            px, py = x + dx, y + dy
            if not (0.03 * cb <= px <= 0.99 * cb and 0.05 * rb <= py <= 0.97 * rb):
                return -1.0                              # text box would leave the canvas
            if obs is not None:
                c = min(c, float(np.min(np.hypot(obs[:, 0] - px, obs[:, 1] - py))))
            for ax_, ay_, r in avoid:
                c = min(c, float(np.hypot(px - ax_, py - ay_) - r))
        return c if cap is None else min(c, cap)

    best = None                                          # maximise clearance - w*distance
    for x in np.linspace(0.05, 0.90, 34) * cb:
        for y in np.linspace(0.05, 0.94, 34) * rb:
            c = clearance(x, y)
            if c <= 0:
                continue
            score = c - w * float(np.hypot(x - t[0], y - t[1]))
            if best is None or score > best[0]:
                best = (score, x, y)
    return np.array([best[1], best[2]]) if best else np.array([0.2 * cb, 0.12 * rb])


def _xy_loops(xy):
    """Split a NaN-separated boundary (x,y) into a list of loop arrays (>=3 pts)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    loops, cur = [], []
    for row in xy:
        if row.shape[0] < 2 or not np.all(np.isfinite(row[:2])):
            if len(cur) >= 3:
                loops.append(np.asarray(cur))
            cur = []
        else:
            cur.append(row[:2])
    if len(cur) >= 3:
        loops.append(np.asarray(cur))
    return loops


def _cells_to_polygon(cells):
    """Union of the cell polygons (outer loop + NaN-separated holes) as one shapely geom."""
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    polys = []
    for c in cells:
        loops = _xy_loops(c)
        if not loops:
            continue
        try:
            p = SPoly(loops[0], list(loops[1:]))
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _fill_geom(ax, geom, M, color, zorder):
    """Fill a shapely (Multi)Polygon (with holes) on ax, scaling coords by M."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    if geom is None or geom.is_empty:
        return
    geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for poly in geoms:
        if poly.is_empty:
            continue
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            pts = np.asarray(ring.coords, float)
            if len(pts) < 3:
                continue
            verts.extend(pts * M)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY])
        if verts:
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                                   edgecolor="none", zorder=zorder))


def _spline3(x, y, n=24):
    """Smooth curve through control points (Catmull-Rom-ish via parametric quadratic)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return x, y
    t = np.linspace(0, 1, len(x))
    tt = np.linspace(0, 1, n)
    try:
        from scipy.interpolate import CubicSpline
        return CubicSpline(t, x)(tt), CubicSpline(t, y)(tt)
    except Exception:
        return np.interp(tt, t, x), np.interp(tt, t, y)


def _arrowhead(ax, p0, p1, color, size):
    """Draw a solid, tail-less filled-triangle arrowhead at ``p1`` pointing from ``p0``.

    ``size`` is the arrowhead length in the axes' data units; ``p0`` and ``p1``
    must already be in those units.
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-9:
        return
    d /= L
    n = np.array([-d[1], d[0]])
    base = p1 - d * size
    tri = [p1, base + n * size * 0.5, base - n * size * 0.5]
    ax.add_patch(MplPolygon(tri, closed=True, facecolor=color, edgecolor=color, zorder=8))


def _callout(ax, text, point, textxy):
    """Draw a labelled arrow from ``textxy`` pointing to ``point``."""
    ax.annotate(text, xy=point, xytext=textxy, fontsize=10, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=_ORANGE, lw=1.6, mutation_scale=18),
                zorder=9)


def _label_beside(ax, text, marker_xy, dx, dy=0.0):
    """Place a plain text label beside a marker with no arrow."""
    ax.text(marker_xy[0] + dx, marker_xy[1] + dy, text, fontsize=10, ha="left",
            va="center", zorder=9)


class _LiveRenderer:
    """Lightweight on-screen renderer for the online planner's live mode.

    Fed one event at a time as the planner runs (``VizOptions(mode="live")``).
    Draws cell outlines, detected cracks, the planned path, and the robot disks
    at the current pose.  This is a progress monitor only; for a polished GIF
    use ``mode="save"`` instead.

    Requires an interactive matplotlib backend (TkAgg or QtAgg); degrades to a
    one-line warning and no-ops on headless environments.
    """

    def __init__(self, style="plain", title="OnlineSCC", every=1, pause=0.001, backend=None):
        self.style = style
        self.publish = style in ("publish", "color")
        self.color_cells = style == "color"
        self.title, self.every, self.pause = title, max(1, every), pause
        self.M = _M_PER_PX if self.publish else 1.0
        self.skel = self.det = None
        self.skel_idx = {}
        self.done, self.cur_poses = [], []               # completed-iters + current-iter poses
        self.planned = np.empty((0, 2))                  # current iter's full planned subXY
        self.cells, self.critP = [], np.empty((0, 2))
        self.bounds, self.cur_it, self.pose_n = (3050, 2898), 0, 0
        self.ok = True
        for be in ([backend] if backend else ["TkAgg", "QtAgg"]):
            try:
                plt.switch_backend(be)
                break
            except Exception:
                continue
        else:
            self.ok = False
            print("[viz] live mode needs a GUI backend (TkAgg/QtAgg) -- none available; "
                  "skipping the live window (use --viz save to write the GIF instead).", flush=True)
            return
        if not backend:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8.2, 7.9))

    def feed(self, e):
        if not self.ok:
            return
        k = e.get("kind")
        if k == "init":
            sk = np.atleast_2d(np.asarray(e["skeleton"]))
            self.skel = np.column_stack([sk[:, 1], sk[:, 0]]).astype(float) if len(sk) else None
            self.det = np.zeros(len(sk), bool) if len(sk) else np.zeros(0, bool)
            self.skel_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(sk)}
            self.bounds = (int(e.get("colBW", 3050)), int(e.get("rowBW", 2898)))
            self._redraw(None); self._pause(self.pause)
        elif k == "iter":
            self.done.extend(self.cur_poses); self.cur_poses = []   # close out previous iter
            self.planned = np.empty((0, 2))
            self.cells = e.get("cells", [])
            self.critP = np.atleast_2d(e.get("critP", np.empty((0, 2))))
            self.cur_it = e.get("it", self.cur_it)
            self._redraw(None); self._pause(0.4)
        else:
            s = e.get("sensed")
            if s is not None and len(s) and self.det is not None:
                for rc in s:
                    i = self.skel_idx.get((int(rc[0]), int(rc[1])))
                    if i is not None:
                        self.det[i] = True
            cur = np.asarray(e["curPt"], float)
            self.cur_poses.append((float(cur[0]), float(cur[1])))
            if "subXY" in e:                              # current iter's evolving full plan
                self.planned = np.atleast_2d(np.asarray(e["subXY"], float))[:, :2]
            self.pose_n += 1
            if self.pose_n % self.every == 0 or e.get("fill"):
                self._redraw(cur); self._pause(self.pause)

    def _pause(self, t):
        try:
            plt.pause(max(1e-4, t))
        except Exception:
            pass

    def _redraw(self, cur):
        ax, M, (cb, rb) = self.ax, self.M, self.bounds
        ax.clear()
        ax.set_xlim(0, cb * M); ax.set_ylim(rb * M, 0); ax.set_aspect("equal")
        ax.set_xlabel(f"x ({'m' if self.publish else 'px'})")
        ax.set_ylabel(f"y ({'m' if self.publish else 'px'})")
        ax.set_title(f"{self.title}\niter {self.cur_it} | pose {self.pose_n}", fontsize=10)
        if self.color_cells and self.cells:                  # cream complement fill + colour cells
            from shapely.geometry import box as _box
            u = _cells_to_polygon(self.cells)
            comp = _box(0, 0, cb, rb)
            comp = comp.difference(u) if u is not None else comp
            _fill_geom(ax, comp, M, _CREAM, 0.3)
            for kk, c in enumerate(self.cells):
                xy = np.atleast_2d(c); col = _PALETTE[kk % len(_PALETTE)]
                ax.fill(xy[:, 0] * M, xy[:, 1] * M, facecolor=col, alpha=0.18, zorder=1)
                ax.plot(xy[:, 0] * M, xy[:, 1] * M, "-", color=col, lw=1.4, zorder=2)
        elif self.publish and self.cells:
            for c in self.cells:
                xy = np.atleast_2d(c)
                ax.plot(xy[:, 0] * M, xy[:, 1] * M, "-", color="0.6", lw=1.0, zorder=2)
        if self.skel is not None:                            # cracks: light + detected thick
            ax.plot(self.skel[:, 0] * M, self.skel[:, 1] * M, ".", color="0.78", ms=0.4, zorder=1)
            if self.det is not None and self.det.any():
                ax.plot(self.skel[self.det, 0] * M, self.skel[self.det, 1] * M, ".",
                        color="0.0", ms=1.4, zorder=2)
        disp = list(self.done)                               # completed iterations + current planned path
        if len(self.planned):
            disp += [tuple(p) for p in self.planned]
        if disp:
            d = np.asarray(disp, float)
            ax.plot(d[:, 0] * M, d[:, 1] * M, "--", color=PATH_COLOR, lw=1.4, zorder=4)
            start_xy = self.done[0] if self.done else disp[0]
            ax.plot(start_xy[0] * M, start_xy[1] * M, "v", color="#0072BD", ms=10, zorder=7)
        if len(self.planned) >= 2:                           # direction arrowhead at path end
            tip = self.planned[-1]; prev = self.planned[max(0, len(self.planned) - 3)]
            _arrowhead(ax, (prev[0] * M, prev[1] * M), (tip[0] * M, tip[1] * M),
                       PATH_COLOR, _HEAD_PX * M)
        if cur is not None:                                  # robot: sensor + nozzle
            ax.add_patch(Circle((cur[0] * M, cur[1] * M), SENSOR_R * M, fc="#FCE9A6",
                                ec="0.0", alpha=0.30, zorder=3))
            ax.add_patch(Circle((cur[0] * M, cur[1] * M), NOZZLE_R * math.sqrt(2) * M,
                                fc="#F4A582", ec="#C0392B", alpha=0.85, zorder=6))

    def hold(self):
        """Block until the user closes the live window."""
        if not self.ok:
            return
        try:
            plt.ioff(); plt.show()
        except Exception:
            pass


class SCCLiveDebug:
    """Interactive per-stage debug figure for the offline SCC planner.

    Attach to a planner run via ``run_scc(viz=VizOptions(mode="live"))`` (or
    construct directly for testing).  The planner calls ``update(stage, **state)``
    at each pipeline stage; the figure accumulates state and redraws, so setting a
    debugger breakpoint during the run reveals the current planner stage visually.

    The figure progresses through the following stages in order:

    1. Workspace background and crack skeleton.
    2. Crack graph (visibility-graph edges + endpoint markers).
    3. Free-space cell decomposition and Reeb-graph critical points.
    4. Reeb graph edges.
    5. Chinese Postman matched links (added traversal edges).
    6. Final coverage path with start and end markers.

    Requires an interactive matplotlib backend (TkAgg or QtAgg).  On headless or
    non-interactive environments it degrades to a no-op and prints a one-line hint,
    so batch runs are unaffected.

    Parameters
    ----------
    title : str
        Figure title prefix.
    rowBW, colBW : int
        Height and width of the workspace in pixels (default 2896 × 3048).
    style : str
        ``"color"`` fills each Morse cell with a distinct colour; any other value
        draws outlines only.
    pause : float
        Seconds to dwell on each stage so it is visible before the next update.
    backend : str or None
        Matplotlib backend name to switch to.  ``None`` tries TkAgg then QtAgg.
    """

    def __init__(self, title="SCC", rowBW=2896, colBW=3048, style="color", pause=0.7, backend=None):
        self.title, self.rowBW, self.colBW = title, rowBW, colBW
        self.color_cells = style == "color"
        self.pause = pause                                   # dwell time per stage so it is visible
        self.state = {}
        self.ok = False
        # Try each GUI backend in turn. The default is often a non-interactive Agg that draws
        # nothing on screen. subplots() is inside the try so a backend that imports but can't
        # open a window (headless) falls through gracefully.
        for be in ([backend] if backend else ["TkAgg", "QtAgg"]):
            try:
                plt.switch_backend(be)
                if not backend:
                    plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(7.2, 7.0))
                self.ok = True
                break
            except Exception:
                continue
        if not self.ok:
            print("[viz] SCC live mode needs a GUI backend (TkAgg/QtAgg) -- none available; "
                  "skipping the live window (use --viz save to write the figure instead).", flush=True)

    def update(self, stage, clear=(), **new):
        """Accumulate planner state and redraw the figure for the given pipeline stage.

        State is persistent across calls; only keys listed in ``clear`` are removed
        before the new values are merged.  For example, the coverage-path stage passes
        ``clear=("reeb",)`` to drop the Reeb graph before drawing the final path.

        Parameters
        ----------
        stage : str
            Short label shown in the figure title, e.g. ``"Crack Graph"``.
        clear : iterable of str
            Keys to remove from the accumulated state before this update.
        **new
            Planner state to add or update.  Recognised keys include ``"skel"``
            (crack skeleton array), ``"cells"`` (list of cell boundary arrays),
            ``"node"`` / ``"crackEdge"`` (crack-graph nodes and edges),
            ``"reeb"`` (list of Reeb edge curves), ``"critP"`` (critical points),
            ``"adds"`` (Chinese Postman matched edges), and ``"path"``
            (final coverage path).
        """
        if not self.ok:
            return
        for k in clear:
            self.state.pop(k, None)
        self.state.update({k: v for k, v in new.items() if v is not None})
        s, ax = self.state, self.ax
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0, self.colBW); ax.set_ylim(self.rowBW, 0)     # image convention (y down)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.set_title(f"{self.title}  |  {stage}", fontsize=10)

        skel = s.get("skel")                                       # cracks (skeleton)
        if skel is not None:
            m, n = np.nonzero(np.asarray(skel))
            ax.plot(n, m, ".", color="0.72", ms=0.4, zorder=1)

        for k, c in enumerate(s.get("cells") or []):               # free-space cells (colour cycle)
            b = c.boundary() if hasattr(c, "boundary") else c
            b = np.atleast_2d(np.asarray(b, float))[:, :2]
            col = _PALETTE[k % len(_PALETTE)]
            if self.color_cells:
                ax.fill(b[:, 0], b[:, 1], facecolor=col, alpha=0.16, zorder=2)
            ax.plot(b[:, 0], b[:, 1], "-", color=col, lw=1.3, zorder=2)

        node, ce = s.get("node"), s.get("crackEdge")               # crack graph: green dashed edges + red stars
        if node is not None and ce is not None and np.size(ce):
            nx = np.atleast_2d(np.asarray(node, float))[:, ::-1]   # (row,col)->(x,y)
            for (u, v) in np.atleast_2d(ce).astype(int):
                ax.plot([nx[u, 0], nx[v, 0]], [nx[u, 1], nx[v, 1]], "--",
                        color="#2CA02C", lw=1.2, zorder=3)
            ax.plot(nx[:, 0], nx[:, 1], "*", color="r", ms=7, zorder=5)

        for e in (s.get("reeb") or []):                            # Reeb edges
            e = np.atleast_2d(np.asarray(e, float))
            if e.shape[1] >= 2:
                e = e[:, [1, 0]]                                   # (row,col) -> (x,y)
                ax.plot(e[:, 0], e[:, 1], "-", color="#7E2F8E", lw=1.4, zorder=3)

        cp = s.get("critP")                                        # critical points
        if cp is not None and np.size(cp):
            cp = np.atleast_2d(np.asarray(cp, float))[:, ::-1]     # (row,col) -> (x,y)
            ax.plot(cp[:, 0], cp[:, 1], "r.", ms=11, zorder=6)

        adds = s.get("adds")                                       # Chinese Postman matched edges (blue dashed)
        if adds is not None and len(adds):
            for seg in adds:
                seg = np.atleast_2d(np.asarray(seg, float))
                ax.plot(seg[:, 0], seg[:, 1], "--", color="#0072BD", lw=1.6, zorder=4)

        PE = s.get("path")                                         # coverage path
        if PE is not None and np.size(PE):
            PE = np.atleast_2d(np.asarray(PE, float))[:, :2]
            ax.plot(PE[:, 0], PE[:, 1], "--", color=PATH_COLOR, lw=1.1, zorder=7)
            ax.plot(PE[0, 0], PE[0, 1], "^", color="#0072BD", ms=11, zorder=8)
            ax.plot(PE[-1, 0], PE[-1, 1], "v", color="#77AC30", ms=11, zorder=8)

        try:
            self.fig.canvas.draw_idle(); plt.pause(max(1e-3, self.pause))
        except Exception:                                          # pragma: no cover
            pass

    def animate_robot(self, PathEdge, s, a, step=1, pose_pause=0.03):
        """Animate the robot traversing the planned path on the current figure.

        At each pose the sensor disk and nozzle footprint disk move to the robot's
        new position while the static scene underneath remains unchanged.

        Parameters
        ----------
        PathEdge : array-like, shape (N, 2)
            Ordered path vertices as (x, y) pairs.
        s : float
            Sensor radius in pixels.
        a : float
            Nozzle footprint radius in pixels; the drawn footprint has radius
            ``a * sqrt(2)`` (the inscribed-square diagonal).
        step : int
            Subsample factor; every ``step``-th pose is drawn.  Use values greater
            than 1 to speed up animation of long paths.
        pose_pause : float
            Seconds to pause between poses (default 0.03).
        """
        if not self.ok or PathEdge is None or not np.size(PathEdge):
            return
        PE = np.atleast_2d(np.asarray(PathEdge, float))[:, :2]
        arts = []
        for i in range(0, PE.shape[0], max(1, int(step))):
            for art in arts:
                try:
                    art.remove()
                except Exception:                                  # pragma: no cover
                    pass
            x, y = PE[i]
            sens = Circle((x, y), s, fc="#FCE9A6", ec="0.0", alpha=0.30, zorder=9)
            foot = Circle((x, y), a * math.sqrt(2), fc="#F4A582", ec="#C0392B", alpha=0.85, zorder=10)
            self.ax.add_patch(sens); self.ax.add_patch(foot)
            arts = [sens, foot]
            self.ax.set_title(f"{self.title}  |  robot pose {i + 1}/{PE.shape[0]}", fontsize=10)
            try:
                self.fig.canvas.draw_idle(); plt.pause(max(1e-3, pose_pause))
            except Exception:                                      # pragma: no cover
                pass

    def hold(self, seconds=3.0):
        """Keep the figure visible after the run completes.

        Parameters
        ----------
        seconds : float or None
            How long to display the figure before returning.  The calling
            program exits normally after this delay and the window closes with
            it.  Pass ``None`` to block indefinitely until the user closes the
            window manually.
        """
        if not self.ok:
            return
        try:
            if seconds is None:
                plt.ioff(); plt.show()                             # block until the user closes the window
            else:
                self.fig.canvas.draw_idle()
                plt.pause(max(0.1, float(seconds)))                # show briefly, then let it exit
                plt.close(self.fig)
        except Exception:                                          # pragma: no cover
            pass


# Standalone rendering is driven by make_gif.py -- there is no __main__ here;
# the renderer takes its data purely from the run_online_scc(_viz=...) event stream.
