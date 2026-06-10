"""Produce the OnlineSCC visualization deliverable via the _viz / VizOptions hook.

Run the pure-Python autonomous planner with a VizOptions; it captures per-step
events and auto-finalizes per `mode`:
  pos            - print the robot position live, no files (fast).
  show           - render the GIF and open it; keep no files.
  save           - render + write per-step PNGs + GIF (headless).  [default]
  show+save      - render + write + open.
  live           - plot on screen as the planner runs (needs a GUI backend).

style: plain / publish (detailed) / color (detailed + colour cells). Saving a
detailed style -> hi-res. e.g. mode=save style=color.

Usage: python make_gif.py [img_n] [mode] [style] [step] [fps]
"""
import os
import sys
import time

os.environ.setdefault("OSCC_PREP_IDENTITY", "1")
import OnlineSCC as D
from .visualize import VizOptions

img_n = sys.argv[1] if len(sys.argv) > 1 else "myCrack8_100_1"
mode = sys.argv[2] if len(sys.argv) > 2 else "save"
style = sys.argv[3] if len(sys.argv) > 3 else "plain"
step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
fps = int(sys.argv[5]) if len(sys.argv) > 5 else 15

opts = VizOptions(
    mode=mode, style=style, step=step, fps=fps,
    out_gif=f"Results/GIF/{img_n}_unKnown.gif",
    frame_dir=f"Results/GIF/{img_n}_unKnown_frames",
    title=f"OnlineSCC: {img_n}",
)

t0 = time.time()
bw = D._preprocess(img_n)
PE, res = D.run_online_scc(img_n=img_n, bw_working=bw, _viz=opts)
n_iter = sum(1 for e in opts.events if e["kind"] == "iter")
n_pose = sum(1 for e in opts.events if e["kind"] == "pose")
print(f"done in {time.time()-t0:.0f}s: mode={mode}  {n_iter} iters, {n_pose} poses, "
      f"res={[round(float(x),3) for x in res]}")
if mode in ("save", "show+save"):
    print(f"-> {opts.out_gif}  +  {opts.frame_dir}/")
