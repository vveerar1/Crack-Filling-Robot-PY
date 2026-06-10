# Crack-Filling Robot — Python

Path-planning and simulation for an autonomous **field robot that scans a surface for complete
coverage and fills every crack it finds**. The robot carries a 360° sensor for crack detection and
a nozzle mounted on an XY gantry within its footprint for dispensing filler.

Two planners are provided:

- **SCC (Sensor-based Complete Coverage)** — *offline*. The environment and crack locations are
  known in advance; the planner computes a low-cost coverage-and-filling path and the robot
  executes it.
- **OnlineSCC** — *online*. Crack locations are **unknown**. The robot scans the area in a
  boustrophedon (zig-zag) pattern, and whenever the sensor detects a crack it plans and fills it
  before resuming the scan — repeating until the whole area is covered.

Work areas are supplied as binary **crack maps** (PNG images), where **each pixel corresponds to a
robot pose**.

---

## How it works

Both planners share a coverage pipeline and differ only in how cracks are discovered and filled:

```
crack map (binary image)
  │
  ├─ Morse Cell Decomposition (MCD)   split the free space into monotone cells at critical points
  ├─ Reeb graph                       one node per critical point, one edge per cell
  ├─ Reeb path                        order the cells for efficient, low-overlap traversal
  └─ Boustrophedon path               generate the zig-zag coverage path at sensor spacing
                                      (with cell-connection for disjoint regions)
```

**Crack filling.**
- In **SCC**, the full crack network is known, so a crack graph is extracted up front and a
  **modified Chinese Postman** routine (Eulerian routing, with integer-programming matching of
  odd-degree nodes when needed) produces the shortest closed filling route.
- In **OnlineSCC**, the robot senses cracks within its sensor radius as it scans, extracts the
  crack skeleton and its endpoints/branch points on the fly, and plans crack-filling waypoints
  using a visibility graph and Chinese Postman routing. Covered area is removed from the free
  space, and the remaining region is re-decomposed until coverage is complete.

The planners report coverage path length, area covered, overlap, and run time.

---

## Requirements

- **Python 3.10+** (developed on 3.13).
- Python packages (installed via `requirements.txt`):
  - `numpy`, `scipy` — numerics and the integer-programming solver (`scipy.optimize.milp`)
  - `shapely`, `pyclipper` — polygon geometry (cells, buffers, boolean operations, offset clipping)
  - `networkx` — graph construction and shortest paths
  - `scikit-image` — image skeletonization and morphology
  - `matplotlib` — figures and animation frames
  - `imageio` — animated GIF assembly

---

## Getting started

1. **Clone** the repository:
   ```bash
   git clone https://github.com/vveerar1/Crack-Filling-Robot-PY.git
   cd Crack-Filling-Robot-PY
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
3. **Run a planner:**
   ```bash
   python OnlineSCC.py    # online — unknown cracks
   python SCC.py          # offline — known cracks
   ```
   With no arguments a planner runs headless on the default map (`myCrack8_100_1`), printing
   progress and a results row. See **Usage** below for everything else.

---

## Usage

Both planners share the same command-line interface. Print the full, authoritative option list at
any time with `--help`:

```bash
python OnlineSCC.py --help
python SCC.py --help
```

### Choosing a crack map

Maps live in `CrackMaps/`: uniform maps are named `myCrack<index>_<density>_<sample>.png`, Gaussian
maps `myCrackGauss_s<sigma>_<density>`. Select one in either of two ways:

- **By full name** (positional argument) — this overrides the selector flags below:
  ```bash
  python SCC.py myCrack8_100_1
  python OnlineSCC.py Gaussian6/myCrackGauss_s20_100
  ```
- **By selector flags:**

  | Flag | Meaning |
  |---|---|
  | `--den PCT` | crack density — one of `35 45 50 65 80 90 95 100` (default `100`) |
  | `--sig S` | Gaussian sigma `5` / `10` / `20`. **Given → a Gaussian map; omitted → a uniform map.** |
  | `--map-num N` | uniform: variant `1–5`; Gaussian: source folder `1–6`. Random if omitted. |

  ```bash
  python SCC.py --den 65 --map-num 3        # uniform, 65 % density, sample 3
  python OnlineSCC.py --sig 20 --den 100    # Gaussian, sigma 20, 100 % density
  ```

### Visualization

By default a run is **headless** — it prints progress and the results row and writes a single
final-path PNG. Add `--viz` to render the robot scanning and filling:

| Flag | Values | Meaning |
|---|---|---|
| `--viz` | `save` · `show` · `show+save` · `live` | `save` writes a GIF + frames; `show` opens a viewer at the end; `live` plots on screen as it runs (needs a GUI backend). |
| `--style` | `plain` · `publish` · `color` | render richness — `publish`/`color` add the Reeb graph, labels and coloured cells, and render hi-res. |

Both planners accept the same animation controls (used by the `save` modes): `--step` (poses per
frame), `--fps`, `--smooth M` (tweened robot motion, in metres), `--gif-max-px` (cap the GIF's larger
side), and `--frames DIR` (also dump per-step PNGs).

```bash
python SCC.py       myCrack8_100_1 --viz save --style color --smooth 0.07
python OnlineSCC.py myCrack8_100_1 --viz save --style color --smooth 0.07
```

You can also render a GIF for any map as a standalone step:

```bash
python -m private.make_gif myCrack8_100_1 save color    # [map] [mode] [style] [step] [fps]
```

### Other options

| Flag | Planner | Meaning |
|---|---|---|
| `-q`, `--quiet` | both | silence the console log (the log file is still written) |
| `--route MODE` | SCC | coverage-route formulation, default `rpp`; run `--help` for the alternatives |
| `--max-iter N` | OnlineSCC | cap the number of outer iterations (debugging) |
| `-o`, `--out` | both | explicit output path (GIF for `--viz save`; PNG otherwise) |

### Outputs and logs

| What | Where |
|---|---|
| Final-path PNG (**always** written) | `Results/SCC/` (SCC) · `Results/OnlineSCC/` (OnlineSCC) |
| Animated GIF + per-step frames (`--viz save`) | `Results/GIF/` |
| Run log (**always** written; console too, unless `-q`) | `logs/<planner>_<map>.log` (rotated) |

Each run also prints a results row:

```
res = [ num_iterations , density , runtime_s , path_length_ft , area_covered_ft2 ]
```

along with coverage and overlap percentages.

### Tests

End-to-end smoke tests run both planners on a bundled crack map and check the resulting coverage:

```bash
pip install pytest
pytest -m "not slow"   # quick: the offline SCC planner (~10 s)
pytest                 # full: also the online OnlineSCC loop (a few minutes)
```

---

## Notes

- Distances are computed in image pixels; `1 px ≈ 2 mm`. The robot footprint, sensor range, and
  base size are configured in `robot_config.json` (in inches, converted to pixels) and read by
  both planners.

---

## Author

Vishnu Veeraraghavan — Automated Control Systems and Robotics Lab.
For questions, contact `vveerar1@binghamton.edu`.
