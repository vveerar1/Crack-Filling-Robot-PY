"""Resolve a crack-map name from CLI density / sigma / map-number flags.

Shared by the SCC and OnlineSCC command lines so both select maps the same way:

  --den N      crack density %, one of 35/45/50/65/80/90/95/100   (default 100)
  --sig N      Gaussian sigma, one of 5/10/20. If given -> a GAUSSIAN map; else UNIFORM.
  --map-num N  the map number:
                 Uniform  -> which of the 5 variants (1..5); random 1..5 if omitted.
                 Gaussian -> which Gaussian set folder (1..6); random 1..6 if omitted.

Built names:
  Uniform   ->  myCrack<k>_<den>_<mapN>                 (CrackMaps/Uniform/<name>.png)
  Gaussian  ->  Gaussian<folder>/myCrackGauss_s<sig>_<den>   (CrackMaps/<name>.mat)
where <k> is the 1-based density index. An explicit map NAME (positional arg) overrides
all of these.
"""
import os
import random
import re

DEN = [35, 45, 50, 65, 80, 90, 95, 100]   # crack density values (%)
SIG = [5, 10, 20]                          # Gaussian sigmas
UNIFORM_MAPS = (1, 5)                      # variant range
GAUSSIAN_FOLDERS = (1, 6)                  # complete Gaussian set folders


def _den_index(den):
    if den not in DEN:
        raise SystemExit(f"--den must be one of {DEN} (got {den})")
    return DEN.index(den) + 1


def resolve_map(name=None, den=100, sig=None, mapnum=None):
    """Return ``(img_n, dd, description)``.

    ``dd`` is the 1-based density index (for the result-row density label).
    ``description`` is a human-readable summary of the chosen map.
    Raises ``SystemExit`` with a clear message on a bad flag or a missing file.
    """
    if name:                                                  # explicit name overrides the flags
        m = re.match(r"myCrack(\d+)_", name.rsplit("/", 1)[-1])
        dd = int(m.group(1)) if (m and 1 <= int(m.group(1)) <= len(DEN)) else _den_index(den)
        return name, dd, name

    dd = _den_index(den)
    if sig is None:                                           # UNIFORM
        mapN = mapnum if mapnum is not None else random.randint(*UNIFORM_MAPS)
        img = f"myCrack{dd}_{den}_{mapN}"
        path = os.path.join("CrackMaps", "Uniform", img + ".png")
        desc = f"Uniform {img}  (density {den}%, map {mapN})"
    else:                                                     # GAUSSIAN
        if sig not in SIG:
            raise SystemExit(f"--sig must be one of {SIG} (got {sig})")
        folder = mapnum if mapnum is not None else random.randint(*GAUSSIAN_FOLDERS)
        img = f"Gaussian{folder}/myCrackGauss_s{sig}_{den}"
        path = os.path.join("CrackMaps", img.replace("/", os.sep) + ".mat")
        desc = f"Gaussian folder {folder}, sigma {sig}, density {den}%  ({img})"

    if not os.path.exists(path):
        raise SystemExit(f"map not found: {path}\n  (resolved from den={den}, sig={sig}, "
                         f"map-num={mapnum})")
    return img, dd, desc
