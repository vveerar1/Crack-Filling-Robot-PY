"""Robot configuration loader for the SCC and OnlineSCC planners.

Reads ``robot_config.json`` (at the repo root) -- edit that file to change the
robot's dimensions; no code changes needed.

The JSON holds one entry per robot (``robot1``, ``robot2``, ...); both planners
use ``robot1`` today, and additional robots can be added for future multi-robot
work. Each robot entry specifies three diameters (inches):

  base_diameter_in       the mobile base
  footprint_diameter_in  the nozzle / fill footprint
  sensor_diameter_in     the 360-degree sensor

from which the working radii in pixels (1 px = 2 mm) are derived:

  r1  base radius        a  footprint radius        s  sensor radius

JSON is used so the same file can also be read by a future ROS2 params loader
(``jsondecode`` is the standard parser there).
"""
import json
import os
from dataclasses import dataclass

from private.utils import inpxMap

_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "robot_config.json")


@dataclass(frozen=True)
class RobotConfig:
    """Robot dimensions (inches) and the working radii (pixels) derived from them."""
    base_diameter_in: float        # mobile base
    footprint_diameter_in: float   # nozzle / fill footprint
    sensor_diameter_in: float      # 360-degree sensor

    @property
    def r1(self):
        """Base radius (px)."""
        return inpxMap(self.base_diameter_in / 2)

    @property
    def a(self):
        """Fill-footprint radius (px)."""
        return inpxMap(self.footprint_diameter_in / 2)

    @property
    def s(self):
        """Sensor radius (px)."""
        return inpxMap(self.sensor_diameter_in / 2)


def load(robot="robot1"):
    """Return the ``RobotConfig`` for ``robot`` ('robot1', ...) from robot_config.json."""
    with open(_JSON) as f:
        d = json.load(f)[robot]
    return RobotConfig(d["base_diameter_in"], d["footprint_diameter_in"], d["sensor_diameter_in"])


ROBOT1 = load("robot1")
