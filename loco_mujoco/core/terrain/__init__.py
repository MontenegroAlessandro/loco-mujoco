from .base import Terrain
from .static import StaticTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain
from .stones_and_holes import StonesHolesTerrain
from .parkour import ParkourTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()
StonesHolesTerrain.register()
ParkourTerrain.register()
