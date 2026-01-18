from .base import ControlFunction
from .default import DefaultControl
from .pd import PDControl, PDControlGait

# register all control functions
DefaultControl.register()
PDControl.register()
PDControlGait.register()
