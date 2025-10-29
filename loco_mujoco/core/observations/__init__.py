from .base import Observation, ObservationIndexContainer, ObservationContainer, ObservationType, StatefulObservation
from .goals import *
from .goals_foot_placement import GoalRandomFootPlacement, GoalRandomChangingFootPlacement, GoalDoubleFootPlacement

# register all goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalTrajRootVelocity.register()
GoalTrajMimic.register()
GoalTrajMimicv2.register()
GoalChangingRandomRootVelocity.register()
GoalRandomFootPlacement.register()
GoalRandomChangingFootPlacement.register()
GoalDoubleFootPlacement.register()