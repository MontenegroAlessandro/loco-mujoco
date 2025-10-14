from .base import Reward
from .default import NoReward, TargetVelocityGoalReward, TargetXVelocityReward, LocomotionReward,FootPlacementReward,FootPlacementLocomotionReward
from .trajectory_based import TargetVelocityTrajReward, MimicReward, CrispBoosterLocomotionReward
from .utils import *

# register all rewards
NoReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
TargetVelocityTrajReward.register()
MimicReward.register()
LocomotionReward.register()
CrispBoosterLocomotionReward.register()
FootPlacementReward.register()
FootPlacementLocomotionReward.register()
