from .base import Reward
from .default import NoReward, TargetVelocityGoalReward, TargetXVelocityReward, LocomotionReward
from .trajectory_based import TargetVelocityTrajReward, MimicReward, CrispBoosterLocomotionReward
from .utils import *
from .foot_placement import FootPlacementReward, FootPlacementLocomotionReward, CrispBoosterLocomotionFootPlacementReward, FootPlacementTargetReward

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
CrispBoosterLocomotionFootPlacementReward.register()
FootPlacementTargetReward.register()