from .base import Reward
from .default import NoReward, TargetVelocityGoalReward, TargetXVelocityReward, LocomotionReward, \
    HumanoidLocomotionReward
from .trajectory_based import TargetVelocityTrajReward, MimicReward, CrispBoosterLocomotionReward
from .utils import *
from .foot_placement import CrispBoosterLocomotionFootPlacementReward

# register all rewards
NoReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
TargetVelocityTrajReward.register()
MimicReward.register()
LocomotionReward.register()
CrispBoosterLocomotionReward.register()
CrispBoosterLocomotionFootPlacementReward.register()
HumanoidLocomotionReward.register()