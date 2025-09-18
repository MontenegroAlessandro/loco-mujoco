# import mujoco
# from mujoco import MjSpec

# from .boostert1 import BoosterT1


# class MjxBoosterT1(BoosterT1):

#     mjx_enabled = True

#     def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
#         if "model_option_conf" not in kwargs.keys():
#             model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
#         else:
#             model_option_conf = kwargs["model_option_conf"]
#             del kwargs["model_option_conf"]
#         super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

#     def _modify_spec_for_mjx(self, spec: MjSpec):
#         """
#         Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
#         some changes to the XML:
#             1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
#                two capsules.
#             2. Disable all contacts except the ones between feet and the floor.

#         Args:
#             spec: Handle to Mujoco XML.

#         Returns:
#             Mujoco XML handle.

#         """

#         foot_geoms = ["right_foot_collision", "left_foot_collision"]

#         # --- Make all geoms have contype and conaffinity of 0 ---
#         for g in spec.geoms:
#             g.contype = 0
#             g.conaffinity = 0

#         # --- Define specific contact pairs ---
#         for g_name in foot_geoms:
#             spec.add_pair(geomname1="floor", geomname2=g_name)

#         return spec


import mujoco
from mujoco import MjSpec

from .boostert1 import BoosterT1


class MjxBoosterT1(BoosterT1):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except the ones between feet and the floor.

        Args:
            spec: Handle to Mujoco XML.

        Returns:
            Mujoco XML handle.

        """

        left_foot_geoms = ["left_foot_1_col", "left_foot_2_col"]
        right_foot_geoms = ["right_foot_1_col", "right_foot_2_col"]

        left_leg_geoms = ["left_knee_col", "left_hip_col"]
        right_leg_geoms = ["right_knee_col", "right_hip_col"]

        torso_geoms = ["torso_col", "waist_col"]
        head_geoms = ["head_col"]

        left_arm_geoms = ["left_forearm_col"]
        right_arm_geoms = ["right_forearm_col"]
        

        # --- Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # for computational efficiency, we will only let some geoms collide with others
        # we are not enabling collisions for safety enforcement right now, but so that:
        # 1. important interactions are not missed (e.g. against the floor)
        # 2. unrealistic dynamics are not learned (e.g. legs passing through each other)
        # --- Define specific contact pairs ---

        # first with the floor
        for g_name in left_foot_geoms + right_foot_geoms: # + left_leg_geoms + right_leg_geoms + torso_geoms + head_geoms + left_arm_geoms + right_arm_geoms:
            spec.add_pair(geomname1="floor", geomname2=g_name)


        # spec.add_pair(geomname1="left_foot_col", geomname2="right_foot_col")
        # spec.add_pair(geomname1="left_knee_col", geomname2="right_knee_col")
        # spec.add_pair(geomname1="left_hip_col", geomname2="right_hip_col")

        # # things the forearms can collide with
        # for g_name in head_geoms + ['torso_col'] + left_leg_geoms + right_leg_geoms:
        #     spec.add_pair(geomname1="left_forearm_col", geomname2=g_name)
        #     spec.add_pair(geomname1="right_forearm_col", geomname2=g_name)


        return spec