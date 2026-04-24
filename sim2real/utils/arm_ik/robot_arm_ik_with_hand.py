import os
import sys

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import Quaternion, SE3
from pinocchio import casadi as cpin

from .robot_arm_ik import G1_29_ArmIK
from .weighted_moving_filter import WeightedMovingFilter

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)


def _resolve_asset_path(path):
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(parent2_dir, path)),
        os.path.abspath(os.path.join(parent2_dir, "..", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


class G1_29_WithHandArmIK(G1_29_ArmIK):  # noqa: N801
    """7-DoF arm IK for the G1 model with hands, with finger joints locked."""

    def __init__(self, Unit_Test=False, Visualization=False, robot_config=None):  # noqa: N803
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        asset_file = _resolve_asset_path(robot_config.get("ARM_IK_ASSET_FILE", robot_config["ASSET_FILE"]))
        asset_root = _resolve_asset_path(robot_config.get("ARM_IK_ASSET_ROOT", robot_config["ASSET_ROOT"]))
        self.robot = pin.RobotWrapper.BuildFromURDF(asset_file, asset_root)

        lower_body_joints = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        hand_joints = [str(name) for name in self.robot.model.names if "hand_" in str(name)]
        self.mixed_jointsToLockIDs = lower_body_joints + hand_joints

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.zeros(self.robot.model.nq),
        )

        left_offset = np.array(robot_config.get("left_ee_local_offset", [0.15, 0.0046, 0.0]), dtype=float)
        right_offset = np.array(robot_config.get("right_ee_local_offset", [0.15, -0.0046, 0.0]), dtype=float)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), left_offset),
                pin.FrameType.OP_FRAME,
            )
        )
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), right_offset),
                pin.FrameType.OP_FRAME,
            )
        )

        self.geom_model = pin.buildGeomFromUrdf(
            self.reduced_robot.model,
            asset_file,
            pin.GeometryType.COLLISION,
            package_dirs=asset_root,
        )

        self.geom_model.addAllCollisionPairs()
        adjacent_pairs = {(self.reduced_robot.model.parents[i], i) for i in range(1, self.reduced_robot.model.njoints)}
        filtered_pairs = []
        for cp in self.geom_model.collisionPairs:
            link1 = self.geom_model.geometryObjects[cp.first].parentJoint
            link2 = self.geom_model.geometryObjects[cp.second].parentJoint
            if (link1, link2) not in adjacent_pairs and (link2, link1) not in adjacent_pairs:
                filtered_pairs.append(cp)
        self.geom_model.collisionPairs[:] = filtered_pairs

        self.data = self.reduced_robot.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
        print("num collision pairs - with-hand:", len(self.geom_model.collisionPairs))
        print(f"With-hand reduced arm nq: {self.reduced_robot.model.nq}")

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        print(self.cq.shape)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost
        )
        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,
            "calc_lam_p": False,
        }
        self.opti.solver("ipopt", opts)

        self.current_L_tf = None
        self.current_R_tf = None
        self.current_L_orientation = None
        self.current_R_orientation = None
        self.speed_factor = 0.02

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(
            np.array([0.4, 0.3, 0.2, 0.1]),
            self.reduced_robot.model.nq,
        )
        self.vis = None

        if self.Visualization:
            raise NotImplementedError("Meshcat visualization is not wired for G1_29_WithHandArmIK yet.")

    def check_self_collision(self, q):
        return False


__all__ = ["G1_29_WithHandArmIK", "Quaternion", "SE3"]
