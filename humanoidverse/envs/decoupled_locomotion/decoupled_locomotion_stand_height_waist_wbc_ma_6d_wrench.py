from isaacgym.torch_utils import *

import torch
from loguru import logger

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma import (
    LeggedRobotDecoupledLocomotionStanceHeightWBC,
)


DEBUG = False


class LeggedRobotDecoupledLocomotionStanceHeightWBC6DWrench(
    LeggedRobotDecoupledLocomotionStanceHeightWBC
):
    """Decoupled WBC tracking task with independent 6D end-effector wrench disturbance.

    This class intentionally leaves the legacy force-only task untouched.  The actor
    observation stays unchanged; force/torque are exposed only through critic
    privileged observations.
    """

    WRENCH_MODES = {
        "none": 0,
        "force_only": 1,
        "torque_only": 2,
        "force_torque": 3,
    }

    SAMPLE_FRAMES = {"hand_local", "base", "env"}

    def __init__(self, config, device):
        self.init_done = False
        self.wrench_cfg = config.wrench
        self.wrench_enabled = bool(self.wrench_cfg.get("enabled", True))
        self.wrench_mode = str(self.wrench_cfg.get("mode", "force_torque"))
        self.sample_frame = str(self.wrench_cfg.get("sample_frame", "hand_local"))
        if self.wrench_mode not in self.WRENCH_MODES:
            raise ValueError(f"Unsupported wrench.mode={self.wrench_mode}. Expected one of {list(self.WRENCH_MODES)}")
        if self.sample_frame not in self.SAMPLE_FRAMES:
            raise ValueError(f"Unsupported wrench.sample_frame={self.sample_frame}. Expected one of {list(self.SAMPLE_FRAMES)}")
        super().__init__(config, device)

        self.wrench_cfg = self.config.wrench
        self.wrench_enabled = bool(self.wrench_cfg.get("enabled", True))
        self.wrench_mode = str(self.wrench_cfg.get("mode", "force_torque"))
        self.sample_frame = str(self.wrench_cfg.get("sample_frame", "hand_local"))

        self.left_hand_link = config.robot.force_control.left_hand_link
        self.right_hand_link = config.robot.force_control.right_hand_link
        self.left_hand_link_index = self.body_names.index(self.left_hand_link)
        self.right_hand_link_index = self.body_names.index(self.right_hand_link)
        logger.info(f"6D wrench left link: {self.left_hand_link}, index: {self.left_hand_link_index}")
        logger.info(f"6D wrench right link: {self.right_hand_link}, index: {self.right_hand_link_index}")

        apply_links = list(self.wrench_cfg.get("apply_links", [self.left_hand_link, self.right_hand_link]))
        expected_links = [self.left_hand_link, self.right_hand_link]
        if apply_links != expected_links:
            raise ValueError(
                f"First 6D wrench training path expects apply_links={expected_links}, got {apply_links}"
            )

        self.left_ankle_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.left_ankle_dof_names]
        self.right_ankle_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.right_ankle_dof_names]

        self.left_wrench_limit_dof_indices = self.left_arm_dof_indices + self.waist_dof_indices
        self.right_wrench_limit_dof_indices = self.right_arm_dof_indices + self.waist_dof_indices

        if self.config.rewards.get("upper_body_motion_scale_curriculum", False):
            initial = self.config.rewards.upper_body_motion_initial_scale
            self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * initial
        else:
            self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)

        if self.config.rewards.get("command_height_scale_curriculum", False):
            initial = self.config.rewards.command_height_scale_initial_scale
            self.command_height_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * initial
        else:
            self.command_height_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)

        force_scale_initial = float(self.wrench_cfg.get("force_scale_initial", 0.1))
        torque_scale_initial = float(self.wrench_cfg.get("torque_scale_initial", 0.05))
        if str(self.wrench_cfg.get("curriculum_mode", "adaptive")) == "fixed":
            force_scale_initial = float(self.wrench_cfg.get("fixed_force_scale", force_scale_initial))
            torque_scale_initial = float(self.wrench_cfg.get("fixed_torque_scale", torque_scale_initial))
        self.apply_force_scale = (
            torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * force_scale_initial
        )
        self.apply_torque_scale = (
            torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * torque_scale_initial
        )
        self.effective_torque_scale = torch.zeros_like(self.apply_torque_scale)

        self.upper_body_tracking_sigma = (
            torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
            * (
                self.config.rewards.upper_body_tracking_sigma_initial_scale
                if self.config.rewards.get("upper_body_tracking_sigma_curriculum", False)
                else self.config.rewards.reward_tracking_sigma.upper_body_dofs
            )
        )
        self.upper_body_dofs_tracking_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.joint_effort_limit_scale = self.config.robot.get("dof_effort_limit_scale", 1.0)
        self.joint_effort_limit = (
            torch.tensor(self.config.robot.dof_effort_limit_list[:], device=self.device, requires_grad=False)
            * self.joint_effort_limit_scale
        )

    def _init_buffers(self):
        super()._init_buffers()
        self._init_wrench_settings()

        self.walking_env_mask = self.commands[:, 4] == 1
        zero_cmd_mask = torch.all(self.commands[:, :2] == 0, dim=-1)
        self.tapping_env_mask = self.walking_env_mask & zero_cmd_mask
        self.num_tapping_env = torch.sum(self.tapping_env_mask).item()
        random_choices = torch.randint(0, 2, (self.num_envs, 1), device=self.device)
        self.random_tapping_dir = torch.where(
            random_choices == 0,
            torch.tensor([1, 0], device=self.device),
            torch.tensor([-1, 0], device=self.device),
        )

    def _init_wrench_settings(self):
        self.left_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ee_apply_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_apply_torque = torch.zeros((self.num_envs, 3), device=self.device)

        self.apply_force_tensor = torch.zeros(
            self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.apply_force_pos_tensor = torch.zeros_like(self.apply_force_tensor)
        self.apply_torque_tensor = torch.zeros_like(self.apply_force_tensor)

        self.force_range_low, self.force_range_high = self._make_xyz_range(self.wrench_cfg.force_range_n)
        self.torque_range_low, self.torque_range_high = self._make_xyz_range(self.wrench_cfg.torque_range_nm)

        self.zero_force_prob = self._make_prob_tensor(self.wrench_cfg.get("zero_force_prob", [0.25, 0.25, 0.25]))
        self.zero_torque_prob = self._make_prob_tensor(self.wrench_cfg.get("zero_torque_prob", [0.5, 0.5, 0.5]))

        self.left_target_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_target_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_target_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_target_torque = torch.zeros((self.num_envs, 3), device=self.device)

        self.filtered_left_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_right_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_left_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_right_torque = torch.zeros((self.num_envs, 3), device=self.device)

        duration_range = self.wrench_cfg.get("resample_duration_steps", [150, 250])
        self.wrench_resample_duration_range = [int(duration_range[0]), int(duration_range[1])]
        self.wrench_resample_duration = torch.randint(
            self.wrench_resample_duration_range[0],
            self.wrench_resample_duration_range[1] + 1,
            (self.num_envs, 1),
            device=self.device,
        )
        self.wrench_resample_step = self.wrench_resample_duration.clone()

        self.use_final_wrench_lpf = bool(self.wrench_cfg.get("use_final_wrench_lpf", True))
        self.lpf_alpha = float(self.wrench_cfg.get("lpf_alpha", 0.05))
        self.torque_limit_aware_bound = bool(self.wrench_cfg.get("torque_limit_aware_bound", True))
        self.joint_limit_margin = float(self.wrench_cfg.get("joint_limit_margin", 0.8))
        self.apply_in_physics_step = bool(self.wrench_cfg.get("apply_in_physics_step", True))

        self.wrench_scale_down = torch.ones((self.num_envs, 2), dtype=torch.float, device=self.device)
        self._resample_wrench_targets(torch.arange(self.num_envs, device=self.device))
        self._zero_wrench_if_disabled()

    def _make_xyz_range(self, cfg_range):
        low = torch.tensor(
            [float(cfg_range.x[0]), float(cfg_range.y[0]), float(cfg_range.z[0])],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        high = torch.tensor(
            [float(cfg_range.x[1]), float(cfg_range.y[1]), float(cfg_range.z[1])],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        return low, high

    def _make_prob_tensor(self, prob):
        if isinstance(prob, float) or isinstance(prob, int):
            prob = [float(prob)] * 3
        return torch.tensor(prob, dtype=torch.float, device=self.device).view(1, 3)

    def _force_mode_enabled(self):
        return self.wrench_enabled and self.wrench_mode in ("force_only", "force_torque")

    def _torque_mode_enabled(self):
        return self.wrench_enabled and self.wrench_mode in ("torque_only", "force_torque")

    def _zero_wrench_if_disabled(self):
        if not self.wrench_enabled or self.wrench_mode == "none":
            self.left_target_force.zero_()
            self.right_target_force.zero_()
            self.left_target_torque.zero_()
            self.right_target_torque.zero_()
        elif self.wrench_mode == "force_only":
            self.left_target_torque.zero_()
            self.right_target_torque.zero_()
        elif self.wrench_mode == "torque_only":
            self.left_target_force.zero_()
            self.right_target_force.zero_()

    def _sample_vec(self, low, high, zero_prob, env_ids):
        values = low[env_ids] + (high[env_ids] - low[env_ids]) * torch.rand((len(env_ids), 3), device=self.device)
        zero_mask = torch.rand((len(env_ids), 3), device=self.device) < zero_prob
        return values * (~zero_mask).float()

    def _resample_wrench_targets(self, env_ids):
        if len(env_ids) == 0:
            return
        self.left_target_force[env_ids] = self._sample_vec(
            self.force_range_low, self.force_range_high, self.zero_force_prob, env_ids
        )
        self.right_target_force[env_ids] = self._sample_vec(
            self.force_range_low, self.force_range_high, self.zero_force_prob, env_ids
        )
        self.left_target_torque[env_ids] = self._sample_vec(
            self.torque_range_low, self.torque_range_high, self.zero_torque_prob, env_ids
        )
        self.right_target_torque[env_ids] = self._sample_vec(
            self.torque_range_low, self.torque_range_high, self.zero_torque_prob, env_ids
        )
        self.wrench_resample_duration[env_ids] = torch.randint(
            self.wrench_resample_duration_range[0],
            self.wrench_resample_duration_range[1] + 1,
            (len(env_ids), 1),
            device=self.device,
        )
        self.wrench_resample_step[env_ids] = 0
        self._zero_wrench_if_disabled()

    def _maybe_resample_wrench_targets(self):
        if not bool(self.wrench_cfg.get("piecewise_constant", True)):
            self._resample_wrench_targets(torch.arange(self.num_envs, device=self.device))
            return
        resample_ids = torch.where(self.wrench_resample_step.squeeze(-1) >= self.wrench_resample_duration.squeeze(-1))[0]
        self._resample_wrench_targets(resample_ids)
        self.wrench_resample_step += 1

    def _target_to_env_frame(self, target, link_index):
        if self.sample_frame == "env":
            return target
        if self.sample_frame == "base":
            return quat_rotate(self.base_quat, target)
        link_quat = self.simulator._rigid_body_rot[:, link_index, :]
        return quat_rotate(link_quat, target)

    def _update_effective_scales(self):
        if self._force_mode_enabled():
            force_scale = self.apply_force_scale
        else:
            force_scale = torch.zeros_like(self.apply_force_scale)

        if self._torque_mode_enabled():
            start_force_scale = float(self.wrench_cfg.get("torque_scale_start_after_force_scale", 0.0))
            torque_gate = (self.apply_force_scale >= start_force_scale).float()
            torque_scale = self.apply_torque_scale * torque_gate
        else:
            torque_scale = torch.zeros_like(self.apply_torque_scale)

        self.effective_torque_scale = torque_scale
        return force_scale, torque_scale

    def _calculate_ee_wrenches(self):
        self._maybe_resample_wrench_targets()
        force_scale, torque_scale = self._update_effective_scales()
        force_enabled = self._force_mode_enabled()
        torque_enabled = self._torque_mode_enabled()

        left_force = self._target_to_env_frame(self.left_target_force, self.left_hand_link_index) * force_scale
        right_force = self._target_to_env_frame(self.right_target_force, self.right_hand_link_index) * force_scale
        left_torque = self._target_to_env_frame(self.left_target_torque, self.left_hand_link_index) * torque_scale
        right_torque = self._target_to_env_frame(self.right_target_torque, self.right_hand_link_index) * torque_scale

        if not force_enabled:
            left_force.zero_()
            right_force.zero_()
            self.filtered_left_force.zero_()
            self.filtered_right_force.zero_()
        if not torque_enabled:
            left_torque.zero_()
            right_torque.zero_()
            self.filtered_left_torque.zero_()
            self.filtered_right_torque.zero_()

        if self.use_final_wrench_lpf:
            left_force = self.lpf_alpha * left_force + (1.0 - self.lpf_alpha) * self.filtered_left_force
            right_force = self.lpf_alpha * right_force + (1.0 - self.lpf_alpha) * self.filtered_right_force
            left_torque = self.lpf_alpha * left_torque + (1.0 - self.lpf_alpha) * self.filtered_left_torque
            right_torque = self.lpf_alpha * right_torque + (1.0 - self.lpf_alpha) * self.filtered_right_torque

        if not force_enabled:
            left_force.zero_()
            right_force.zero_()
        if not torque_enabled:
            left_torque.zero_()
            right_torque.zero_()

        left_force, left_torque, left_scale = self._limit_wrench_by_joint_margin(
            left_force,
            left_torque,
            self.left_hand_link_index,
            self.left_wrench_limit_dof_indices,
        )
        right_force, right_torque, right_scale = self._limit_wrench_by_joint_margin(
            right_force,
            right_torque,
            self.right_hand_link_index,
            self.right_wrench_limit_dof_indices,
        )
        self.wrench_scale_down[:, 0] = left_scale
        self.wrench_scale_down[:, 1] = right_scale

        self.filtered_left_force[:] = left_force
        self.filtered_right_force[:] = right_force
        self.filtered_left_torque[:] = left_torque
        self.filtered_right_torque[:] = right_torque

        if not self.apply_in_physics_step:
            left_force.zero_()
            right_force.zero_()
            left_torque.zero_()
            right_torque.zero_()

        self.apply_force_tensor.zero_()
        self.apply_torque_tensor.zero_()
        self.apply_force_pos_tensor[:, self.left_hand_link_index, :] = self.simulator._rigid_body_pos[
            :, self.left_hand_link_index, :
        ]
        self.apply_force_pos_tensor[:, self.right_hand_link_index, :] = self.simulator._rigid_body_pos[
            :, self.right_hand_link_index, :
        ]
        self.apply_force_tensor[:, self.left_hand_link_index, :] = left_force
        self.apply_force_tensor[:, self.right_hand_link_index, :] = right_force
        self.apply_torque_tensor[:, self.left_hand_link_index, :] = left_torque
        self.apply_torque_tensor[:, self.right_hand_link_index, :] = right_torque

        self.left_ee_apply_force = quat_rotate_inverse(self.base_quat, left_force.clone())
        self.right_ee_apply_force = quat_rotate_inverse(self.base_quat, right_force.clone())
        self.left_ee_apply_torque = quat_rotate_inverse(self.base_quat, left_torque.clone())
        self.right_ee_apply_torque = quat_rotate_inverse(self.base_quat, right_torque.clone())

    def _limit_wrench_by_joint_margin(self, force, torque, link_index, dof_indices):
        if not self.torque_limit_aware_bound:
            return force, torque, torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        jacobian = self.simulator.jacobian[:, link_index, :, 6:]
        jv = jacobian[:, :3, dof_indices]
        jw = jacobian[:, 3:6, dof_indices]
        tau_ext = (
            torch.bmm(jv.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(jw.transpose(1, 2), torque.unsqueeze(-1)).squeeze(-1)
        )

        # Gravity compensation torque is not available here, so the first version
        # uses a conservative fraction of configured joint effort limits.
        margin = self.joint_effort_limit[dof_indices].view(1, -1) * self.joint_limit_margin
        joint_scale = margin / (torch.abs(tau_ext) + 1.0e-6)
        scale = torch.clamp(torch.min(joint_scale, dim=1).values, max=1.0)
        return force * scale.unsqueeze(-1), torque * scale.unsqueeze(-1), scale

    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)

        if self.apply_in_physics_step:
            if self._force_mode_enabled():
                self.simulator.apply_rigid_body_force_at_pos_tensor(
                    self.apply_force_tensor,
                    self.apply_force_pos_tensor,
                )
            if self._torque_mode_enabled():
                self.simulator.apply_rigid_body_torque_tensor(self.apply_torque_tensor)

        self.simulator.apply_torques_at_dof(self.torques)

    def _physics_step(self):
        self.render()
        self._calculate_ee_wrenches()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _reset_filtered_wrenches(self, env_ids):
        self.filtered_left_force[env_ids] = 0.0
        self.filtered_right_force[env_ids] = 0.0
        self.filtered_left_torque[env_ids] = 0.0
        self.filtered_right_torque[env_ids] = 0.0
        self.left_ee_apply_force[env_ids] = 0.0
        self.right_ee_apply_force[env_ids] = 0.0
        self.left_ee_apply_torque[env_ids] = 0.0
        self.right_ee_apply_torque[env_ids] = 0.0

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self._resample_wrench_targets(env_ids)

        self.walking_env_mask = self.commands[:, 4] == 1
        zero_cmd_mask = torch.all(self.commands[:, :2] == 0, dim=-1)
        self.tapping_env_mask = self.walking_env_mask & zero_cmd_mask
        self.num_tapping_env = torch.sum(self.tapping_env_mask).item()
        tapping_envs_to_resample = self.tapping_env_mask[env_ids]
        tapping_indices = env_ids[tapping_envs_to_resample]
        random_choices = torch.randint(0, 2, (tapping_indices.shape[0], 1), device=self.device)
        self.random_tapping_dir[tapping_indices] = torch.where(
            random_choices == 0,
            torch.tensor([1, 0], device=self.device),
            torch.tensor([-1, 0], device=self.device),
        )

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True

        if self.config.rewards.upper_body_motion_scale_curriculum:
            self._update_upper_body_motion_scale_curriculum(env_ids)
        else:
            self.action_scale_upper_body[env_ids] = 1.0

        self._update_wrench_scale_curriculum(env_ids)

        if self.config.rewards.get("command_height_scale_curriculum", False):
            self._update_command_height_curriculum(env_ids)
        else:
            self.command_height_scale[env_ids] = 1.0

        if self.config.rewards.get("upper_body_tracking_sigma_curriculum", False):
            self._update_upper_body_tracking_sigma_curriculum(env_ids)

        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)
        self._reset_robot_states_callback(env_ids, target_states)

        self.upper_body_dofs_tracking_reward[env_ids] *= 0.0
        self._reset_filtered_wrenches(env_ids)
        self._resample_wrench_targets(env_ids)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf

    def _update_wrench_scale_curriculum(self, env_ids):
        if len(env_ids) == 0:
            return

        curriculum_mode = str(self.wrench_cfg.get("curriculum_mode", "adaptive"))
        if curriculum_mode == "fixed":
            self.apply_force_scale[env_ids] = float(
                self.wrench_cfg.get("fixed_force_scale", self.wrench_cfg.get("force_scale_initial", 0.1))
            )
            self.apply_torque_scale[env_ids] = float(
                self.wrench_cfg.get("fixed_torque_scale", self.wrench_cfg.get("torque_scale_initial", 0.05))
            )
            return
        if curriculum_mode not in ("adaptive", "adaptive_floor"):
            raise ValueError(
                f"Unsupported wrench.curriculum_mode={curriculum_mode}. Expected adaptive, adaptive_floor, or fixed."
            )

        current_iteration = self._get_wrench_curriculum_iteration()
        force_warmup_iterations = int(self.wrench_cfg.get("force_curriculum_warmup_iterations", 0))
        force_allow_down_after = int(
            self.wrench_cfg.get("force_curriculum_allow_down_after_iterations", force_warmup_iterations)
        )
        force_disable_down_until = int(self.wrench_cfg.get("disable_force_down_until_iterations", 0))
        force_in_warmup = current_iteration < force_warmup_iterations
        force_allow_down = current_iteration >= max(force_allow_down_after, force_disable_down_until)
        force_floor = float(self.wrench_cfg.get("force_scale_floor_during_warmup", 0.0))

        torque_warmup_iterations = int(self.wrench_cfg.get("torque_curriculum_warmup_iterations", 0))
        torque_allow_down_after = int(
            self.wrench_cfg.get("torque_curriculum_allow_down_after_iterations", torque_warmup_iterations)
        )
        torque_disable_down_until = int(self.wrench_cfg.get("disable_torque_down_until_iterations", 0))
        torque_in_warmup = current_iteration < torque_warmup_iterations
        torque_allow_down = current_iteration >= max(torque_allow_down_after, torque_disable_down_until)
        torque_floor = float(self.wrench_cfg.get("torque_scale_floor_during_warmup", 0.0))

        up_threshold = self.config.rewards.get("force_scale_up_threshold", 210)
        down_threshold = self.config.rewards.get("force_scale_down_threshold", 200)
        force_up = float(self.wrench_cfg.get("force_scale_up", self.config.rewards.get("force_scale_up", 0.02)))
        force_down = float(self.wrench_cfg.get("force_scale_down", self.config.rewards.get("force_scale_down", 0.02)))
        torque_up = float(self.wrench_cfg.get("torque_scale_up", force_up))
        torque_down = float(self.wrench_cfg.get("torque_scale_down", force_down))

        scale_up_mask = self.episode_length_buf[env_ids] > up_threshold
        scale_down_mask = self.episode_length_buf[env_ids] < down_threshold
        scale_up_ids = env_ids[torch.where(scale_up_mask)[0]]
        scale_down_ids = env_ids[torch.where(scale_down_mask)[0]]

        self.apply_force_scale[scale_up_ids] += force_up
        if force_allow_down:
            self.apply_force_scale[scale_down_ids] -= force_down

        force_min = float(self.wrench_cfg.get("force_scale_min", 0.0))
        if force_in_warmup:
            force_min = max(force_min, force_floor)
        else:
            force_min = max(force_min, float(self.wrench_cfg.get("force_scale_min_after_warmup", force_min)))
        if current_iteration < force_disable_down_until:
            force_min = max(force_min, force_floor)
        self.apply_force_scale[env_ids] = torch.clip(
            self.apply_force_scale[env_ids],
            force_min,
            float(self.wrench_cfg.get("force_scale_max", 1.0)),
        )

        torque_ready = self.apply_force_scale[env_ids] >= float(
            self.wrench_cfg.get("torque_scale_start_after_force_scale", 0.0)
        )
        torque_env_ids = env_ids[torch.where(torque_ready.squeeze(-1))[0]]
        if len(torque_env_ids) > 0:
            torque_up_ids = scale_up_ids[
                torch.where(
                    self.apply_force_scale[scale_up_ids].squeeze(-1)
                    >= float(self.wrench_cfg.get("torque_scale_start_after_force_scale", 0.0))
                )[0]
            ]
            torque_down_ids = scale_down_ids[
                torch.where(
                    self.apply_force_scale[scale_down_ids].squeeze(-1)
                    >= float(self.wrench_cfg.get("torque_scale_start_after_force_scale", 0.0))
                )[0]
            ]
            self.apply_torque_scale[torque_up_ids] += torque_up
            if torque_allow_down:
                self.apply_torque_scale[torque_down_ids] -= torque_down

            torque_min = float(self.wrench_cfg.get("torque_scale_min", 0.0))
            if torque_in_warmup:
                torque_min = max(torque_min, torque_floor)
            else:
                torque_min = max(
                    torque_min,
                    float(self.wrench_cfg.get("torque_scale_min_after_warmup", torque_min)),
                )
            if current_iteration < torque_disable_down_until:
                torque_min = max(torque_min, torque_floor)
            self.apply_torque_scale[torque_env_ids] = torch.clip(
                self.apply_torque_scale[torque_env_ids],
                torque_min,
                float(self.wrench_cfg.get("torque_scale_max", 1.0)),
            )

    def _get_wrench_curriculum_iteration(self):
        # The env does not receive PPO's iteration index directly. For this
        # isolated 6D-wrench path, estimate it from rollout steps; current PPO
        # configs use 24 steps/env unless overridden for smoke tests.
        steps_per_iteration = int(self.wrench_cfg.get("curriculum_steps_per_iteration", 24))
        return int(self.common_step_counter // max(steps_per_iteration, 1))

    def _update_upper_body_tracking_sigma_curriculum(self, env_ids):
        upper_body_tracking_error = self.upper_body_dofs_tracking_reward[env_ids] / self.episode_length_buf[
            env_ids
        ].clamp_min(1)
        scale_up_mask = upper_body_tracking_error < self.config.rewards.upper_body_tracking_sigma_scale_up_threshold
        scale_down_mask = upper_body_tracking_error > self.config.rewards.upper_body_tracking_sigma_scale_down_threshold
        scale_up_ids = env_ids[torch.where(scale_up_mask)[0]]
        scale_down_ids = env_ids[torch.where(scale_down_mask)[0]]
        self.upper_body_tracking_sigma[scale_up_ids] += self.config.rewards.upper_body_tracking_sigma_scale_up
        self.upper_body_tracking_sigma[scale_down_ids] -= self.config.rewards.upper_body_tracking_sigma_scale_down
        self.upper_body_tracking_sigma[env_ids] = torch.clip(
            self.upper_body_tracking_sigma[env_ids],
            self.config.rewards.upper_body_tracking_sigma_min,
            self.config.rewards.upper_body_tracking_sigma_max,
        )

    def _compute_reward(self):
        super()._compute_reward()
        self._update_wrench_logs()

    def _update_wrench_logs(self):
        left_force_norm = torch.norm(self.apply_force_tensor[:, self.left_hand_link_index, :], dim=-1)
        right_force_norm = torch.norm(self.apply_force_tensor[:, self.right_hand_link_index, :], dim=-1)
        left_torque_norm = torch.norm(self.apply_torque_tensor[:, self.left_hand_link_index, :], dim=-1)
        right_torque_norm = torch.norm(self.apply_torque_tensor[:, self.right_hand_link_index, :], dim=-1)
        force_norm = torch.cat([left_force_norm, right_force_norm], dim=0)
        torque_norm = torch.cat([left_torque_norm, right_torque_norm], dim=0)

        upper_torque = torch.abs(self.torques[:, self.upper_dof_indices])
        upper_limit = self.torque_limits[self.upper_dof_indices].view(1, -1).clamp_min(1.0e-6)
        upper_saturation = upper_torque > upper_limit * self.joint_limit_margin

        self.log_dict["apply_force_scale"] = torch.mean(self.apply_force_scale.detach())
        self.log_dict["apply_torque_scale"] = torch.mean(self.effective_torque_scale.detach())
        self.log_dict["force_norm_mean"] = torch.mean(force_norm.detach())
        self.log_dict["force_norm_max"] = torch.max(force_norm.detach())
        self.log_dict["torque_norm_mean"] = torch.mean(torque_norm.detach())
        self.log_dict["torque_norm_max"] = torch.max(torque_norm.detach())
        scale_factor = self.wrench_scale_down.detach()
        self.log_dict["wrench_scale_down_fraction"] = torch.mean((scale_factor < 0.999).float())
        self.log_dict["wrench_scale_factor_mean"] = torch.mean(scale_factor)
        self.log_dict["wrench_scale_factor_min"] = torch.min(scale_factor)
        self.log_dict["wrench_mode"] = torch.tensor(
            float(self.WRENCH_MODES[self.wrench_mode]), dtype=torch.float, device=self.device
        )
        self.log_dict["reset_rate"] = torch.mean(self.reset_buf.float())
        self.log_dict["upper_body_torque_saturation_ratio"] = torch.mean(upper_saturation.float())

    def _reward_tracking_upper_body_dofs(self):
        upper_body_pos = self.simulator.dof_pos[:, self.upper_dof_indices]
        upper_body_dofs_error = torch.sum(torch.square(upper_body_pos - self.ref_upper_dof_pos), dim=1)
        upper_body_dofs_tracking_reward = torch.exp(
            -upper_body_dofs_error / self.upper_body_tracking_sigma.squeeze(-1)
        )
        self.upper_body_dofs_tracking_reward += upper_body_dofs_tracking_reward
        return upper_body_dofs_tracking_reward

    def _reward_tracking_walk_base_height(self):
        total_apply_force = torch.norm(
            self.apply_force_tensor[:, self.left_hand_link_index, :]
            + self.apply_force_tensor[:, self.right_hand_link_index, :],
            dim=1,
        )
        base_height_error = torch.abs(self.commands[:, 8] - self.simulator.robot_root_states[:, 2]) * (
            1 - torch.clip(total_apply_force / 50, 0, 1)
        )
        return torch.exp(-base_height_error / self.config.rewards.reward_tracking_sigma.base_height) * self.commands[:, 4]

    def _reward_tracking_stance_base_height(self):
        base_height_error = torch.abs(self.commands[:, 8] - self.simulator.robot_root_states[:, 2])
        return torch.exp(-base_height_error / self.config.rewards.reward_tracking_sigma.base_height) * (
            1 - self.commands[:, 4]
        )

    def _reward_penalty_ankle_roll(self):
        left_ankle_roll = self.simulator.dof_pos[:, self.left_ankle_dof_indices[1:2]]
        right_ankle_roll = self.simulator.dof_pos[:, self.right_ankle_dof_indices[1:2]]
        return torch.sum(torch.abs(left_ankle_roll) + torch.abs(right_ankle_roll), dim=1)

    def _reward_penalty_stance_feet_vel(self):
        left_ee_vel = torch.cat(
            [
                self.simulator._rigid_body_vel[:, self.left_hand_link_index, 0:3],
                self.simulator._rigid_body_ang_vel[:, self.left_hand_link_index, 0:3],
            ],
            dim=1,
        )
        right_ee_vel = torch.cat(
            [
                self.simulator._rigid_body_vel[:, self.right_hand_link_index, 0:3],
                self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, 0:3],
            ],
            dim=1,
        )
        return (torch.norm(left_ee_vel, dim=1) + torch.norm(right_ee_vel, dim=1)) * (1 - self.commands[:, 4])

    def _reward_penalty_ee_lin_acc(self):
        left_ee_lin_acc = self.simulator._rigid_body_vel[:, self.left_hand_link_index, 0:3] - self.last_left_ee_vel
        right_ee_lin_acc = self.simulator._rigid_body_vel[:, self.right_hand_link_index, 0:3] - self.last_right_ee_vel
        end_effector_acc = torch.cat([left_ee_lin_acc.unsqueeze(1), right_ee_lin_acc.unsqueeze(1)], dim=1)
        return torch.sum(torch.norm(end_effector_acc, dim=2), dim=1)

    def _reward_penalty_ee_ang_acc(self):
        left_ee_ang_acc = (
            self.simulator._rigid_body_ang_vel[:, self.left_hand_link_index, 0:3] - self.last_left_ee_ang_vel
        )
        right_ee_ang_acc = (
            self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, 0:3] - self.last_right_ee_ang_vel
        )
        end_effector_ang_acc = torch.cat([left_ee_ang_acc.unsqueeze(1), right_ee_ang_acc.unsqueeze(1)], dim=1)
        return torch.sum(torch.norm(end_effector_ang_acc, dim=2), dim=1)

    def _get_obs_left_ee_apply_force(self):
        return self.left_ee_apply_force

    def _get_obs_right_ee_apply_force(self):
        return self.right_ee_apply_force

    def _get_obs_left_ee_apply_torque(self):
        return self.left_ee_apply_torque

    def _get_obs_right_ee_apply_torque(self):
        return self.right_ee_apply_torque
