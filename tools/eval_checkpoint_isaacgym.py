#!/usr/bin/env python3
"""Finite IsaacGym checkpoint evaluation without training or ONNX export."""

import csv
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoidverse"))

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.utils.logging import HydraLoggerBridge
from utils.config_utils import *  # noqa: F401,F403,E402


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return default


def _float(value):
    try:
        return float(value)
    except Exception:
        return None


@hydra.main(config_path="../humanoidverse/config", config_name="base", version_base="1.1")
def main(config):
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch  # noqa: E402

    from utils.common import seeding  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config  # noqa: E402

    os.chdir(hydra.utils.get_original_cwd())

    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval_checkpoint.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    if config.checkpoint is None:
        raise ValueError("checkpoint=/path/to/model.pt is required")

    eval_cfg = _cfg_get(config, "eval", None)
    num_steps = int(_cfg_get(eval_cfg, "num_steps", 2000))
    print_interval = int(_cfg_get(eval_cfg, "print_interval", 100))
    load_optimizer = bool(_cfg_get(eval_cfg, "load_optimizer", False))
    command = _cfg_get(eval_cfg, "command", None)
    record_video = bool(_cfg_get(eval_cfg, "record_video", False))

    default_output_dir = Path("logs_eval") / "checkpoint_eval" / Path(config.checkpoint).stem
    output_dir = Path(str(_cfg_get(eval_cfg, "output_dir", default_output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    config.use_wandb = False
    config.algo.config.load_optimizer = load_optimizer
    config.env.config.save_rendering_dir = str(output_dir / "renderings")
    config.env.config.ckpt_dir = str(Path(config.checkpoint).parent)

    if hasattr(config, "device") and config.device is not None:
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pre_process_config(config)
    if config.seed is not None:
        seeding(config.seed, torch_deterministic=config.torch_deterministic)

    logger.info(f"Checkpoint: {config.checkpoint}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"num_envs={config.num_envs}, num_steps={num_steps}, headless={config.headless}")

    env = instantiate(config=config.env, device=device)
    algo = instantiate(device=device, env=env, config=config.algo, log_dir=None)
    algo.setup()
    warm_start_cfg = _cfg_get(config.algo.config, "warm_start", None)
    warm_start_enabled = bool(_cfg_get(warm_start_cfg, "enabled", False))
    if warm_start_enabled:
        logger.info("Using actor-only warm_start loader; skipping full checkpoint load.")
    else:
        algo.load(config.checkpoint)
    algo._eval_mode()
    env.set_is_evaluating(command=command)

    policy = algo._get_inference_policy()
    obs_dict = env.reset_all()
    action_dim = int(config.robot.actions_dim)
    actor_state = {
        "obs": obs_dict,
        "actions": torch.zeros(env.num_envs, action_dim, device=device),
    }

    if record_video:
        if env.viewer is None:
            logger.warning("record_video=True requested, but viewer is not available. Use headless=False.")
        else:
            env.simulator.user_is_recording = True
            env.simulator.user_recording_state_change = True

    rows = []
    nonfinite_count = 0
    done_count = 0
    reward_sum = 0.0
    step_count = 0

    def collect_step_metrics(step, actions, rewards, dones):
        nonlocal nonfinite_count, done_count, reward_sum, step_count

        total_reward = None
        if isinstance(rewards, dict):
            reward_terms = [v.detach() for v in rewards.values()]
            total_reward_tensor = sum(reward_terms)
            total_reward = total_reward_tensor.mean()
        else:
            total_reward_tensor = rewards.detach()
            total_reward = total_reward_tensor.mean()

        done_float = dones.float()
        done_count += int(dones.sum().item())
        reward_sum += float(total_reward.item())
        step_count += 1

        tensors_to_check = [actions, dones, total_reward_tensor]
        tensors_to_check.extend(obs_dict.values())
        for tensor in tensors_to_check:
            if not torch.isfinite(tensor).all():
                nonfinite_count += 1

        row = {
            "step": step,
            "reward_mean": _float(total_reward.item()),
            "done_frac": _float(done_float.mean().item()),
            "action_norm_mean": _float(torch.linalg.norm(actions, dim=1).mean().item()),
            "action_abs_max": _float(actions.abs().max().item()),
        }

        if hasattr(env, "projected_gravity"):
            row["root_tilt_xy_norm_mean"] = _float(torch.linalg.norm(env.projected_gravity[:, :2], dim=1).mean().item())
            row["root_tilt_xy_norm_max"] = _float(torch.linalg.norm(env.projected_gravity[:, :2], dim=1).max().item())
        if hasattr(env, "simulator") and hasattr(env.simulator, "robot_root_states"):
            row["base_height_mean"] = _float(env.simulator.robot_root_states[:, 2].mean().item())
            row["base_height_min"] = _float(env.simulator.robot_root_states[:, 2].min().item())
        if hasattr(env, "ref_upper_dof_pos") and hasattr(env, "upper_dof_indices"):
            err = env.simulator.dof_pos[:, env.upper_dof_indices] - env.ref_upper_dof_pos
            row["upper_dof_rmse"] = _float(torch.sqrt(torch.mean(err * err)).item())
            if hasattr(env, "_reward_tracking_upper_body_dofs"):
                row["tracking_upper_body_dofs_raw"] = _float(env._reward_tracking_upper_body_dofs().mean().item())
        if hasattr(env, "torques") and hasattr(env, "upper_dof_indices"):
            limits = env.torque_limits[env.upper_dof_indices].view(1, -1).clamp_min(1.0e-6)
            upper_ratio = env.torques[:, env.upper_dof_indices].abs() / limits
            row["upper_body_torque_saturation_ratio"] = _float((upper_ratio > 0.9).float().mean().item())
            row["upper_body_torque_ratio_mean"] = _float(upper_ratio.mean().item())
            row["upper_body_torque_ratio_max"] = _float(upper_ratio.max().item())
        if hasattr(env, "left_ee_apply_force") and hasattr(env, "right_ee_apply_force"):
            force_norm = torch.linalg.norm(env.left_ee_apply_force, dim=1) + torch.linalg.norm(env.right_ee_apply_force, dim=1)
            row["ee_force_norm_mean"] = _float(force_norm.mean().item())
            row["ee_force_norm_max"] = _float(force_norm.max().item())
        if hasattr(env, "left_ee_apply_torque") and hasattr(env, "right_ee_apply_torque"):
            torque_norm = torch.linalg.norm(env.left_ee_apply_torque, dim=1) + torch.linalg.norm(env.right_ee_apply_torque, dim=1)
            row["ee_torque_norm_mean"] = _float(torque_norm.mean().item())
            row["ee_torque_norm_max"] = _float(torque_norm.max().item())

        return row

    with torch.no_grad():
        for step in range(num_steps):
            actions = policy(actor_state["obs"]["actor_obs"])
            actor_state["actions"] = actions
            obs_dict, rewards, dones, extras = env.step(actor_state)
            actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})

            row = collect_step_metrics(step, actions, rewards, dones)
            rows.append(row)

            if print_interval > 0 and (step % print_interval == 0 or step == num_steps - 1):
                logger.info(
                    "step={step} reward={reward:.4f} done_frac={done:.4f} "
                    "upper_rmse={upper_rmse} tilt={tilt}".format(
                        step=step,
                        reward=row.get("reward_mean", 0.0),
                        done=row.get("done_frac", 0.0),
                        upper_rmse=row.get("upper_dof_rmse", None),
                        tilt=row.get("root_tilt_xy_norm_mean", None),
                    )
                )

    if record_video and env.viewer is not None:
        env.simulator.user_is_recording = False
        env.simulator.user_recording_state_change = True
        env.render()

    csv_path = output_dir / "metrics_timeseries.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def summarize(key):
        vals = [row[key] for row in rows if row.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "final": vals[-1],
        }

    summary = {
        "checkpoint": str(config.checkpoint),
        "warm_start_report": getattr(algo, "warm_start_report", None),
        "num_envs": int(config.num_envs),
        "num_steps": num_steps,
        "total_policy_steps": int(num_steps * config.num_envs),
        "done_count": done_count,
        "reset_rate": done_count / float(num_steps * config.num_envs),
        "reward_mean_over_steps": reward_sum / max(step_count, 1),
        "nonfinite_count": nonfinite_count,
        "actor_obs_dim": int(config.robot.algo_obs_dim_dict["actor_obs"] * config.obs.history_length.actor_obs),
        "critic_obs_dim": int(config.robot.algo_obs_dim_dict["critic_obs"] * config.obs.history_length.critic_obs),
        "action_dim": action_dim,
        "summaries": {key: summarize(key) for key in fieldnames if key != "step"},
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    logger.info(json.dumps(summary, indent=2))
    logger.info(f"Wrote {summary_path}")
    logger.info(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
