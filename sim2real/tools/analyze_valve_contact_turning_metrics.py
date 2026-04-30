import argparse
import csv
import math
import os

import numpy as np


TURN_STATES = {"turn_valve", "turn_hold"}


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _stats(values):
    values = _finite(values)
    if values.size == 0:
        return {"mean": math.nan, "p90": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _abs_stats(values):
    return _stats(np.abs(_finite(values)))


def _angle_diff_deg(a, b):
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def load_csv(path):
    string_fields = {
        "state",
        "contact_topology",
        "real_contact_topology",
        "contact_pairs",
        "real_contact_pairs",
        "debug_contact_pairs",
        "task_state",
        "segment_result",
        "failure_reason",
        "abort_reason",
    }
    rows = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for raw in reader:
            row = {}
            for key, value in raw.items():
                row[key] = value if key in string_fields else _parse_float(value)
            rows.append(row)
    return rows


def turn_rows(rows):
    return [row for row in rows if row.get("state") in TURN_STATES]


def summarize(rows):
    if not rows:
        empty_stats = {"mean": math.nan, "p90": math.nan, "p95": math.nan, "max": math.nan}
        return {
            "samples": 0,
            "turn_samples": 0,
            "has_turn": False,
            "target_deg": math.nan,
            "actual_final_deg": math.nan,
            "final_error_deg": math.nan,
            "run_final_actual_deg": math.nan,
            "run_final_error_deg": math.nan,
            "trajectory_mae_deg": math.nan,
            "ee_error": empty_stats,
            "relative_slip": empty_stats,
            "contact_force": empty_stats,
            "contact_count": empty_stats,
            "real_contact_force": empty_stats,
            "real_contact_count": empty_stats,
            "contact_health_ratio": empty_stats,
            "soft_connect_active": empty_stats,
            "angle_brake_torque_abs": empty_stats,
            "base_tilt": empty_stats,
            "base_to_valve_x": empty_stats,
            "base_to_valve_y": empty_stats,
            "turn_base_approach_m": math.nan,
            "run_base_approach_m": math.nan,
            "turn_base_yaw_drift_deg": math.nan,
            "run_base_yaw_drift_deg": math.nan,
            "valve_vel_abs": empty_stats,
            "success_fraction": math.nan,
            "ptim_fraction": math.nan,
            "real_ptim_fraction": math.nan,
            "start_state": "",
            "end_state": "no_samples",
            "duration_s": 0.0,
            "final_topology": "",
            "final_real_topology": "",
            "final_contact_count": math.nan,
            "final_contact_force": math.nan,
            "final_real_contact_count": math.nan,
            "final_real_contact_force": math.nan,
            "final_grasp_success": math.nan,
            "failure_reason": "",
            "abort_reason": "",
        }

    selected = turn_rows(rows)
    has_turn = bool(selected)
    # 失败日志可能在 close_hand 阶段就结束。此时仍然要输出抓握/接触诊断，
    # 所以退化为分析全程样本，而不是直接报错。
    analysis_rows = selected if has_turn else rows

    last = analysis_rows[-1]
    target_deg = float(last.get("turn_target_delta_deg", math.nan))
    actual_final_deg = float(last.get("turn_actual_delta_deg", math.nan))
    final_error_deg = float(last.get("turn_final_error_deg", target_deg - actual_final_deg))
    run_last = rows[-1] if has_turn else analysis_rows[-1]
    run_final_actual_deg = float(run_last.get("turn_actual_delta_deg", actual_final_deg))
    run_final_error_deg = float(run_last.get("turn_final_error_deg", target_deg - run_final_actual_deg))
    success_fraction = np.mean([1.0 if row.get("grasp_success", 0.0) >= 0.5 else 0.0 for row in analysis_rows])
    ptim_fraction = np.mean([1.0 if row.get("contact_topology") == "PTIM" else 0.0 for row in analysis_rows])
    real_ptim_fraction = np.mean(
        [1.0 if row.get("real_contact_topology", row.get("contact_topology")) == "PTIM" else 0.0 for row in analysis_rows]
    )
    base_tilt = [
        max(abs(row.get("base_roll_deg", math.nan)), abs(row.get("base_pitch_deg", math.nan)))
        for row in analysis_rows
    ]
    angle_errors = _finite([row.get("turn_angle_error_deg") for row in analysis_rows])
    duration_s = 0.0
    turn_base_approach_m = math.nan
    run_base_approach_m = math.nan
    turn_base_yaw_drift_deg = math.nan
    run_base_yaw_drift_deg = math.nan
    if has_turn:
        duration_s = float(selected[-1].get("time_s", 0.0) - selected[0].get("time_s", 0.0))
        start_x = float(selected[0].get("base_to_valve_x", math.nan))
        turn_end_x = float(selected[-1].get("base_to_valve_x", math.nan))
        run_end_x = float(rows[-1].get("base_to_valve_x", math.nan))
        if math.isfinite(start_x) and math.isfinite(turn_end_x):
            turn_base_approach_m = start_x - turn_end_x
        if math.isfinite(start_x) and math.isfinite(run_end_x):
            run_base_approach_m = start_x - run_end_x
        start_yaw = float(selected[0].get("base_yaw_deg", math.nan))
        turn_end_yaw = float(selected[-1].get("base_yaw_deg", math.nan))
        run_end_yaw = float(rows[-1].get("base_yaw_deg", math.nan))
        turn_base_yaw_drift_deg = abs(_angle_diff_deg(turn_end_yaw, start_yaw))
        run_base_yaw_drift_deg = abs(_angle_diff_deg(run_end_yaw, start_yaw))

    return {
        "samples": len(rows),
        "turn_samples": len(selected),
        "has_turn": has_turn,
        "target_deg": target_deg,
        "actual_final_deg": actual_final_deg,
        "final_error_deg": final_error_deg,
        "run_final_actual_deg": run_final_actual_deg,
        "run_final_error_deg": run_final_error_deg,
        "trajectory_mae_deg": float(np.mean(np.abs(angle_errors))) if angle_errors.size else math.nan,
        "ee_error": _stats([row.get("ee_error_m") for row in analysis_rows]),
        "relative_slip": _stats([row.get("relative_slip_m") for row in analysis_rows]),
        "contact_force": _stats([row.get("contact_normal_force") for row in analysis_rows]),
        "contact_count": _stats([row.get("contact_count") for row in analysis_rows]),
        "real_contact_force": _stats(
            [row.get("real_contact_normal_force", row.get("contact_normal_force")) for row in analysis_rows]
        ),
        "real_contact_count": _stats(
            [row.get("real_contact_count", row.get("contact_count")) for row in analysis_rows]
        ),
        "contact_health_ratio": _stats([row.get("contact_health_ratio") for row in analysis_rows]),
        "soft_connect_active": _stats([row.get("soft_connect_active") for row in analysis_rows]),
        "angle_brake_torque_abs": _abs_stats([row.get("angle_brake_torque") for row in analysis_rows]),
        "base_tilt": _stats(base_tilt),
        "base_to_valve_x": _stats([row.get("base_to_valve_x") for row in analysis_rows]),
        "base_to_valve_y": _stats([row.get("base_to_valve_y") for row in analysis_rows]),
        "turn_base_approach_m": float(turn_base_approach_m),
        "run_base_approach_m": float(run_base_approach_m),
        "turn_base_yaw_drift_deg": float(turn_base_yaw_drift_deg),
        "run_base_yaw_drift_deg": float(run_base_yaw_drift_deg),
        "valve_vel_abs": _abs_stats([row.get("valve_vel") for row in analysis_rows]),
        "success_fraction": float(success_fraction),
        "ptim_fraction": float(ptim_fraction),
        "real_ptim_fraction": float(real_ptim_fraction),
        "start_state": analysis_rows[0].get("state"),
        "end_state": rows[-1].get("state"),
        "duration_s": duration_s,
        "final_topology": last.get("contact_topology", ""),
        "final_real_topology": last.get("real_contact_topology", last.get("contact_topology", "")),
        "final_contact_count": float(last.get("contact_count", math.nan)),
        "final_contact_force": float(last.get("contact_normal_force", math.nan)),
        "final_real_contact_count": float(last.get("real_contact_count", last.get("contact_count", math.nan))),
        "final_real_contact_force": float(
            last.get("real_contact_normal_force", last.get("contact_normal_force", math.nan))
        ),
        "final_grasp_success": float(last.get("grasp_success", math.nan)),
        "failure_reason": str(run_last.get("failure_reason", "")),
        "abort_reason": str(run_last.get("abort_reason", "")),
    }


def write_report(path, csv_path, summary):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# 阀门真实接触转动测评",
        "",
        f"- CSV：`{csv_path}`",
        f"- 样本数：`{summary['samples']}`",
        f"- 转动阶段样本数：`{summary['turn_samples']}`",
        f"- 转动阶段时长：`{summary['duration_s']:.2f} s`",
        f"- 最终状态：`{summary['end_state']}`",
        f"- 是否进入转动阶段：`{summary['has_turn']}`",
        "",
        "## 角度控制",
        "",
        f"- 目标转角：`{summary['target_deg']:.2f} deg`",
        f"- 转动/保持阶段末尾实际转角：`{summary['actual_final_deg']:.2f} deg`",
        f"- 转动/保持阶段末尾误差：`{summary['final_error_deg']:.2f} deg`",
        f"- 整次运行末尾实际转角：`{summary['run_final_actual_deg']:.2f} deg`",
        f"- 整次运行末尾误差：`{summary['run_final_error_deg']:.2f} deg`",
        f"- 轨迹角度 MAE：`{summary['trajectory_mae_deg']:.2f} deg`",
        f"- 阀门角速度 P95：`{summary['valve_vel_abs']['p95']:.3f} rad/s`",
        "",
        "## 接触和稳定性",
        "",
        f"- 末端误差均值 / P90：`{summary['ee_error']['mean'] * 100.0:.2f} / {summary['ee_error']['p90'] * 100.0:.2f} cm`",
        f"- 相对滑移均值 / P90：`{summary['relative_slip']['mean'] * 100.0:.2f} / {summary['relative_slip']['p90'] * 100.0:.2f} cm`",
        f"- 接触力均值 / P90：`{summary['contact_force']['mean']:.2f} / {summary['contact_force']['p90']:.2f} N`",
        f"- 接触点数量均值 / P90：`{summary['contact_count']['mean']:.2f} / {summary['contact_count']['p90']:.2f}`",
        f"- 最终接触拓扑 / 接触点 / 接触力：`{summary['final_topology']} / {summary['final_contact_count']:.0f} / {summary['final_contact_force']:.2f} N`",
        f"- 真实手部接触拓扑 / 接触点 / 接触力：`{summary['final_real_topology']} / {summary['final_real_contact_count']:.0f} / {summary['final_real_contact_force']:.2f} N`",
        f"- 最终 `grasp_success`：`{summary['final_grasp_success']:.0f}`",
        f"- `grasp_success` 占比：`{summary['success_fraction']:.2f}`",
        f"- `PTIM` 完整接触占比：`{summary['ptim_fraction']:.2f}`",
        f"- 真实手部 `PTIM` 占比：`{summary['real_ptim_fraction']:.2f}`",
        f"- contact health ratio 均值 / P90：`{summary['contact_health_ratio']['mean']:.2f} / {summary['contact_health_ratio']['p90']:.2f}`",
        f"- soft connect active ratio：`{summary['soft_connect_active']['mean']:.2f}`",
        f"- angle brake torque 峰值：`{summary['angle_brake_torque_abs']['max']:.2f}`",
        f"- base tilt 均值 / P90：`{summary['base_tilt']['mean']:.2f} / {summary['base_tilt']['p90']:.2f} deg`",
        f"- base 到阀门中心 dx 均值 / P90：`{summary['base_to_valve_x']['mean']:.3f} / {summary['base_to_valve_x']['p90']:.3f} m`",
        f"- base 到阀门中心 dy 均值 / P90：`{summary['base_to_valve_y']['mean']:.3f} / {summary['base_to_valve_y']['p90']:.3f} m`",
        f"- 转动/保持阶段 base 靠近量：`{summary['turn_base_approach_m']:.3f} m`",
        f"- 整次运行 base 靠近量：`{summary['run_base_approach_m']:.3f} m`",
        f"- 转动/保持阶段 base yaw 漂移：`{summary['turn_base_yaw_drift_deg']:.2f} deg`",
        f"- 整次运行 base yaw 漂移：`{summary['run_base_yaw_drift_deg']:.2f} deg`",
        f"- failure_reason：`{summary['failure_reason']}`",
        f"- abort_reason：`{summary['abort_reason']}`",
    ]
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def maybe_plot(rows, output_png):
    if not output_png:
        return
    if not rows:
        print("plot skipped: no samples")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped: matplotlib unavailable: {exc}")
        return

    selected = turn_rows(rows)
    if not selected:
        selected = rows
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    t0 = selected[0].get("time_s", 0.0)
    t = np.asarray([row.get("time_s", math.nan) - t0 for row in selected], dtype=float)
    reference = np.asarray([row.get("turn_reference_delta_deg", math.nan) for row in selected], dtype=float)
    command = np.asarray([row.get("turn_command_delta_deg", math.nan) for row in selected], dtype=float)
    actual = np.asarray([row.get("turn_actual_delta_deg", math.nan) for row in selected], dtype=float)
    final_error = np.asarray([row.get("turn_final_error_deg", math.nan) for row in selected], dtype=float)
    slip_cm = 100.0 * np.asarray([row.get("relative_slip_m", math.nan) for row in selected], dtype=float)
    ee_cm = 100.0 * np.asarray([row.get("ee_error_m", math.nan) for row in selected], dtype=float)
    force = np.asarray([row.get("contact_normal_force", math.nan) for row in selected], dtype=float)
    contact_count = np.asarray([row.get("contact_count", math.nan) for row in selected], dtype=float)
    base_tilt = np.asarray([
        max(abs(row.get("base_roll_deg", math.nan)), abs(row.get("base_pitch_deg", math.nan)))
        for row in selected
    ], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, reference, label="reference", linewidth=1.8)
    axes[0].plot(t, command, label="command", linewidth=1.4)
    axes[0].plot(t, actual, label="actual", linewidth=1.8)
    axes[0].set_ylabel("angle (deg)")
    axes[0].legend(loc="best", ncol=3)

    axes[1].plot(t, final_error, color="#d62728")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("final err (deg)")

    axes[2].plot(t, ee_cm, label="EE error")
    axes[2].plot(t, slip_cm, label="relative slip")
    axes[2].set_ylabel("distance (cm)")
    axes[2].legend(loc="best")

    axes[3].plot(t, force, label="normal force (N)")
    axes[3].plot(t, contact_count, label="contact count")
    axes[3].plot(t, base_tilt, label="base tilt (deg)")
    axes[3].set_xlabel("turn-stage time (s)")
    axes[3].legend(loc="best", ncol=3)

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Analyze true-contact valve turning metrics CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--report", default=None)
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    summary = summarize(rows)
    print(f"target_deg={summary['target_deg']:.6f}")
    print(f"actual_final_deg={summary['actual_final_deg']:.6f}")
    print(f"final_error_deg={summary['final_error_deg']:.6f}")
    print(f"run_final_actual_deg={summary['run_final_actual_deg']:.6f}")
    print(f"run_final_error_deg={summary['run_final_error_deg']:.6f}")
    print(f"trajectory_mae_deg={summary['trajectory_mae_deg']:.6f}")
    print(f"ee_mean_m={summary['ee_error']['mean']:.6f}")
    print(f"relative_slip_p90_m={summary['relative_slip']['p90']:.6f}")
    print(f"contact_force_mean_n={summary['contact_force']['mean']:.6f}")
    print(f"base_tilt_p90_deg={summary['base_tilt']['p90']:.6f}")
    print(f"turn_base_approach_m={summary['turn_base_approach_m']:.6f}")
    print(f"run_base_approach_m={summary['run_base_approach_m']:.6f}")
    print(f"turn_base_yaw_drift_deg={summary['turn_base_yaw_drift_deg']:.6f}")
    print(f"run_base_yaw_drift_deg={summary['run_base_yaw_drift_deg']:.6f}")
    print(f"success_fraction={summary['success_fraction']:.6f}")
    print(f"ptim_fraction={summary['ptim_fraction']:.6f}")
    print(f"contact_health_ratio_mean={summary['contact_health_ratio']['mean']:.6f}")
    print(f"soft_connect_active_ratio={summary['soft_connect_active']['mean']:.6f}")
    print(f"angle_brake_peak_torque={summary['angle_brake_torque_abs']['max']:.6f}")
    print(f"failure_reason={summary['failure_reason']}")
    print(f"abort_reason={summary['abort_reason']}")
    if args.report:
        write_report(args.report, args.csv_path, summary)
    maybe_plot(rows, args.plot)


if __name__ == "__main__":
    main()
