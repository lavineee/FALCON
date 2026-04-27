import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


TRACKING_STATES = {"turn_valve", "segment_hold"}


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _stats(values):
    values = _finite(values)
    if values.size == 0:
        return {
            "mean": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max_abs": math.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p90": float(np.percentile(np.abs(values), 90)),
        "p95": float(np.percentile(np.abs(values), 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def load_csv(path):
    rows = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed = dict(row)
            for key, value in row.items():
                if key == "state":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = math.nan
            rows.append(parsed)
    return rows


def summarize(rows):
    tracking_rows = [row for row in rows if row.get("state") in TRACKING_STATES and row.get("segment_id", -1) >= 0]
    by_segment = defaultdict(list)
    for row in tracking_rows:
        by_segment[int(row["segment_id"])].append(row)

    segments = []
    for segment_id in sorted(by_segment):
        segment_rows = by_segment[segment_id]
        last = segment_rows[-1]
        angle_errors = [row["angle_error_deg"] for row in segment_rows]
        final_angle_errors = [
            row.get("final_angle_error_deg", row["angle_error_deg"])
            for row in segment_rows
        ]
        ee_errors = [row["ee_error_m"] for row in segment_rows]
        stretch = [row["connect_stretch_m"] for row in segment_rows]
        segments.append(
            {
                "segment_id": segment_id,
                "target_delta_deg": float(last["segment_target_delta_deg"]),
                "actual_final_delta_deg": float(last["actual_delta_deg"]),
                "final_error_deg": float(last.get("final_angle_error_deg", last["angle_error_deg"])),
                "angle_mae_deg": _stats(angle_errors)["mae"],
                "final_angle_mae_deg": _stats(final_angle_errors)["mae"],
                "angle_p95_deg": _stats(angle_errors)["p95"],
                "ee_mean_m": _stats(ee_errors)["mean"],
                "ee_p95_m": _stats(ee_errors)["p95"],
                "stretch_mean_m": _stats(stretch)["mean"],
                "stretch_p95_m": _stats(stretch)["p95"],
                "duration_s": float(segment_rows[-1]["time_s"] - segment_rows[0]["time_s"]),
            }
        )

    overall = {
        "samples": len(rows),
        "tracking_samples": len(tracking_rows),
        "segments": len(segments),
        "angle": _stats([row["angle_error_deg"] for row in tracking_rows]),
        "final_angle": _stats([
            row.get("final_angle_error_deg", row["angle_error_deg"])
            for row in tracking_rows
        ]),
        "ee": _stats([row["ee_error_m"] for row in tracking_rows]),
        "stretch": _stats([row["connect_stretch_m"] for row in tracking_rows]),
    }
    return overall, segments


def write_report(path, csv_path, overall, segments):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# 阀门角度跟踪测评",
        "",
        f"- CSV：`{csv_path}`",
        f"- 样本数：`{overall['samples']}`",
        f"- 跟踪样本数：`{overall['tracking_samples']}`",
        f"- 目标段数：`{overall['segments']}`",
        "",
        "## 总体误差",
        "",
        f"- 轨迹角度 MAE：{overall['angle']['mae']:.3f} deg",
        f"- 轨迹角度 RMSE：{overall['angle']['rmse']:.3f} deg",
        f"- 轨迹角度 P95：{overall['angle']['p95']:.3f} deg",
        f"- 轨迹角度最大绝对误差：{overall['angle']['max_abs']:.3f} deg",
        f"- 相对最终目标角度 MAE：{overall['final_angle']['mae']:.3f} deg",
        f"- 末端误差均值：{overall['ee']['mean'] * 100.0:.2f} cm",
        f"- 末端误差 P95：{overall['ee']['p95'] * 100.0:.2f} cm",
        f"- 软连接拉伸均值：{overall['stretch']['mean'] * 100.0:.2f} cm",
        f"- 软连接拉伸 P95：{overall['stretch']['p95'] * 100.0:.2f} cm",
        "",
        "## 分段结果",
        "",
        "| segment | target deg | actual final deg | final error deg | angle MAE deg | EE mean cm | stretch mean cm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for segment in segments:
        lines.append(
            "| {segment_id} | {target_delta_deg:.2f} | {actual_final_delta_deg:.2f} | "
            "{final_error_deg:.2f} | {angle_mae_deg:.2f} | {ee_mean_cm:.2f} | {stretch_mean_cm:.2f} |".format(
                **segment,
                ee_mean_cm=segment["ee_mean_m"] * 100.0,
                stretch_mean_cm=segment["stretch_mean_m"] * 100.0,
            )
        )
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def maybe_plot(rows, output_png):
    if not output_png:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped: matplotlib unavailable: {exc}")
        return

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    time_s = np.asarray([row["time_s"] for row in rows], dtype=float)
    desired_delta = np.asarray([row["desired_delta_deg"] for row in rows], dtype=float)
    command_delta = np.asarray([row.get("command_delta_deg", math.nan) for row in rows], dtype=float)
    actual_delta = np.asarray([row["actual_delta_deg"] for row in rows], dtype=float)
    angle_error = np.asarray([row["angle_error_deg"] for row in rows], dtype=float)
    ee_error_cm = 100.0 * np.asarray([row["ee_error_m"] for row in rows], dtype=float)
    stretch_cm = 100.0 * np.asarray([row["connect_stretch_m"] for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_s, desired_delta, label="reference delta")
    if np.isfinite(command_delta).any():
        axes[0].plot(time_s, command_delta, label="command delta", alpha=0.8)
    axes[0].plot(time_s, actual_delta, label="actual delta")
    axes[0].set_ylabel("angle (deg)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_s, angle_error)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("angle error (deg)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_s, ee_error_cm, label="EE error")
    axes[2].plot(time_s, stretch_cm, label="connect stretch")
    axes[2].set_ylabel("distance (cm)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Analyze valve angle tracking metrics CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--report", default=None)
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    overall, segments = summarize(rows)
    for key, value in overall["angle"].items():
        print(f"angle_{key}_deg={value:.6f}")
    for key, value in overall["final_angle"].items():
        print(f"final_angle_{key}_deg={value:.6f}")
    for key, value in overall["ee"].items():
        print(f"ee_{key}_m={value:.6f}")
    for key, value in overall["stretch"].items():
        print(f"stretch_{key}_m={value:.6f}")
    for segment in segments:
        print(
            "segment={segment_id} target={target_delta_deg:.3f} "
            "actual_final={actual_final_delta_deg:.3f} final_error={final_error_deg:.3f} "
            "angle_mae={angle_mae_deg:.3f} ee_mean={ee_mean_m:.6f} stretch_mean={stretch_mean_m:.6f}".format(
                **segment
            )
        )
    if args.report:
        write_report(args.report, args.csv_path, overall, segments)
    maybe_plot(rows, args.plot)


if __name__ == "__main__":
    main()
