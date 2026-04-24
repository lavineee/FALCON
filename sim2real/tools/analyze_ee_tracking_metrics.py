import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


def _float(row, key, default=np.nan):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_percentile(values, percentile):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, percentile))


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _safe_rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values ** 2)))


def _fmt_m(value):
    if not np.isfinite(value):
        return "nan"
    return f"{value * 100.0:.2f} cm"


def _fmt(value, suffix=""):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}{suffix}"


def load_metrics(path):
    rows = []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if "error_m" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} does not look like a raw EE tracking metrics CSV; "
                "expected an 'error_m' column."
            )
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    time_s = np.asarray([_float(row, "time_s") for row in rows], dtype=float)
    target_id = np.asarray([int(_float(row, "target_id", -1)) for row in rows], dtype=int)
    error_m = np.asarray([_float(row, "error_m") for row in rows], dtype=float)
    target_age_s = np.asarray([_float(row, "target_age_s") for row in rows], dtype=float)
    if np.all(~np.isfinite(target_age_s)):
        first_time_by_target = {}
        ages = []
        for t, tid in zip(time_s, target_id):
            first_time_by_target.setdefault(tid, t)
            ages.append(t - first_time_by_target[tid])
        target_age_s = np.asarray(ages, dtype=float)

    return {
        "rows": rows,
        "time_s": time_s,
        "target_id": target_id,
        "error_m": error_m,
        "target_age_s": target_age_s,
        "ik_error_m": np.asarray([_float(row, "ik_error_m") for row in rows], dtype=float),
        "servo_error_m": np.asarray([_float(row, "servo_error_m") for row in rows], dtype=float),
        "current_speed_mps": np.asarray([_float(row, "current_speed_mps") for row in rows], dtype=float),
    }


def summarize(data, steady_after_s):
    error = data["error_m"]
    steady_mask = data["target_age_s"] >= steady_after_s
    steady_error = error[steady_mask]
    ik_error = data["ik_error_m"][steady_mask]
    servo_error = data["servo_error_m"][steady_mask]
    speed = data["current_speed_mps"][steady_mask]

    grouped = defaultdict(list)
    grouped_final = {}
    for tid, age, err in zip(data["target_id"], data["target_age_s"], error):
        grouped[int(tid)].append((age, err))
        grouped_final[int(tid)] = err

    final_errors = np.asarray(list(grouped_final.values()), dtype=float)
    steady_mean_by_target = []
    for samples in grouped.values():
        steady_samples = [err for age, err in samples if age >= steady_after_s]
        if steady_samples:
            steady_mean_by_target.append(float(np.mean(steady_samples)))

    return {
        "samples": int(error.size),
        "duration_s": float(np.nanmax(data["time_s"]) - np.nanmin(data["time_s"])),
        "targets": int(len(grouped)),
        "overall_mean_m": _safe_mean(error),
        "overall_rms_m": _safe_rms(error),
        "overall_p90_m": _safe_percentile(error, 90),
        "overall_p95_m": _safe_percentile(error, 95),
        "overall_max_m": _safe_percentile(error, 100),
        "steady_mean_m": _safe_mean(steady_error),
        "steady_rms_m": _safe_rms(steady_error),
        "steady_p90_m": _safe_percentile(steady_error, 90),
        "steady_p95_m": _safe_percentile(steady_error, 95),
        "steady_max_m": _safe_percentile(steady_error, 100),
        "final_median_m": _safe_percentile(final_errors, 50),
        "final_p90_m": _safe_percentile(final_errors, 90),
        "ik_steady_mean_m": _safe_mean(ik_error),
        "servo_steady_mean_m": _safe_mean(servo_error),
        "speed_steady_p95_mps": _safe_percentile(speed, 95),
        "steady_mean_by_target_median_m": _safe_percentile(steady_mean_by_target, 50),
        "steady_mean_by_target_p90_m": _safe_percentile(steady_mean_by_target, 90),
    }


def write_report(path, title, csv_path, summary, steady_after_s):
    lines = [
        f"# {title}",
        "",
        f"- 数据文件：`{csv_path}`",
        f"- 稳态统计窗口：每个目标切换后 `{steady_after_s:.2f}s` 之后",
        f"- 样本数：`{summary['samples']}`",
        f"- 目标点数量：`{summary['targets']}`",
        f"- 记录时长：`{summary['duration_s']:.2f}s`",
        "",
        "## 总体误差",
        "",
        f"- mean：{_fmt_m(summary['overall_mean_m'])}",
        f"- RMS：{_fmt_m(summary['overall_rms_m'])}",
        f"- P90：{_fmt_m(summary['overall_p90_m'])}",
        f"- P95：{_fmt_m(summary['overall_p95_m'])}",
        f"- max：{_fmt_m(summary['overall_max_m'])}",
        "",
        "## 稳态误差",
        "",
        f"- steady mean：{_fmt_m(summary['steady_mean_m'])}",
        f"- steady RMS：{_fmt_m(summary['steady_rms_m'])}",
        f"- steady P90：{_fmt_m(summary['steady_p90_m'])}",
        f"- steady P95：{_fmt_m(summary['steady_p95_m'])}",
        f"- steady max：{_fmt_m(summary['steady_max_m'])}",
        f"- final median：{_fmt_m(summary['final_median_m'])}",
        f"- final P90：{_fmt_m(summary['final_p90_m'])}",
        "",
        "## 误差分解",
        "",
        f"- IK 解到最终目标的稳态均值：{_fmt_m(summary['ik_steady_mean_m'])}",
        f"- 实际末端到 IK 解末端的稳态均值：{_fmt_m(summary['servo_steady_mean_m'])}",
        f"- 稳态末端速度 P95：{_fmt(summary['speed_steady_p95_mps'], ' m/s')}",
        "",
        "解释：如果 IK 误差大，主要是目标插值、IK 权重或可达性问题；如果 servo 误差大，主要是关节 PD、前馈、模型阻尼或 residual 叠加导致实际关节没有贴住 IK 参考。",
        "",
    ]
    with open(path, "w") as file:
        file.write("\n".join(lines))


def maybe_plot(data, output_png):
    if not output_png:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    time_s = data["time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, data["error_m"] * 100.0, label="total")
    if np.any(np.isfinite(data["ik_error_m"])):
        axes[0].plot(time_s, data["ik_error_m"] * 100.0, label="ik")
    if np.any(np.isfinite(data["servo_error_m"])):
        axes[0].plot(time_s, data["servo_error_m"] * 100.0, label="servo")
    axes[0].set_ylabel("error [cm]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, data["current_speed_mps"])
    axes[1].set_ylabel("EE speed [m/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(time_s, data["target_id"], where="post")
    axes[2].set_ylabel("target id")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Summarize EE tracking metrics CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--steady_after_s", type=float, default=2.0)
    parser.add_argument("--report", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--title", default="末端跟踪评测报告")
    args = parser.parse_args()

    data = load_metrics(args.csv_path)
    summary = summarize(data, args.steady_after_s)

    for key, value in summary.items():
        if isinstance(value, float):
            if math.isnan(value):
                print(f"{key}: nan")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        write_report(args.report, args.title, args.csv_path, summary, args.steady_after_s)
    if args.plot:
        maybe_plot(data, args.plot)


if __name__ == "__main__":
    main()
