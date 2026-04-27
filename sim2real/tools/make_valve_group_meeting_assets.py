import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


TRACKING_STATES = {"turn_valve", "segment_hold"}


def configure_fonts():
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path):
    rows = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for raw in reader:
            row = dict(raw)
            for key, value in raw.items():
                if key != "state":
                    row[key] = parse_float(value)
            rows.append(row)
    return rows


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def stat(values, fn, default=math.nan):
    values = finite(values)
    if values.size == 0:
        return default
    return float(fn(values))


def summarize_segments(rows):
    by_segment = defaultdict(list)
    for row in rows:
        if row.get("state") in TRACKING_STATES and row.get("segment_id", -1) >= 0:
            by_segment[int(row["segment_id"])].append(row)

    segments = []
    for segment_id in sorted(by_segment):
        segment_rows = by_segment[segment_id]
        last = segment_rows[-1]
        angle_errors = [row["angle_error_deg"] for row in segment_rows]
        ee_errors = [row["ee_error_m"] for row in segment_rows]
        stretch = [row["connect_stretch_m"] for row in segment_rows]
        complete = any(row["state"] == "segment_hold" for row in segment_rows)
        segments.append(
            {
                "segment_id": segment_id,
                "target_delta_deg": float(last["segment_target_delta_deg"]),
                "actual_final_delta_deg": float(last["actual_delta_deg"]),
                "final_error_deg": float(last.get("final_angle_error_deg", last["angle_error_deg"])),
                "angle_mae_deg": stat(np.abs(angle_errors), np.mean),
                "ee_mean_cm": 100.0 * stat(ee_errors, np.mean),
                "stretch_mean_cm": 100.0 * stat(stretch, np.mean),
                "duration_s": float(last["time_s"] - segment_rows[0]["time_s"]),
                "complete": complete,
            }
        )
    return segments


def tracking_arrays(rows):
    track = [row for row in rows if row.get("state") in TRACKING_STATES and row.get("segment_id", -1) >= 0]
    if not track:
        raise ValueError("CSV does not contain valve tracking rows")
    keys = [
        "time_s",
        "desired_delta_deg",
        "command_delta_deg",
        "actual_delta_deg",
        "angle_error_deg",
        "ee_error_m",
        "connect_stretch_m",
        "segment_id",
        "segment_target_delta_deg",
    ]
    data = {key: np.asarray([parse_float(row.get(key)) for row in track], dtype=float) for key in keys}
    data["state"] = [row["state"] for row in track]
    return track, data


def make_summary_figure(rows, segments, output):
    _, data = tracking_arrays(rows)
    t = data["time_s"] - data["time_s"][0]
    complete_segments = [item for item in segments if item["complete"]]
    final_errors = [item["final_error_deg"] for item in complete_segments]

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax_angle = fig.add_subplot(grid[0, :])
    ax_err = fig.add_subplot(grid[1, 0])
    ax_dist = fig.add_subplot(grid[1, 1])

    ax_angle.plot(t, data["desired_delta_deg"], label="参考角", linewidth=2.0, color="#2ca02c")
    ax_angle.plot(t, data["command_delta_deg"], label="闭环命令角", linewidth=1.8, color="#ffbf00")
    ax_angle.plot(t, data["actual_delta_deg"], label="阀门实际角", linewidth=2.0, color="#d62728")
    ax_angle.set_title("阀门角度闭环跟踪：参考/命令/实际")
    ax_angle.set_ylabel("角度增量 (deg)")
    ax_angle.grid(True, alpha=0.25)
    ax_angle.legend(loc="best", ncol=3)

    ids = [item["segment_id"] for item in segments]
    errors = [item["final_error_deg"] for item in segments]
    colors = ["#4c78a8" if item["complete"] else "#f58518" for item in segments]
    labels = [str(item["segment_id"]) + ("" if item["complete"] else "*") for item in segments]
    ax_err.bar(labels, errors, color=colors)
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.set_title("各段最终角度误差（* 表示时长截断）")
    ax_err.set_xlabel("segment")
    ax_err.set_ylabel("target - actual (deg)")
    ax_err.grid(True, axis="y", alpha=0.25)

    ax_dist.plot(t, 100.0 * data["ee_error_m"], label="末端位置误差", color="#9467bd", linewidth=1.8)
    ax_dist.plot(t, 100.0 * data["connect_stretch_m"], label="软连接拉伸", color="#17becf", linewidth=1.8)
    ax_dist.set_title("末端误差与软连接拉伸")
    ax_dist.set_xlabel("跟踪时间 (s)")
    ax_dist.set_ylabel("距离 (cm)")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(loc="best")

    angle_mae = stat(np.abs(data["angle_error_deg"]), np.mean)
    angle_p95 = stat(np.abs(data["angle_error_deg"]), lambda x: np.percentile(x, 95))
    ee_mean = 100.0 * stat(data["ee_error_m"], np.mean)
    stretch_mean = 100.0 * stat(data["connect_stretch_m"], np.mean)
    final_mae = stat(np.abs(final_errors), np.mean)
    summary_text = (
        f"轨迹角 MAE {angle_mae:.2f} deg | P95 {angle_p95:.2f} deg\n"
        f"完整段最终误差 MAE {final_mae:.2f} deg | 末端均值 {ee_mean:.2f} cm | 软连接均值 {stretch_mean:.2f} cm"
    )
    ax_angle.text(
        0.01,
        0.04,
        summary_text,
        transform=ax_angle.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaaaaa", alpha=0.85),
    )
    fig.suptitle("FALCON 带手模型阀门转动 baseline 测评", fontsize=18, fontweight="bold")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_method_diagram(output):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_axis_off()
    boxes = [
        ("MuJoCo 状态\n阀门中心/轴线/角度", (0.05, 0.58)),
        ("角度闭环外环\n参考角 - 实际角", (0.28, 0.58)),
        ("阀门平面圆弧目标\n生成掌心期望点", (0.51, 0.58)),
        ("Base 坐标系 IK\n7DoF 手臂位置跟踪", (0.74, 0.58)),
        ("RL residual + PD\n驱动上肢/保持全身平衡", (0.51, 0.18)),
        ("软 connect 约束\n接触后带动阀门", (0.28, 0.18)),
        ("阀门实际角反馈\n形成下一步修正", (0.05, 0.18)),
    ]
    for text, (x, y) in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f4f7fb", ec="#4c78a8", lw=1.6),
        )

    arrows = [
        ((0.14, 0.58), (0.20, 0.58)),
        ((0.37, 0.58), (0.43, 0.58)),
        ((0.60, 0.58), (0.66, 0.58)),
        ((0.74, 0.50), (0.58, 0.26)),
        ((0.43, 0.18), (0.37, 0.18)),
        ((0.20, 0.18), (0.14, 0.18)),
        ((0.05, 0.26), (0.05, 0.50)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2.0, color="#4c78a8"))

    ax.text(
        0.5,
        0.92,
        "当前 baseline：不是手指真实抓握，而是软点约束下验证“接触后能否按角度转动”",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.04,
        "部署替换点：MuJoCo 特权状态 -> 深度相机/视觉估计阀门位姿与角度；软 connect -> 真实抓握与接触保持",
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_animation_frame(data, frame_idx, output):
    idx = int(frame_idx)
    t = data["time_s"] - data["time_s"][0]
    current_time = t[idx]
    desired = data["desired_delta_deg"][idx]
    command = data["command_delta_deg"][idx]
    actual = data["actual_delta_deg"][idx]
    target = data["segment_target_delta_deg"][idx]
    segment = int(data["segment_id"][idx])
    error = data["angle_error_deg"][idx]

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])
    ax_wheel = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    theta = np.linspace(0, 2 * np.pi, 300)
    ax_wheel.plot(np.cos(theta), np.sin(theta), color="#777777", linewidth=2)
    ax_wheel.scatter([0], [0], color="#4c78a8", s=40)

    def ray(angle_deg, color, label, lw):
        rad = np.deg2rad(angle_deg)
        ax_wheel.plot([0, np.cos(rad)], [0, np.sin(rad)], color=color, linewidth=lw, label=label)
        ax_wheel.scatter([np.cos(rad)], [np.sin(rad)], color=color, s=50)

    ray(target, "#9467bd", "最终目标", 2.0)
    ray(command, "#ffbf00", "闭环命令", 3.0)
    ray(actual, "#d62728", "实际角", 3.0)
    ray(desired, "#2ca02c", "参考角", 1.8)
    ax_wheel.set_aspect("equal")
    ax_wheel.set_xlim(-1.25, 1.25)
    ax_wheel.set_ylim(-1.25, 1.25)
    ax_wheel.set_title(f"segment {segment}: target {target:+.0f} deg")
    ax_wheel.legend(loc="upper right", fontsize=9)
    ax_wheel.grid(True, alpha=0.2)

    ax_curve.plot(t[: idx + 1], data["desired_delta_deg"][: idx + 1], color="#2ca02c", label="参考角")
    ax_curve.plot(t[: idx + 1], data["command_delta_deg"][: idx + 1], color="#ffbf00", label="闭环命令")
    ax_curve.plot(t[: idx + 1], data["actual_delta_deg"][: idx + 1], color="#d62728", label="实际角")
    ax_curve.axvline(current_time, color="#333333", linewidth=1.0, alpha=0.6)
    ax_curve.set_xlim(max(0.0, current_time - 20.0), min(t[-1], current_time + 3.0))
    ymin = np.nanmin([data["desired_delta_deg"].min(), data["command_delta_deg"].min(), data["actual_delta_deg"].min()]) - 8
    ymax = np.nanmax([data["desired_delta_deg"].max(), data["command_delta_deg"].max(), data["actual_delta_deg"].max()]) + 8
    ax_curve.set_ylim(ymin, ymax)
    ax_curve.set_xlabel("跟踪时间 (s)")
    ax_curve.set_ylabel("角度增量 (deg)")
    ax_curve.set_title(f"t={current_time:.1f}s | 轨迹误差 {error:+.2f} deg")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(loc="best", fontsize=9)

    fig.suptitle("阀门角度闭环跟踪动画", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=120)
    plt.close(fig)


def make_mp4(frame_paths, output, fps=12):
    try:
        import cv2
    except Exception:
        return False
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return False
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return False
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            writer.write(frame)
    writer.release()
    return True


def make_gif(rows, output, frame_count=90):
    _, data = tracking_arrays(rows)
    indices = np.linspace(0, len(data["time_s"]) - 1, frame_count, dtype=int)
    temp_dir = Path(output).with_suffix("")
    temp_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    frame_paths = []
    for frame_idx, row_idx in enumerate(indices):
        frame_path = temp_dir / f"frame_{frame_idx:03d}.png"
        draw_animation_frame(data, row_idx, frame_path)
        frame_paths.append(frame_path)
        frames.append(Image.open(frame_path).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    for frame in frames:
        frame.close()
    mp4_output = Path(output).with_suffix(".mp4")
    make_mp4(frame_paths, mp4_output)


def write_markdown(rows, segments, output_dir, summary_png, method_png, gif_path, mp4_path):
    _, data = tracking_arrays(rows)
    complete_segments = [item for item in segments if item["complete"]]
    final_errors = [item["final_error_deg"] for item in complete_segments]
    t0 = rows[0]["time_s"]
    t1 = rows[-1]["time_s"]
    angle_mae = stat(np.abs(data["angle_error_deg"]), np.mean)
    angle_p95 = stat(np.abs(data["angle_error_deg"]), lambda x: np.percentile(x, 95))
    ee_mean = 100.0 * stat(data["ee_error_m"], np.mean)
    ee_p95 = 100.0 * stat(data["ee_error_m"], lambda x: np.percentile(x, 95))
    stretch_mean = 100.0 * stat(data["connect_stretch_m"], np.mean)
    stretch_p95 = 100.0 * stat(data["connect_stretch_m"], lambda x: np.percentile(x, 95))
    final_mae = stat(np.abs(final_errors), np.mean)

    lines = [
        "# 组会汇报素材：带手 G1 阀门角度闭环跟踪 baseline",
        "",
        "## 一句话结论",
        "",
        (
            "在“软点连接”接触代理下，7DoF 末端位置控制已经可以驱动阀门完成多段正负角度跟踪；"
            f"完整段最终角度误差 MAE 为 **{final_mae:.2f} deg**，轨迹角度 MAE 为 **{angle_mae:.2f} deg**。"
            "这说明当前控制链路具备继续推进真实抓握/转动阀门任务的基础，但还不能等同于真实手指抓握。"
        ),
        "",
        "## 推荐插入素材",
        "",
        f"- 方法示意图：`{method_png.name}`",
        f"- 结果总览图：`{summary_png.name}`",
        f"- 角度跟踪视频：`{mp4_path.name}`",
        f"- 角度跟踪动图：`{gif_path.name}`",
        "",
        f"![方法示意图](./{method_png.name})",
        "",
        f"![结果总览图](./{summary_png.name})",
        "",
        "## 实验设置",
        "",
        "- 机器人模型：G1 29DoF with hand，策略仍控制本体 29DoF，手部控制器只做开/合状态。",
        "- 连接方式：MuJoCo equality `connect`，连接掌心抓取点和阀门固定点，是软点约束，不是手指真实抓握。",
        "- 控制方式：阀门角度外环闭环，根据 `参考角 - 实际角` 调整末端圆弧命令点，再由 7DoF IK 和原有 policy 执行。",
        "- 测试序列：`20, -60, 35, -45, 55, -30, 40, -50 deg`，总记录时长约 `68 s`。",
        "",
        "## 关键结果",
        "",
        f"- 轨迹角度 MAE：`{angle_mae:.2f} deg`，P95：`{angle_p95:.2f} deg`。",
        f"- 完整段最终角度误差 MAE：`{final_mae:.2f} deg`。",
        f"- 末端位置误差均值：`{ee_mean:.2f} cm`，P95：`{ee_p95:.2f} cm`。",
        f"- 软连接拉伸均值：`{stretch_mean:.2f} cm`，P95：`{stretch_p95:.2f} cm`。",
        "",
        "## 组会汇报结构建议",
        "",
        "1. 背景：为什么从末端点跟踪推进到阀门转动角度跟踪。",
        "2. 当前 baseline 的边界：软 `connect` 不是手指真实抓握，但可以隔离验证接触后控制能力。",
        "3. 方法：阀门几何状态读取、角度外环闭环、圆弧末端目标生成、7DoF IK 和原 policy 执行。",
        "4. 可视化：MuJoCo 中显示最终目标线、闭环命令线、实际阀门径向线和角度标签。",
        "5. 结果：展示总览图和动画，说明完整段最终误差约 2.39 deg。",
        "6. 局限：特权状态、软连接、未建模摩擦和真实抓握。",
        "7. 下一步：视觉接口抽象、接触模型逐步真实化、站位和预抓取姿态优化。",
        "",
        "## 分段结果",
        "",
        "| segment | 目标角 deg | 最终实际角 deg | 最终误差 deg | 轨迹 MAE deg | EE 均值 cm | 软连接均值 cm | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in segments:
        state = "完整" if item["complete"] else "截断"
        lines.append(
            "| {segment_id} | {target_delta_deg:.1f} | {actual_final_delta_deg:.1f} | "
            "{final_error_deg:.2f} | {angle_mae_deg:.2f} | {ee_mean_cm:.2f} | "
            "{stretch_mean_cm:.2f} | {state} |".format(**item, state=state)
        )

    lines.extend(
        [
            "",
            "## 需要在组会上说明的限制",
            "",
            "- 当前连接不是完整抓握模型：没有摩擦、滑移、手指包络和力闭合。",
            "- 角度闭环目前使用 MuJoCo 特权角度；真实部署时需要由深度相机/视觉估计阀门中心、轴线和当前角度。",
            "- 最后一段因为总时长到点被截断，`-50 deg` 目标还没进入 hold，最终误差不能作为稳态误差看。",
            "",
            "## 下一步建议",
            "",
            "1. 把 MuJoCo 特权阀门位姿接口抽象成 `ValveGeometryProvider`，方便之后替换为深度相机估计。",
            "2. 在保持软连接 baseline 的同时，逐步引入更真实的接触：先加入掌心/手指接触几何，再考虑摩擦和滑移。",
            "3. 针对阀门任务重新设计初始站位和预抓取姿态，避免躯干靠阀门过近影响转动空间。",
            "4. 做闭环参数扫描：反馈增益、命令提前量、阀门阻尼/摩擦，找到误差和稳定性之间的折中。",
            "",
        ]
    )
    report_path = output_dir / "group_meeting_valve_closed_loop_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create group-meeting assets for valve angle tracking.")
    parser.add_argument("--csv", default="/tmp/falcon_valve_angle_tracking_closed_loop_1min.csv")
    parser.add_argument("--output_dir", default="sim2real/eval_outputs/group_meeting_valve_closed_loop")
    parser.add_argument("--gif_frames", type=int, default=90)
    args = parser.parse_args()

    configure_fonts()
    rows = load_rows(args.csv)
    segments = summarize_segments(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_png = output_dir / "valve_closed_loop_summary.png"
    method_png = output_dir / "valve_closed_loop_method.png"
    gif_path = output_dir / "valve_closed_loop_tracking_animation.gif"
    mp4_path = output_dir / "valve_closed_loop_tracking_animation.mp4"

    make_summary_figure(rows, segments, summary_png)
    make_method_diagram(method_png)
    make_gif(rows, gif_path, frame_count=args.gif_frames)
    write_markdown(rows, segments, output_dir, summary_png, method_png, gif_path, mp4_path)

    print(summary_png)
    print(method_png)
    print(mp4_path)
    print(gif_path)
    print(output_dir / "group_meeting_valve_closed_loop_report.md")


if __name__ == "__main__":
    main()
