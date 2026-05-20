# G1 D435I `T_base_camera` 标定工具

这个目录只做视觉外参标定，不接入机器人控制，不启动 lowcmd，也不修改 FALCON 策略主逻辑。

## 坐标定义

目标外参是 G1 头部 D435I 的 `camera_color_optical_frame` 到 FALCON `base` frame：

```text
p_base = R_base_camera @ p_camera + t_base_camera
```

也就是 `T_base_camera` 表示 camera frame -> base frame。

RealSense/OpenCV optical frame 约定：

- `x` 向右
- `y` 向下
- `z` 向前

OpenCV `solvePnP` 和 `cv2.aruco.estimatePoseSingleMarkers` 输出的是 object/tag frame 到 camera frame 的位姿：

```text
T_camera_tag
p_camera = R_camera_tag @ p_tag + t_camera_tag
```

人工测量的是 tag frame 到 base frame 的位姿：

```text
T_base_tag
```

因此每条样本给出的外参候选为：

```text
T_base_camera = T_base_tag @ inv(T_camera_tag)
```

注意：人工填写 `T_base_tag` 时，tag frame 必须和 OpenCV ArUco marker object-point 约定一致。当前工具使用 marker 中心为原点、角点顺序为 top-left, top-right, bottom-right, bottom-left。

## 阶段 1: 在 G1 ROS1 电脑采集 `T_camera_tag`

在 192.168.123.164 的 ROS1 Noetic 环境运行。默认 topic：

- `/camera/color/image_raw`
- `/camera/color/camera_info`
- `/camera/aligned_depth_to_color/image_raw`

示例：

```bash
cd /path/to/FALCON/sim2real/vision/valve_vision_ros1/calibration
python3 calib_collect_tag_pose_ros1.py \
  --tag-id 0 \
  --tag-size-m 0.08 \
  --output samples/T_camera_tag_samples.jsonl
```

看到稳定检测后，每按一次 Enter 保存一条 sample。也可以加 `--auto-save-stable` 让工具在最近若干帧稳定时自动保存。

每条 sample 会保存：

- `T_camera_tag.translation_m`
- `T_camera_tag.rotation_matrix`
- `T_camera_tag.rvec`
- `T_camera_tag.tvec`
- `corners_px`
- `reprojection_error_px`
- `depth_median_m`
- `depth_vs_pnp_error_m`
- image/camera_info topic

如果 `cv2.aruco` 不存在，说明当前 OpenCV 缺少 contrib 模块，需要安装系统 OpenCV contrib 或 `opencv-contrib-python`。

## 阶段 2: 填写 `T_base_tag` 并求解 `T_base_camera`

把每条采样位置对应的人工测量值填到：

```text
config/base_tag_poses.yaml
```

格式：

```yaml
samples:
  - sample_id: 0
    T_base_tag:
      translation_m: [0.70, 0.00, 0.90]
      rotation_matrix:
        - [1.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0]
        - [0.0, 0.0, 1.0]
  - sample_id: 1
    T_base_tag:
      translation_m: [0.70, 0.10, 0.90]
      rotation_rpy_deg: [0.0, 0.0, 0.0]
```

求解：

```bash
python3 calib_solve_T_base_camera.py \
  --samples samples/T_camera_tag_samples.jsonl \
  --base-tag-poses config/base_tag_poses.yaml \
  --output output/T_base_camera.yaml
```

solver 会先用所有样本求初值，再按默认阈值剔除异常样本：

- `--max-translation-error-m 0.05`
- `--max-rotation-error-deg 5.0`

然后重新融合最终外参。平移默认取 median，可用：

```bash
--translation-aggregate mean
```

输出文件：

```text
output/T_base_camera.yaml
```

其中包含完整 `T_base_camera` 和可复制到 `valve_vision_g1.yaml` 的片段：

```yaml
camera_extrinsic:
  T_base_camera:
    translation_m: [...]
    rotation_matrix:
      - [...]
      - [...]
      - [...]
```

## 阶段 3: live sanity check

在 ROS1 电脑运行：

```bash
python3 check_T_base_camera_live.py \
  --tag-id 0 \
  --tag-size-m 0.08 \
  --extrinsic output/T_base_camera.yaml
```

实时输出：

- `p_cam`
- `p_base`
- `depth_m`
- `tag_id`
- `reprojection_error_px`

最小符号检查：

- tag 往机器人前方移动，`p_base.x` 应增大
- tag 往机器人左侧移动，`p_base.y` 应增大
- tag 往上移动，`p_base.z` 应增大

如果误差超过 5 cm，不允许直接接机器人 approach。先检查 tag 尺寸、角点检测质量、`T_base_tag` 人工测量、base frame 定义和相机固定状态。

## 推荐采样方式

- 机器人站定
- 腰部保持 0
- tag 放在 6-10 个不同位置
- 每个位置人工测量 `T_base_tag`
- tag 姿态尽量覆盖不同 `x/y/z` 和轻微角度变化
- 每个位置保存前观察 `reprojection_error_px` 和 `depth_vs_pnp_error_m`

## 当前限制

第一版假设相机相对 base 固定。如果腰部或头部会动，后续需要建模 `T_base_camera(q)`，或者通过 URDF/Pinocchio 按当前关节状态实时更新外参。

## 离线自检

本机不需要 ROS 也能运行 solver 自检：

```bash
python3 tests/test_calib_solve_T_base_camera.py
```

该测试会构造已知 `T_base_camera` 和多组 `T_camera_tag/T_base_tag`，确认 solver 能恢复外参。

