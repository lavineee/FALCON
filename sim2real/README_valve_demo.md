# MuJoCo 阀门主动旋转 Demo 说明

## 1. 本次完成内容

本次基于现有 `sim2real` 框架，完成了论文第四章对应的 MuJoCo 阀门专项测试 demo，核心目标是实现“**阀门主动旋转，机器人抗扭稳定**”的 sim2sim 演示链路。

本次完成的主要内容如下：

### 1.1 阀门场景搭建与接入
已在 G1 的 MuJoCo 场景中接入自定义阀门模型，并完成了以下配置：
- 阀门整体位置调整
- 阀门朝向旋转，使其正对机器人
- 阀门轮转轴 `valve_hinge` 接入仿真
- 阀门锚点 `right_hand_valve_anchor` 增加，用于后续手-阀门固连

对应主要文件：
- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase.xml`
- `humanoidverse/data/robots/g1/valve_body_include.xml`

---

### 1.2 手-阀门固连逻辑
已实现手与阀门之间的 **weld 固连机制**，并支持在仿真运行过程中通过键盘动态开关，而不是在场景导入时默认固连。

这样可以保证：
- 机器人初始导入时不被 weld 拉偏
- 等机器人落地并稳定后，再建立固连
- 更适合做演示和专项测试

---

### 1.3 阀门主动扭矩驱动
已实现对阀门关节 `valve_hinge` 的主动力矩注入，采用：

- `qfrc_applied[valve_dof]`

对阀门施加外部扭矩，而不是依赖阀门 actuator 控制。

这样可以直接实现：
- 恒定扭矩主动旋转
- 键盘实时调节扭矩
- 更适合做抗扭实验

---

### 1.4 MuJoCo 交互控制逻辑
已在 `sim2real/sim_env/loco_manip.py` 中加入阀门任务控制逻辑，包括：

- weld 开关
- 阀门扭矩开关
- 阀门扭矩增减
- 阀门扭矩清零

键盘控制方式如下：

- `J`：切换 hand-valve weld 开/关
- `K`：切换阀门主动扭矩开/关
- `U`：阀门扭矩 +1
- `I`：阀门扭矩 -1
- `O`：阀门扭矩清零

---

### 1.5 阀门状态实时可视化
已增加独立的实时波形窗口，用于显示当前阀门状态，包含三项曲线：

- `Valve Torque Cmd`
- `Valve Angle`
- `Valve Vel`

该波形窗口由独立进程绘制，不影响仿真主线程。

对应主要文件：
- `sim2real/sim_env/loco_manip.py`

---

### 1.6 自动启动脚本
已编写自动化脚本，用于完成整个 demo 启动流程，包括：

1. 清理旧的 `sim_env` / `policy` 进程
2. 打开 MuJoCo sim_env 终端
3. 自动启动 `sim_env`
4. 自动向 MuJoCo 窗口发送 6 次数字 `8`，使机器人从悬挂状态落地
5. 打开 policy 终端
6. 自动启动 policy
7. 自动执行后续按键序列：
   - policy 终端发送 `]`
   - MuJoCo 窗口发送 `9`
   - MuJoCo 窗口发送 `j`

对应脚本：
- `launch_valve_demo_auto_v2.sh`
- 或后续更新版本脚本

---

## 2. 主要改动文件

### 场景与模型
- `humanoidverse/data/robots/g1/scene_g1_29dof_freebase.xml`
- `humanoidverse/data/robots/g1/valve_body_include.xml`

### 仿真逻辑
- `sim2real/sim_env/loco_manip.py`

### 吊绳参数
- `sim2real/utils/sdk2py_bridge/base/basic_sdk2py_bridge.py`

### 配置文件
- `sim2real/config/g1/g1_29dof_falcon.yaml`

### 自动脚本
- `launch_valve_demo_auto_v2.sh`
- 相关启动脚本版本

---

## 3. 启动前环境准备

每次启动前，建议先进入环境并清理 Python 路径污染：

```bash
conda activate fcreal
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

---

## 4. 手动启动方式

### 4.1 启动 MuJoCo 环境
在 `sim2real/` 目录下执行：

```bash
python sim_env/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml
```

---

### 4.2 启动策略
在另一个终端中执行：

```bash
python rl_policy/loco_manip/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml --model_path=models/falcon/g1_29dof.onnx
```
conda activate fcreal
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python rl_policy/loco_manip/loco_manip_valve_turn.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
---

## 5. 自动脚本使用方式

当前推荐直接使用自动脚本启动完整流程。

### 5.1 运行脚本
```bash
chmod +x launch_valve_demo_auto_v2.sh
./launch_valve_demo_auto_v2.sh
```

---

### 5.2 脚本自动完成的流程
脚本会自动完成以下步骤：

1. 清理残留后台进程
2. 启动 `sim_env`
3. 等待 MuJoCo 窗口弹出
4. 自动向 MuJoCo 发送 6 次数字 `8`
5. 等待机器人脚部着地
6. 启动 `policy`
7. 自动发送：
   - `]`
   - `9`
   - `j`

这样可以直接进入阀门主动旋转测试流程。

---

## 6. 模型切换方式

自动脚本支持通过 `MODEL_PATH` 覆盖当前策略模型。

例如切换到 `model_10000.onnx`：

```bash
MODEL_PATH="models/falcon/model_10000.onnx" ./launch_valve_demo_auto_v2.sh
```

说明：
- `CONFIG_PATH` 决定使用哪个 yaml 配置文件
- `MODEL_PATH` 决定最终加载哪个 onnx 模型
- 当前启动链路中，脚本传入的 `MODEL_PATH` 优先级高于 yaml 中的 `model_path`

因此更换模型时，通常只需要修改：

```bash
MODEL_PATH="..."
```

即可。

---

## 7. 自动脚本常用可调参数

除了 `MODEL_PATH`，脚本还支持以下常用参数覆盖：

### 配置文件
```bash
CONFIG_PATH="config/g1/g1_29dof_falcon.yaml"
```

### MuJoCo 窗口名
```bash
MUJOCO_WINDOW_NAME="MuJoCo"
```

### 自动落地按键次数
```bash
PRESS_COUNT=6
```

### 自动落地按键间隔
```bash
PRESS_INTERVAL_SEC=0.35
```

### 落地后等待时间
```bash
WAIT_AFTER_PRESS_SEC=2.0
```

示例：

```bash
MODEL_PATH="models/falcon/model_10000.onnx" PRESS_INTERVAL_SEC=0.5 WAIT_AFTER_PRESS_SEC=4.0 ./launch_valve_demo_auto_v2.sh
```

---

## 8. 当前演示使用流程

推荐的实际演示流程如下：

1. 确认环境：
   ```bash
   conda activate fcreal
   unset PYTHONPATH
   export PYTHONNOUSERSITE=1
   ```

2. 运行自动脚本：
   ```bash
   ./launch_valve_demo_auto_v2.sh
   ```

3. 脚本自动完成：
   - MuJoCo 启动
   - 自动落地
   - policy 启动
   - 后续任务按键发送

4. 在 MuJoCo 中通过键盘手动控制：
   - `J`：建立/解除固连
   - `K`：开始/停止阀门主动扭转
   - `U/I/O`：调节阀门扭矩

5. 结合实时波形窗口观察：
   - 扭矩输出
   - 阀门角度
   - 阀门角速度

---

## 9. 当前版本目标

当前版本主要用于完成以下演示目标：

- 机器人与阀门在 MuJoCo 中完成 sim2sim 对接
- 机器人在接触后固连条件下进行阀门抗扭测试
- 阀门主动旋转，机器人进行受扰稳定控制
- 支持不同策略模型快速切换与对比测试

该版本已经能够支撑论文第四章中 MuJoCo 阀门场景补充验证的 demo 录制。
