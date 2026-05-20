# G1 A/B Test Worksite Snapshot

## 1. Metadata

- Record time: `<YYYY-MM-DD HH:MM:SS timezone>`
- Operator: `<记录人或 unknown>`
- Robot tag: `<robot_tag>`
- Robot role: `<robot_role>`
- Test type: `<test_type>`
- Notes: `<notes>`
- Physical setup:
  - battery: `<unknown / not applicable / 实测说明>`
  - rope state: `<unknown / not applicable / 实测说明>`
  - floor: `<unknown / not applicable / 实测说明>`
  - robot posture: `<unknown / not applicable / 实测说明>`
  - valve contact: `<unknown / not applicable / 实测说明>`
  - hand connected: `<unknown / not applicable / 实测说明>`
  - RealSense connected: `<unknown / not applicable / 实测说明>`
- Safety status:
  - remote controller available: `<unknown / not applicable / 实测说明>`
  - damping tested: `<unknown / not applicable / 实测说明>`
  - emergency operator: `<unknown / not applicable / 实测说明>`
  - no-contact test only: `<yes / no；说明是否只记录代码现场>`

如果本次没有实机测试，以上实机相关字段写 `unknown` 或 `not applicable`，并明确说明：本次只记录代码现场，未执行实机。

## 2. Exact command under test

记录当前标准实机部署命令：

```bash
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon_real.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

## 3. Repository state

- Repository root: `<repo path>`
- Branch: `<git branch>`
- HEAD: `<git commit hash>`
- Git status source: `git status --short`
- Git diff stat source: `git diff --stat`

说明：这里记录的是创建本次现场记录文件前的代码现场，用于之后和实机结果对比。

```text
<git status --short output>
```

```text
<git diff --stat output>
```

## 4. Deployment-relevant paths and hashes

说明：hash 使用 `git hash-object` 读取文件内容得到，属于 Git blob hash，不是 `sha256sum`。

- Policy entry path: `<path>`
  - git hash-object: `<hash>`
- Config path: `<path>`
  - git hash-object: `<hash>`
- Model path: `<path>`
  - git hash-object: `<hash>`

## 5. Execution constraints for this record

- 未运行机器人策略。
- 未连接 G1。
- 未发 DDS。
- 未调用 `MotionSwitcher.ReleaseMode()`。
- 未向 `rt/lowcmd` 发布。
- 未修改任何已有控制代码。
- 未创建自动化 logger 脚本。

## 6. Read-only commands used

```bash
<只读命令列表>
```
