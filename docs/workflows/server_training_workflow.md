# Server Training Workflow

This workflow keeps code review local, while running longer IsaacGym training
jobs manually on the server.

Server SSH alias:

```sshconfig
Host lab-tang-u2204
    HostName 114.214.211.226
    User ubuntu
    Port 6142
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Goals

- Local Codex edits code, scripts, and docs.
- For quick iteration, local working tree can be synced directly to the server.
- GitHub branch push/pull remains optional when a clean history is useful.
- Training runs manually in `tmux` on the server.
- Logs and checkpoints stay out of git.
- Artifacts are copied back with `rsync`/`scp` for local analysis.

## Safety Rules

Do not commit:

- `logs/`
- `runs/`
- `artifacts/`
- `*.pt`
- `*.onnx`
- TensorBoard event files
- MuJoCo valve demo outputs

The current `.gitignore` already ignores these categories. The uploaded
`model_10000.pt` checkpoint is intentionally ignored by git and should be copied
to the server separately.

## Simple Full-Worktree Sync

When the server should match the local working tree, use rsync instead of
carefully staging a Git branch. This copies local edited files and untracked
source/config/docs to the server, while excluding training outputs and caches.

Dry-run first:

```bash
DRY_RUN=1 \
REMOTE_HOST=lab-tang-u2204 \
REMOTE_REPO=/home/ubuntu/FALCON \
tools/server_training/sync_worktree_to_server
```

Then actually sync:

```bash
DRY_RUN=0 \
REMOTE_HOST=lab-tang-u2204 \
REMOTE_REPO=/home/ubuntu/FALCON \
tools/server_training/sync_worktree_to_server
```

By default this includes `*.pt`, so `/home/lavine/project/FALCON/model_10000.pt`
will be copied if present. To exclude checkpoints:

```bash
DRY_RUN=0 INCLUDE_CHECKPOINT=0 tools/server_training/sync_worktree_to_server
```

By default it does not delete extra files on the server. To mirror more
strictly, use `DELETE=1` only after a dry-run:

```bash
DRY_RUN=1 DELETE=1 tools/server_training/sync_worktree_to_server
DRY_RUN=0 DELETE=1 tools/server_training/sync_worktree_to_server
```

After sync, SSH in and inspect:

```bash
ssh lab-tang-u2204
cd /home/ubuntu/FALCON
git status --short
ls -lh model_10000.pt
```

## Local Code Flow

Check local branch and diff:

```bash
git branch --show-current
git status --short
git diff --stat
git diff
```

Create or switch to a training branch:

```bash
git switch -c train/6d-wrench-warmstart
```

Stage only source/config/docs needed for training:

```bash
git add \
  humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py \
  humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.py \
  humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_6d_wrench.yaml \
  humanoidverse/config/obs/dec_loco/g1_29dof_obs_6d_wrench_history_wolinvel_ma.yaml \
  humanoidverse/config/exp/decoupled_locomotion_stand_height_waist_wbc_6d_wrench_ma_ppo_ma_env.yaml \
  docs/TRAINING_6D_WRENCH_AUDIT.md \
  docs/worklogs/6d_wrench_curriculum_stability.md \
  docs/workflows/server_training_workflow.md \
  tools/server_training
```

Commit and push:

```bash
git commit -m "Add 6D wrench warm-start training path"
git push -u origin train/6d-wrench-warmstart
```

## Server Setup

SSH in:

```bash
ssh lab-tang-u2204
```

Clone once if needed:

```bash
git clone git@github.com:lavineee/FALCON.git ~/FALCON
```

Copy the trained FALCON checkpoint separately from local to server:

```bash
scp -P 6142 /home/lavine/project/FALCON/model_10000.pt \
  ubuntu@114.214.211.226:/home/ubuntu/FALCON/model_10000.pt
```

The checkpoint is ignored by git on both machines.

## Pull Branch On Server

From local, after pushing:

```bash
REMOTE_HOST=lab-tang-u2204 \
REMOTE_REPO=/home/ubuntu/FALCON \
BRANCH=train/6d-wrench-warmstart \
tools/server_training/remote_pull_branch
```

Equivalent manual server commands:

```bash
ssh lab-tang-u2204
cd ~/FALCON
git fetch origin train/6d-wrench-warmstart
git switch train/6d-wrench-warmstart
git pull --ff-only origin train/6d-wrench-warmstart
```

## Start Training In tmux

On the server:

```bash
cd ~/FALCON
SESSION=falcon_6d_256x100 \
CONDA_ENV=fcgym \
NUM_ENVS=256 \
ITERATIONS=100 \
SEED=1 \
CHECKPOINT=/home/ubuntu/FALCON/model_10000.pt \
tools/server_training/run_6d_wrench_warmstart_tmux
```

Attach:

```bash
tmux attach -t falcon_6d_256x100
```

Detach from tmux with `Ctrl-b d`.

The run writes under:

```text
logs/server_warmstart_6d_wrench/
```

## Pull Artifacts Back

After a server run completes, copy the run directory back into ignored local
artifacts:

```bash
REMOTE_HOST=lab-tang-u2204 \
REMOTE_PATH=/home/ubuntu/FALCON/logs/server_warmstart_6d_wrench/<run-dir> \
LOCAL_DIR=/home/lavine/project/FALCON/artifacts/server_runs \
tools/server_training/sync_artifacts_from_server
```

Dry-run first:

```bash
DRY_RUN=1 \
REMOTE_HOST=lab-tang-u2204 \
REMOTE_PATH=/home/ubuntu/FALCON/logs/server_warmstart_6d_wrench/<run-dir> \
LOCAL_DIR=/home/lavine/project/FALCON/artifacts/server_runs \
tools/server_training/sync_artifacts_from_server
```

## Recommended Progression

1. `256 env / 100-it` warm-start stability check.
2. Inspect TensorBoard scalars locally.
3. Only if stable, increase iterations.
4. Do not export ONNX until a separate evaluation phase is approved.
5. Do not run MuJoCo valve transfer until ONNX export is explicitly approved.
