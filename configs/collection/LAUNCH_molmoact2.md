# Launching active-UQ DROID data collection with the MolmoAct2 policy

Same flow as the Pi0.5 runbook, plus **one extra process** (the MolmoAct2 inference
server) that must be up **before** the open-world server, and everything server-side moves
from `della-ani` to **`della-gpu`** (the H200 where `~/molmoact/` + the checkpoint live).

```
NUC robot server ──(robot LAN)── robot laptop ──SSH tunnel :8765── della-gpu
                                                                      ├─ open-world server  (GPU 0, :8765)
                                                                      └─ MolmoAct2 serve.py (GPU 0, :9999, localhost)
```

Order matters: **NUC → MolmoAct2 serve.py → open-world server → tunnel → robot.**
The open-world server calls `policy.connect()` at startup and exits immediately if
`serve.py` isn't listening.

Prereqs (one-time, Phase B): `~/molmoact/` + `~/molmoact/checkpoints/MolmoAct2-DROID/` +
the `uv` venv at `$MA2/.venv` on della-gpu; the `open-world` repo's `uv` env and WM
checkpoints working on della-gpu (same as your Pi0.5 setup). See the plan file
`~/.claude/plans/here-was-how-i-async-forest.md` §B1–B2.

---

## 1. Start the robot server (NUC) — unchanged

```bash
ssh 172.16.0.5            # password is " "
tmux new -s robot_server
conda activate polymetis-local
cd droid
python scripts/server/run_server.py
# detach: Ctrl-b d
```

## 1.1 (optional) Clean up stale tunnels on the robot laptop — unchanged

```bash
pkill -f "8765:localhost:8765"
```

## 2. Start the MolmoAct2 inference server on della-gpu — NEW

```bash
ssh -J yx2653@tigressgateway.princeton.edu yx2653@della-gpu.princeton.edu
tmux new -s ma2_server

MA2=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/molmoact
export CUDA_VISIBLE_DEVICES=0            # same GPU as the world model (H200 has room)

$MA2/.venv/bin/python $MA2/serve.py \
    --checkpoint $MA2/checkpoints/MolmoAct2-DROID \
    --port 9999 --device cuda --warmup_candidates 2,4
# detach: Ctrl-b d
```

**Wait for `Listening on 127.0.0.1:9999 -- waiting for DROID client ...`** (after a
`~5-10 s` load + `Warm-up complete`). **`--warmup_candidates` MUST include this
run's `active_uq.num_candidates`** — **2** for the policy_only configs
(`droid_hardware_active_uq_molmoact2.yaml`, `..._policy_only.yaml`), **4** for
`..._false_future.yaml`. The action-flow CUDA graph is captured per n_candidates;
a value that isn't warmed silently drops that run to the ~4x-slower eager path.
The comma list warms both so one server serves either config.

## 3. Start the open-world collection server on della-gpu

```bash
tmux new -s uq_server            # or a second window in ma2_server
cd /scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world
source scripts/setup.bash
export CUDA_VISIBLE_DEVICES=0

uv run python scripts/run_droid_hardware_active_uq.py \
    --config configs/collection/droid_hardware_active_uq_molmoact2.yaml
```

Startup logs to expect, in order:
- `Connecting to MolmoAct2 server at 127.0.0.1:9999` → `MolmoAct2 server connected.`
  (if it hangs/errors here, step 2 isn't actually listening)
- `Loading UQ world model from checkpoints/wm_droid/...`
- **`Socket comms server listening on 127.0.0.1:8765`** ← wait for this before step 4.

## 4. Open a fresh tunnel on the robot laptop — della-gpu, not della-ani

```bash
ssh -N -L 8765:localhost:8765 -J yx2653@tigressgateway.princeton.edu yx2653@della-gpu.princeton.edu
```

## 5. Launch the robot — unchanged

```bash
cd /home/tennyyin/projects/irom/UQ_Data_Collection/uq_data_collection
conda activate robot

python examples/droid/main.py \
    --external_camera right --num_trajectories_to_collect 1 \
    --comms_mode socket --remote_host localhost --remote_port 8765
```

Start with **1 trajectory** (matches `hardware.num_trajectories: 1` in the config) as the
pilot. After it completes, inspect
`data/droid_hardware_active_uq_collected/droid_ctrl_world_hardware_molmoact2_single_hist_v1/`:
- `decision_metrics.jsonl` — for each decision, is there real spread across the 8
  candidates in the `uq_metric` (`mean_pdf_diff`) and the other 6 metrics? Near-identical
  values ⇒ the WM isn't discriminating MolmoAct2 candidates; see the plan's R1.
- `candidate_viz/000001/decision_*_cam*.mp4` — the chosen vs. rejected predicted rollouts.
- episode `annotation/train/*.json` → `extra_annotation.policy_backend == "molmoact2"`,
  `mean_uq_score_spread`.

Then raise both `hardware.num_trajectories` (config) and `--num_trajectories_to_collect`
(robot) for a real run.

---

## Watch / debug

- MolmoAct2 server (`tmux a -t ma2_server`): logs `Step N | candidates[8, 15, 8]` every 50
  decisions. One decision ≈ 0.2–1 s.
- open-world server (`tmux a -t uq_server`): logs `[traj K] starting rollout`, per-decision
  progress, `[traj K] done: D decisions, R round-trips -> episode NNNNNN`.
- Robot refused connection ⇒ step 3 hadn't printed `Socket comms server listening` yet, or
  the tunnel (step 4) points at the wrong host.
- open-world server exits right after "Connecting to MolmoAct2 server" ⇒ `serve.py` (step
  2) isn't listening on `:9999`. `ss -ltn | grep 9999` on della-gpu to check.
- To swap back to Pi0.5: use `--config configs/collection/droid_hardware_active_uq.yaml`
  and skip step 2 (no separate policy server).
