# Diagnostic pilot run — MolmoAct2 "moves not as expected"

One `policy_only` trajectory with per-decision logging, to pin down whether the model's
output is bad on the live scene, or it's execution / latency.

Machines: **NUC** (robot server), **della-gpu** (MolmoAct2 + collection server),
**local** (robot control laptop). `MA2=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/molmoact`,
`OW=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world`.

---

## 0. Sync the code to della-gpu

**On the dev box** (where you edit — `mae-iromlab-2`):
```bash
cd ~/projects/irom/UQ_Data_Collection/open-world
git add -A && git commit -m "molmoact2: image_size + round-trip timing log" && git push origin libero

# serve.py is not in this repo -- rsync it (tiny; only if della-gpu's is older than Aug 27):
rsync -aP ~/molmoact2/serve.py <della-gpu>:$MA2/serve.py
```

**On della-gpu:**
```bash
cd $OW && git pull origin libero
# sanity: the three fixes are present
grep -n "image_size" openworld/policies/molmoact2_client.py | head -1
grep -n "round-trip" openworld/policies/molmoact2_client.py | head -1
grep -n "dense, NO current-prepend" scripts/run_droid_hardware_active_uq.py
grep -n "infer_candidate_chunks" $MA2/serve.py | head -1
```

---

## 1. Robot server — NUC (unchanged)

```bash
ssh 172.16.0.5            # password is " "
tmux new -s robot_server
conda activate polymetis-local
cd droid
python scripts/server/run_server.py       # Ctrl-b d to detach
```

## 1.1 Clean stale tunnels on the local machine

```bash
pkill -f "8765:localhost:8765"
```

## 2. MolmoAct2 server — della-gpu

```bash
ssh -J yx2653@tigressgateway.princeton.edu yx2653@della-gpu.princeton.edu
tmux new -s ma2_server
MA2=/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/molmoact
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $MA2/.venv/bin/python $MA2/serve.py \
    --checkpoint $MA2/checkpoints/MolmoAct2-DROID \
    --port 9999 --device cuda --warmup_candidates 2
# Ctrl-b d
```

Wait for **all** of:
- `MolmoAct2 internal API check passed.`
- `Warming up batched candidate path (n_candidates=2) ...`
- `Listening on 127.0.0.1:9999 -- waiting for DROID client ...`

If `--warmup_candidates` doesn't include `2`, each replan runs ~4x slower (eager path).

## 3. Collection server — della-gpu

```bash
tmux new -s uq_server            # or a second window
cd $OW
source scripts/setup.bash
export CUDA_VISIBLE_DEVICES=0

uv run python scripts/run_droid_hardware_active_uq.py \
    --config configs/collection/droid_hardware_active_uq_molmoact2.yaml
```

That config is `policy_only: true`, `num_candidates: 2`, `image_size: 480`,
`debug_dump_n: 5`, `num_trajectories: 1`. Expect, in order:
- `Connecting to MolmoAct2 server at 127.0.0.1:9999` -> `MolmoAct2 server connected.`
- `active_uq.policy_only=true -- skipping world model load entirely` (no WM)
- **`Socket comms server listening on 127.0.0.1:8765`** <- wait for this.

**Leave this terminal visible — the diagnostic lines print here.**

## 4. SSH tunnel — local machine

```bash
ssh -N -L 8765:localhost:8765 -J yx2653@tigressgateway.princeton.edu yx2653@della-gpu.princeton.edu
```

## 5. Robot — local machine

```bash
cd ~/projects/irom/UQ_Data_Collection/uq_data_collection
conda activate robot

python examples/droid/main.py \
    --external_camera right --num_trajectories_to_collect 1 \
    --comms_mode socket --remote_host localhost --remote_port 8765
```

(Optional, to remove the safety filter as a variable: in `examples/droid/main.py`
set `disable_look_at_boundary_test = True` (~line 470) and `enable_safety_filter = False`.)

Press Enter when prompted; let it run ~15-30 s.

---

## 6. Collect these (from the **della-gpu `uq_server` terminal**)

**a. `[ma2 dump 0..4]` lines** — paste the *"returned chunk[0] per-step joint drift"* list.
- values ~`0.0` all the way -> the model's output is broken on your live obs.
- ramps to ~`0.1-0.3` by step 14 -> the model is fine; it's execution/cadence.

**b. `[ma2] infer_candidates round-trip Xs` lines** — the della-gpu-local server time.
- ~`0.16-0.20 s` -> server fast.
- `>1 s` -> CUDA graph not hit (warmup mismatch) or WM wasn't skipped.

**c. The saved images** — `$OW/data/droid_hardware_active_uq_collected/ma2_debug/ma2_obs_000N.png`.
`scp` one back and eyeball: 3 panels (wrist | right-exterior | left-exterior), right scene,
not black / frozen / wrong camera.

**d. From the `main.py` terminal** — the tqdm rate (`N it/s`).
- ~`15` -> robot at full rate.
- ~`2-5` -> the `policy_client.infer()` round-trip over the tunnel is the bottleneck.

**e.** Does your **openpi** collection over the same tunnel run at ~15 it/s or also ~3-5?

Send a/b/c/d/e and we'll know which of {model output, della-gpu compute, tunnel bandwidth,
robot execution} is the problem.

## Reset after
Set `debug_dump_dir: null` / `debug_dump_n: 0` in the config before a real collection run.
