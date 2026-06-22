# Uncertainty-Quantification Data for LIBERO

Tooling for generating LIBERO sim data with **known ground-truth uncertainty**, so a
world model's (or policy's) predicted uncertainty can be *calibrated* against it.

There are two complementary generators:

| Script | Output | Use it for |
|---|---|---|
| `scripts/collect_libero_branch_sets.py` | **branch sets** — one shared history, many futures | per-task counterfactual sweeps; "how far do futures diverge under a hidden variable?" |
| `scripts/build_uq_dataset.py` | a **2×2 uncertainty-labelled dataset** | controlled aleatoric × epistemic training data with per-scenario uncertainty labels |

Both share the same physics/replay machinery and both write the standard WM data
format (identical to `preprocess_libero_for_wm.py`), so the output is drop-in for
world-model training.

## Setup

```bash
git -C external/openpi submodule update --init third_party/libero
uv sync --extra libero

# LIBERO human demos (the shared-history / replay source). Once.
uv run python external/openpi/third_party/libero/benchmark_scripts/download_libero_datasets.py \
    --datasets libero_goal libero_object libero_spatial
```

Everything below renders off-screen on the GPU, so prefix runs with:

```bash
export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0
```

---

## Core idea

The uncertainty in a robot task decomposes into two kinds, instantiated by two
different mechanisms:

- **Aleatoric (variance, irreducible)** — how widely the *outcome* spreads when an
  **unobserved physical variable** is drawn at random. Here the hidden variable is
  the manipulated object's **sliding friction** `mu`. An object whose final position
  is very sensitive to `mu` is high-variance; one that barely moves is low-variance.
  This is **measured** from rollouts, never assumed.

- **Epistemic (data, reducible)** — how much the **model has seen** an object.
  Realized purely by **training frequency**: an object that appears in 100 episodes
  is well-known; one in 10 is not. Epistemic uncertainty ≈ `sigma / sqrt(N)`.

The whole point of the hidden-variable design: the **history is shared** across
futures up to the moment of contact (the robot is kinematically pinned to a real
demo, so its motion is byte-identical across branches), and only the future diverges
once friction bites. So divergence is attributable to the hidden variable alone.

---

## 1. Branch sets — `collect_libero_branch_sets.py`

A **branch set** = one LIBERO human demonstration + a fan of counterfactual futures
that replay the demo's exact actions under a swept hidden variable.

- **branch 0** — the demonstration: full-state playback of the recorded demo
  (reproduces the original successful trajectory, `is_demonstration=True`).
- **branch 1..N** — counterfactuals: robot pinned to the demo's joint trajectory,
  object responds to a swept `mu` (friction) and/or mass scale.

```bash
# one task, friction sweep
uv run --no-sync python scripts/collect_libero_branch_sets.py \
    --suite libero_goal --task-id 5 \
    --mu-sweep 0.05 0.1 0.2 0.4 0.8 \
    --out data/libero_branch_sets

# all tasks in a suite, one demo each
uv run --no-sync python scripts/collect_libero_branch_sets.py \
    --suite libero_goal --task-ids -1 --num-branch-sets 1 \
    --mu-sweep 0.05 0.1 0.2 0.4 --res 96 \
    --out data/libero_branch_sets

# mass counterfactuals instead of / in addition to friction
uv run --no-sync python scripts/collect_libero_branch_sets.py \
    --suite libero_goal --task-id 5 \
    --mass-sweep 0.5 1.0 2.0 4.0 --mass-base-mu 0.1
```

Key flags:

| Flag | Meaning |
|---|---|
| `--suite` | LIBERO suite (`libero_goal`, `libero_object`, `libero_spatial`, `libero_10`) |
| `--task-id` / `--task-ids` | single task, or a list; `--task-ids -1` = all tasks in the suite |
| `--num-branch-sets` | demos per task (each demo = one branch set) |
| `--mu-sweep` | counterfactual sliding frictions (the hidden variable) |
| `--mass-sweep` | counterfactual mass-scale factors (off by default) |
| `--object-body` | manipulated object body; default = auto-detect (free-joint body that moves most in the demo) |
| `--min-demo-disp` | skip tasks whose object moves less than this (m) — no slideable object |
| `--tint-by-mu` / `--no-tint-by-mu` | colour the object a distinct hue per friction value (on by default) so each branch is visually identifiable |
| `--res`, `--fps` | render resolution / fps |
| `--encode-latents` | also write SVD latents + a WM sample index (needs `--svd-path`) |

The manipulated object is **auto-detected** (the free-joint body with the largest
demo displacement). Tasks with no slideable object (knob/button) are skipped.

### Output layout

```
<out>/<suite>/raw_videos/{agentview,wrist}/<eid>.mp4   sim renders
<out>/<suite>/states/<eid>.npz                          eef/object/actions/success
<out>/<suite>/annotation/train/<eid>.json              tags + metadata
<out>/<suite>/latent_videos/.../<eid>.pt               (with --encode-latents)
<out>/branch_manifest.json                              per-set / per-task summary
```

Each episode's annotation carries `branch_set_id`, `branch_id`, `is_demonstration`,
`hidden_var`, `mu` / `mass_scale`, `is_success`, `contact_frame_idx`, and
`object_displacement_m`. `branch_manifest.json` adds the per-set **outcome spread**
(`disp_spread_cm`) and success count — a direct aleatoric-variance readout per task.

---

## 2. The 2×2 dataset — `build_uq_dataset.py`

Builds a dataset whose scenarios have **labelled** uncertainty, crossing the two
axes above into a 2×2 grid. Each object is sampled at a controlled frequency and
rendered with a fixed colour (texture) identifying its set.

Driven by a registry YAML (`configs/uq/objects_2x2.yaml`). Runs in three phases:

1. **Measure variance** — friction probe sweep per object → `sigma_cm` (std of final
   object displacement). Blow-ups (object ejected) above `max_disp_cm` are clipped
   and flagged, so a degenerate sim doesn't masquerade as high variance.
2. **Assign counts** — `data_level` → number of episodes (the epistemic knob).
3. **Generate** — for each object, draw N episodes, each a *distinct* draw (`mu`
   sampled from the hidden-var distribution; demos cycled), so more data = more
   *coverage*, not repetition.

```bash
# preview the sampling plan (measures variance, generates nothing)
uv run --no-sync python scripts/build_uq_dataset.py \
    --config configs/uq/objects_2x2.yaml --plan-only

# generate
uv run --no-sync python scripts/build_uq_dataset.py \
    --config configs/uq/objects_2x2.yaml --out data/uq_2x2

# cheap dry run: scale every count down (e.g. 0.05)
uv run --no-sync python scripts/build_uq_dataset.py \
    --config configs/uq/objects_2x2.yaml --out /tmp/uq_test --scale 0.05

# also write SVD latents + WM sample list
uv run --no-sync python scripts/build_uq_dataset.py \
    --config configs/uq/objects_2x2.yaml --out data/uq_2x2 --encode-latents
```

### Registry (`configs/uq/objects_2x2.yaml`)

```yaml
hidden_var: {name: object_sliding_friction, distribution: loguniform, low: 0.03, high: 0.6}
data_levels: {large: 100, small: 10}      # episodes per object, per level
count_mode: flat                          # flat | coverage  (see below)
variance_probe: {num_mu: 5, split: median, threshold_cm: 6.0, max_disp_cm: 60.0}

objects:
  - {name: salad_hi_large,  suite: libero_object, task_id: 2, variance_level: high, data_level: large, color: [0.95,0.80,0.10]}
  - {name: soup_hi_small,   suite: libero_object, task_id: 0, variance_level: high, data_level: small, color: [0.90,0.15,0.15]}
  - {name: cheese_lo_large, suite: libero_object, task_id: 1, variance_level: low,  data_level: large, color: [0.20,0.40,1.00]}
  - {name: tomato_lo_small, suite: libero_object, task_id: 5, variance_level: low,  data_level: small, color: [0.20,0.85,0.30]}
```

- **`variance_level`** — `high` / `low` to pin, or `auto` to bin by the measured
  `sigma` (median or fixed threshold via `variance_probe.split`).
- **`data_level`** — selects the episode count from `data_levels`. Assign it
  *independently* of variance, or your cells collapse onto the diagonal.
- **`color`** — fixed per-object RGB tint; omit to auto-assign a hue per cell.

### `count_mode`: flat vs coverage

This is the one subtlety that decides whether the 2×2 corners are clean:

- **`flat`** — `N = data_levels[level]` literally (100 vs 10). Simple, but the axes
  **interact**: 10 draws is "small" for a high-variance object yet "plenty" for a
  low-variance one (epistemic ≈ `sigma/sqrt(N)`).
- **`coverage`** — `N = base * (sigma / ref_sigma_cm)**2`, clamped to
  `[min_count, max_count]`. Makes epistemic depend *only* on the data level, so the
  variance and data axes stay independent (clean corners) — at the cost that
  low-variance objects get few episodes (they're already well-covered).

### Output layout

Standard WM format under `<out>/<suite>/` (`raw_videos/`, `states/`,
`annotation/train/`, optional `latent_videos/`), plus `<out>/uq_manifest.json` with
per-object `sigma_cm`, `N`, success counts, and per-cell tallies. Every episode's
annotation carries `uncertainty_cell` (e.g. `high_var/large_data`), `variance_level`,
`data_level`, `mu`, and `sigma_cm` — the ground-truth labels for calibration.

---

## Caveats

- **Epistemic only counts if it's in *training*.** The 100-vs-10 frequencies are a
  property of the dataset the model *trains on*. Evaluate on **held-out** instances —
  don't reuse these exact episodes as the eval set, or there is no epistemic gap.
- **Make draws diverse.** N copies of the same trajectory carry the information of
  one. The generator samples a fresh `mu` per episode (and cycles demos) for exactly
  this reason; keep the hidden-var range wide enough to matter.
- **The hidden variable must be unobservable to the model.** If `mu` is fed in, the
  spread is no longer uncertainty.
- **Validate variance bins empirically.** Use `--plan-only` to read the measured
  `sigma` ladder before committing — don't assume which objects are high/low. The
  LIBERO object suite, for instance, has few naturally low-variance objects.
- **Watch for the texture shortcut.** If a colour deterministically encodes a cell, a
  model can read uncertainty off the colour instead of reasoning. Fine for pure
  calibration tests; for generalization, counterbalance the texture↔cell assignment
  and hold out some textures.
