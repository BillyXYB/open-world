"""Resume of ``droid_wm_uq_bootstrap_v1.py`` after an external OOM-kill.

The original run (tag ``droid_flow_matching_uq_bootstrap_v1``) was killed by
the kernel OOM-killer (rank 0 received a raw SIGKILL with no Python
traceback -- not a CUDA OOM, which raises a catchable exception) at step
186993/220000 on the shared, non-SLURM-managed della-ani node. Training
itself never errored; this is purely an infra interruption.

This config resumes from the last saved checkpoint (checkpoint-180000.pt,
the most recent ``checkpointing_steps=10_000`` boundary before the kill) via
a bare state_dict load for the model PLUS the matching optimizer state
(optimizer-180000.pt), so Adam's momentum/variance carry over instead of
restarting cold -- see train_wm.py: ckpt_path loads model weights,
optimizer_state_path loads optimizer state, independently.

IMPORTANT: train_wm.py's global_step always starts at 0 regardless of
ckpt_path/optimizer_state_path (see its loop setup) -- there is no resume of
the step counter itself. To finish the original 220_000-step budget without
re-running steps 0-180000, max_train_steps is set to 40_000 (the remainder:
220_000 - 180_000) here, and the output_dir/tag are kept DISTINCT from the
original run so this run's checkpoint-10000.pt (real step 190000) does not
collide with and overwrite the original run's own checkpoint-10000.pt (real
step 10000). This mirrors the existing warm-start pattern used by
droid_wm_uq_bootstrap_only_v1.py (separate tag, separate output_dir, bare
state_dict load from another run's checkpoint).

Everything else (architecture, dataset, distractor mix, schedule knobs other
than max_train_steps) is unchanged from droid_wm_uq_bootstrap_v1.py -- this
is a continuation, not a new experiment.
"""

import os

from openworld.training.world_model.config import LiberoWMArgs

_ORIG_CKPT_DIR = "checkpoints/wm_droid/droid_flow_matching_uq_bootstrap_v1"
_RESUME_STEP = 180_000
_TARGET_STEP = 220_000


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/data"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        # Resume weights + optimizer state from the last checkpoint before
        # the OOM-kill -- see module docstring.
        ckpt_path=os.path.join(_ORIG_CKPT_DIR, f"checkpoint-{_RESUME_STEP}.pt"),
        optimizer_state_path=os.path.join(_ORIG_CKPT_DIR, f"optimizer-{_RESUME_STEP}.pt"),

        # ----- Dataset: reuse vidwm's existing droid_ctrl_world data -----
        dataset_root_path=data_root,
        dataset_meta_info_path=os.path.join(data_root, "dataset_meta_info"),
        dataset_names="droid_ctrl_world",
        dataset_cfgs="dataset_meta_info/droid_ctrl_world",
        prob=(1.0,),
        annotation_name="annotation",

        # ----- Compute (unchanged from the original run) -----
        train_batch_size=3,
        gradient_accumulation_steps=2,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule -----
        learning_rate=1e-5,
        max_train_steps=_TARGET_STEP - _RESUME_STEP,  # 40_000 steps remaining -- see module docstring
        checkpointing_steps=10_000,
        validation_steps=10_000,
        max_grad_norm=1.0,

        # ----- Architecture (DROID-specific: 3 cams, 192x320, unchanged) -----
        num_cams=3,
        height=192,
        width=320,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=3,

        # ----- Loss / sampling defaults -----
        flow_map_type="flow_matching",
        distance_conditioning=False,

        # ----- UQ head -----
        predict_uncertainty=True,
        uncertainty_weight=0.01,

        # ----- History-future-overlap augmentation (unchanged) -----
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        history_overlap_noise_scale=0.3,
        fixed_overlap_k=True,

        # ----- Four-way future mix (unchanged from the original run) -----
        p_false_future=0.15,
        p_shifted_future=0.15,
        p_bootstrap_future=0.40,
        bootstrap_cache_root="outputs/bootstrap_futures_cache_droid",
        zero_overlap_action=True,

        # ----- Fixed-length masked peek (unchanged) -----
        mask_history_from_peek=True,

        # Distinct tag/output_dir from the original run -- see module
        # docstring on why this must NOT share droid_flow_matching_uq_bootstrap_v1's
        # checkpoint directory.
        tag="droid_flow_matching_uq_bootstrap_v1_resume",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
