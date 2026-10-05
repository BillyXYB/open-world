"""DROID world-model training config: bootstrap-ONLY ablation, warm-started
from ``droid_flow_matching_uq_false_future_v3``'s checkpoint-220000.pt.

Companion to ``droid_wm_uq_bootstrap_v1.py`` (the from-scratch, four-way-mix
run). That config still keeps v3's easy distractors (p_shifted_future,
p_false_future) alongside the new bootstrap one, as an in-run baseline
against the pre-bootstrap collapse behavior. This config isolates the
opposite end of the question instead: drop every distractor EXCEPT
bootstrap, so the overlap-peek content is ALWAYS either (a) nothing at all
("zero peek" -- the p_history_future_overlap=0.5 branch simply not firing,
unchanged from v3) or (b) the self-sampled bootstrap future
(p_bootstrap_future=1.0 -- see dataset.py's bootstrap_anchors_all: this
fires with the realized rate exactly matching the configured value, no
cache-miss dilution). No true-future peek, no shifted/other-episode peek at
all. This directly tests whether the logvar head can learn "peek present
(bootstrap) => should NOT be more confident than peek absent" as a clean
two-way contrast, without the easier other-episode/shifted-time distractors
around to let it fall back on a cheaper structural tell.

Unlike droid_wm_uq_bootstrap_v1.py, this config warm-starts from v3's
checkpoint-220000.pt (bare state_dict load -- see train_wm.py -- NOT a full
resume; global_step/optimizer/LR schedule restart at 0) and trains for a
short, Phase-B-style budget (max_train_steps=40_000) rather than matching
v3's own from-scratch duration: the goal here is a fast, isolated read on
the zero-vs-bootstrap contrast on top of v3's already-reasonable DROID
generation quality, not a fully independent training run.

Tag bumped to droid_flow_matching_uq_bootstrap_only_v1 so this run's
checkpoints land in their own directory, separate from both v3's and
droid_wm_uq_bootstrap_v1's.
"""

import os

from openworld.training.world_model.config import LiberoWMArgs

_V3_CKPT_DIR = "checkpoints/wm_droid/droid_flow_matching_uq_false_future_v3"


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/data"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        # Warm start from v3's final checkpoint -- bare state_dict load (see
        # train_wm.py), NOT a full resume. See module docstring.
        ckpt_path=os.path.join(_V3_CKPT_DIR, "checkpoint-220000.pt"),

        # ----- Dataset: reuse vidwm's existing droid_ctrl_world data -----
        dataset_root_path=data_root,
        dataset_meta_info_path=os.path.join(data_root, "dataset_meta_info"),
        dataset_names="droid_ctrl_world",
        dataset_cfgs="dataset_meta_info/droid_ctrl_world",
        prob=(1.0,),
        annotation_name="annotation",

        # ----- Compute (unchanged from v3) -----
        train_batch_size=3,
        gradient_accumulation_steps=2,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule -----
        learning_rate=1e-5,
        max_train_steps=40_000,   # Phase-B-style short budget -- see module docstring
        checkpointing_steps=10_000,
        validation_steps=10_000,
        max_grad_norm=1.0,

        # ----- Architecture (DROID-specific: 3 cams, 192x320, unchanged from v3) -----
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

        # ----- History-future-overlap augmentation (unchanged from v3) -----
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        history_overlap_noise_scale=0.3,
        fixed_overlap_k=True,

        # ----- Bootstrap-ONLY peek content -- see module docstring -----
        p_false_future=0.0,
        p_shifted_future=0.0,
        p_bootstrap_future=1.0,   # every firing peek is the bootstrap future; no true-future peek
        bootstrap_cache_root="outputs/bootstrap_futures_cache_droid",
        zero_overlap_action=True,

        # ----- Fixed-length masked peek (unchanged from v3) -----
        mask_history_from_peek=True,

        tag="droid_flow_matching_uq_bootstrap_only_v1",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
