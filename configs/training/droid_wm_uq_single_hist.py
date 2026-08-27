"""DROID world-model training config: UQ head + single-history augmentation.

DROID analog of libero_wm_uq_single_hist.py. Extends droid_wm_uq.py by
enabling the single-history augmentation (p_single_history=0.5). With 50%
probability the full history window is replaced by the most-recent history
frame repeated for all slots ([h6, h6, ..., h6] instead of [h1, h2, ..., h6]).
This closes the train/test gap for scenarios where only the immediately
preceding frame is available as context.

v1: fixes down_sample (1 -> 3) — see droid_wm_uq.py's module docstring for
why. Tag bumped v0 -> v1 for consistency with the other DROID configs (this
one was never actually trained under v0, so there's no mixing risk, but v1
uniformly means "trained with the down_sample=3 fix" across all three).
"""

import os

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/data"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path="/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/checkpoints/wm_droid/droid_flow_matching_uq_single_hist_v1/checkpoint-20000.pt",  # no compatible open-world-trained DROID checkpoint

        # ----- Dataset: reuse vidwm's existing droid_ctrl_world data -----
        dataset_root_path=data_root,
        dataset_meta_info_path=os.path.join(data_root, "dataset_meta_info"),
        dataset_names="droid_ctrl_world",
        dataset_cfgs="dataset_meta_info/droid_ctrl_world",
        prob=(1.0,),
        annotation_name="annotation",

        # ----- Compute -----
        train_batch_size=4,
        gradient_accumulation_steps=2,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule -----
        learning_rate=1e-5,
        max_train_steps=500_000,
        checkpointing_steps=10_000,
        validation_steps=10_000,
        max_grad_norm=1.0,

        # ----- Architecture (DROID-specific: 3 cams, 192x320) -----
        num_cams=3,
        height=192,
        width=320,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=3,  # see module docstring: droid_ctrl_world state is at 3x the latent rate

        # ----- Loss / sampling defaults -----
        flow_map_type="flow_matching",
        distance_conditioning=False,

        # ----- UQ head -----
        predict_uncertainty=True,
        uncertainty_weight=0.01,

        # ----- Single-history augmentation -----
        # 50% of training samples replace full history with the most-recent frame repeated.
        p_single_history=0.5,

        tag="droid_flow_matching_uq_single_hist_v1",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
