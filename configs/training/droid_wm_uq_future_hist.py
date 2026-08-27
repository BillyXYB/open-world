"""DROID world-model training config: UQ head + future-in-history augmentation.

DROID analog of libero_wm_uq_future_hist.py. Extends droid_wm_uq.py by
enabling the future-in-history augmentation (p_future_in_history=0.5). With
50% probability the training anchor is shifted forward by a random delta so
that recent history slots contain GT frames from what would have been the
prior chunk's prediction window. This closes the train/test gap for
iterative (self-consistency) epistemic UQ, where rolled2 history includes
model-predicted frames rather than pure GT frames.

v1: fixes down_sample (1 -> 3) — see droid_wm_uq.py's module docstring for
why. Tag bumped v0 -> v1 so this fix's checkpoints land in a fresh directory
instead of mixing with the old, mis-aligned v0 checkpoints (already trained
to checkpoint-100000.pt under checkpoints/wm_droid/droid_flow_matching_uq_future_hist_v0/).
"""

import os

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/data"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path=None,  # no compatible open-world-trained DROID checkpoint yet

        # ----- Dataset: reuse vidwm's existing droid_ctrl_world data -----
        dataset_root_path=data_root,
        dataset_meta_info_path=os.path.join(data_root, "dataset_meta_info"),
        dataset_names="droid_ctrl_world",
        dataset_cfgs="dataset_meta_info/droid_ctrl_world",
        prob=(1.0,),
        annotation_name="annotation",

        # ----- Compute -----
        train_batch_size=4,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule -----
        learning_rate=1e-5,
        max_train_steps=500_000,
        checkpointing_steps=10_000,
        validation_steps=5_000,
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

        # ----- Future-in-history augmentation -----
        # 50% of training samples use a forward-shifted anchor so that recent
        # history slots contain GT frames from the prior prediction window.
        p_future_in_history=0.5,

        tag="droid_flow_matching_uq_future_hist_v1",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
