"""LIBERO world-model training config: UQ head + future-in-history augmentation.

Extends libero_wm_uq.py by enabling the future-in-history augmentation
(p_future_in_history=0.5).  With 50% probability the training anchor is
shifted forward by a random delta so that recent history slots contain GT
frames from what would have been the prior chunk's prediction window.  This
closes the train/test gap for iterative (self-consistency) epistemic UQ, where
rolled2 history includes model-predicted frames rather than pure GT frames.
"""

import os

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/open-world/data/"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path=None,  # set to e.g. checkpoints/wm/checkpoint-120000.pt to warm-start

        # ----- Dataset -----
        dataset_root_path=os.path.join(data_root, "wm_training/libero_processed_5hz"),
        dataset_meta_info_path=os.path.join(data_root, "wm_training/libero_processed_5hz"),
        dataset_names="libero_spatial+libero_object+libero_goal+libero_10",
        dataset_cfgs="libero_spatial+libero_object+libero_goal+libero_10",
        prob=(0.25, 0.25, 0.25, 0.25),

        # ----- Compute -----
        train_batch_size=4,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule -----
        learning_rate=1e-5,
        max_train_steps=500_000,
        checkpointing_steps=5_000,
        validation_steps=5_000,
        max_grad_norm=1.0,

        # ----- Architecture (LIBERO-specific) -----
        num_cams=2,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=1,

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

        tag="libero_flow_matching_uq_future_hist_v0",
    )
    return args
