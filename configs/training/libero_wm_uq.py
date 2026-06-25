"""LIBERO world-model training config with uncertainty quantification (UQ) head.

Extends libero_wm.py by enabling the per-pixel aleatoric uncertainty head
(predict_uncertainty=True). The UQ head adds a parallel conv_out_logvar branch
inside the UNet that predicts log-variance, trained with an NLL auxiliary loss
weighted by uncertainty_weight.
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
        checkpointing_steps=2_000,
        validation_steps=2_000,
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

        tag="libero_flow_matching_uq_v0",
    )
    return args
