"""DROID world-model training config with uncertainty quantification (UQ) head.

DROID analog of libero_wm_uq.py. Trains the same CrtlWorld flow-matching
video model (with the per-pixel aleatoric uncertainty head,
predict_uncertainty=True) but reads the DROID dataset instead of LIBERO.

Reuses the same pre-encoded droid_ctrl_world data that
svd_ac_video_model/vidwm's DroidDataset already trains on
(/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world). No dataset-loader changes
are needed: LiberoLatentDataset was written to mirror the same on-disk schema
(annotation/{split}/{episode_id}.json + {split}_sample.json + stat.json +
pre-encoded per-camera VAE latents), just parameterized by num_cams/height/
width instead of hardcoding DROID's 3-camera, 192x320 layout.

v1: fixes down_sample (1 -> 3). droid_ctrl_world's action/state arrays
(observation.state.cartesian_position/gripper_position) are stored at the
raw ~15 Hz control rate, while the pre-encoded latents are at the 5 Hz WM
rate (raw_length/video_length ~= 3.0 across sampled episodes, matching
vidwm's own datasets/droid/droid.yaml `downsample: 3`). With down_sample=1,
LiberoLatentDataset read action state at the wrong temporal offset relative
to its paired latent frame. Tag bumped v0 -> v1 so this fix's checkpoints
land in a fresh directory instead of mixing with the old, mis-aligned v0
checkpoints.
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
        checkpointing_steps=2_000,
        validation_steps=2_000,
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

        tag="droid_flow_matching_uq_v1",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
