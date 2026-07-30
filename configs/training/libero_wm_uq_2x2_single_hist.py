"""LIBERO world-model training config: UQ head + single-history augmentation, on UQ 2x2 data.

Same architecture as libero_wm_uq_single_hist.py (UQ head + p_single_history=0.5)
but the dataset points to the 2x2 UQ benchmark collection in data/libero_uq_2x2.

The 2x2 UQ data has episodes annotated with:
  variance_level: high | low   (measured via friction probe)
  data_level:     large | small (number of friction environments per task)
  uncertainty_cell: e.g. "high_var/large_data"
  mu: friction coefficient used for that episode
  sigma_cm: task-level friction sensitivity in cm

After training, evaluate UQ metrics per cell with:
  scripts/replay_libero_wm_traj.py + scripts/aggregate_uq_by_cell.py
"""

from openworld.training.world_model.config import LiberoWMArgs

_UQ_ROOT = "/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/data/libero_uq_2x2"


def get_args() -> LiberoWMArgs:
    args = LiberoWMArgs(
        # ----- Paths -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path=None,

        # ----- Dataset -----
        dataset_root_path=_UQ_ROOT,
        dataset_meta_info_path=_UQ_ROOT,
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

        # ----- Single-history augmentation -----
        # 50% of training samples replace full history with the most-recent frame repeated.
        p_single_history=0.5,

        tag="libero_wm_uq_2x2_single_hist_v0",
    )
    return args
