"""LIBERO world-model training config: trained on UQ 2x2 benchmark data.

Same architecture as libero_wm_uq_future_hist.py (UQ head + future-in-history
augmentation) but the dataset points to the 2x2 UQ collection in
data/libero_uq_2x2 instead of the standard libero_processed_5hz baseline.

The 2x2 UQ data has episodes annotated with:
  variance_level: high | low   (measured via friction probe)
  data_level:     large | small (number of friction environments per task)
  uncertainty_cell: e.g. "high_var/large_data"
  mu: friction coefficient used for that episode
  sigma_cm: task-level friction sensitivity in cm

After training, use scripts/build_uq_test_manifest.py to build a
cell-grouped test manifest from the val split, then evaluate UQ metrics
per cell to verify the expected ranking:
  high_var/small_data > high_var/large_data >= low_var/small_data > low_var/large_data
"""

import os

from openworld.training.world_model.config import LiberoWMArgs

_UQ_ROOT = "/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/data/libero_uq_2x2"


def get_args() -> LiberoWMArgs:
    args = LiberoWMArgs(
        # ----- Paths -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path=None,

        # ----- Dataset -----
        # stat.json lives at <_UQ_ROOT>/stat.json (shared across all 4 suites).
        # Generate it once with scripts/build_uq_test_manifest.py or the
        # inline one-liner in jobs/train_wm_uq_2x2.sh.
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

        # ----- Architecture (LIBERO-specific, matches baseline) -----
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
        p_future_in_history=0.5,

        tag="libero_wm_uq_2x2_future_hist_v0",
    )
    return args
