"""LIBERO world-model training config: UQ head + single-history augmentation, on policy-probe 2x2 data.

Same architecture as libero_wm_uq_2x2_single_hist.py but the dataset points to
data/libero_uq_2x2_policy_probe — episodes collected with the pi0.5 policy under
friction variation, where sigma_cm is measured via policy rollouts rather than
kinematic demo replay.
"""

from openworld.training.world_model.config import LiberoWMArgs

_UQ_ROOT = "/scratch/gpfs/AM43/yx2653/projects/UQ_Data_Collection/open-world/data/libero_uq_2x2_policy_probe"


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
        p_single_history=0.5,

        tag="libero_wm_uq_2x2_policy_probe_single_hist_v0",
    )
    return args
