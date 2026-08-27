"""LIBERO world-model training config: UQ head + history-future-overlap augmentation.

Warm-starts from libero_wm_uq_future_hist.py's checkpoint and enables the
history-future-overlap augmentation (p_history_future_overlap=0.5) instead of
future-in-history (mutually exclusive -- see config.py __post_init__). With
50% probability, the history window is grown by k in [1, num_frames-1] extra
slots holding frame_now+1..frame_now+k -- the SAME future frames that also
appear in the noised target block -- teaching a self-refinement objective:
reconstruct frames the model is simultaneously shown as (near-clean) context.

This closes the train/test gap for the "future_overlap" epistemic-UQ mode in
scripts/replay_libero_wm_traj.py, where pass 2 splices pass-1's own predicted
future frames into history and re-predicts the SAME target range (rather than
chaining forward to the next chunk, which is what "iterative" already covers).
"""

import os

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/open-world/data/"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        # Warm-start from the existing, valid future_hist checkpoint -- the
        # augmentation is additive to what it already learned, not a from-scratch run.
        # ckpt_path="checkpoints/wm_libero/libero_flow_matching_uq_future_hist_v0/checkpoint-75000.pt",
        ckpt_path=None,  # train from base SVD -- see module docstring for why
        
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

        # ----- History-future-overlap augmentation -----
        # Mutually exclusive with p_future_in_history/p_single_history (config.py
        # __post_init__) -- explicitly disable the former even though its class
        # default is already 0.0, to make the swap unambiguous at a glance.
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        # Bumped 0.05 -> 0.3 (matches the noise cap already used for true-past
        # history) before this ever started training, based on the DROID run
        # showing the logvar head collapsing to near-uniform overconfidence on
        # near-clean overlap slots (mean_pdf_diff ~2.68, stdev 0.13 across 739
        # chunks -- a near-constant global shift, not content-driven epistemic
        # signal; see droid checkpoint-190000's replay_epi_future_overlap).
        history_overlap_noise_scale=0.3,

        tag="libero_flow_matching_uq_future_overlap_v0",
    )
    return args
