"""DROID world-model training config: UQ head + history-future-overlap augmentation.

Enables the history-future-overlap augmentation (p_history_future_overlap=0.5)
instead of future-in-history (mutually exclusive -- see config.py
__post_init__). With 50% probability, the history window is grown by k in
[1, num_frames-1] extra slots holding frame_now+1..frame_now+k -- the SAME
future frames that also appear in the noised target block -- teaching a
self-refinement objective: reconstruct frames the model is simultaneously
shown as (near-clean) context. This closes the train/test gap for the
"future_overlap" epistemic-UQ mode in scripts/replay_libero_wm_traj.py, where
pass 2 splices pass-1's own predicted future frames into history and
re-predicts the SAME target range.

Deliberately NOT warm-started from any existing DROID checkpoint: the only
one that exists, droid_flow_matching_uq_future_hist_v0, is itself flagged
misaligned (trained with down_sample=1, should be 3 -- see
droid_wm_uq_future_hist.py's docstring), and its down_sample=3 fix (v1) has
never actually been trained (no checkpoint on disk). Rather than wait on that
or inherit the known bug, this config trains fresh from the base SVD
checkpoint (ckpt_path=None) with the corrected down_sample=3 from the start,
so it doesn't depend on droid_wm_uq_future_hist_v1 (or any other DROID
checkpoint) ever being trained.

v1: fixes history_overlap_noise_scale (0.05 -> 0.3). v0 was trained to
checkpoint-190000.pt and its replay (uq_epi_mode=future_overlap) showed the
logvar head collapsing to near-uniform overconfidence on overlap slots
(mean_pdf_diff ~2.68, stdev 0.13 across 739 chunks -- a near-constant global
shift, not content-driven epistemic signal) -- the near-clean (0.05-noise)
overlap context let the model shortcut to "trivially certain" instead of
learning graded confidence. Restarting fresh (not warm-started from v0) since
v0's weights may have baked in the shortcut. Tag bumped v0 -> v1 so this
run's checkpoints land in a fresh directory instead of overwriting v0's
checkpoint-{90000,110000,...,190000}.pt at the same step numbers (train_wm.py
never auto-resumes from output_dir; it only loads a checkpoint when
ckpt_path is set, so a same-tag restart would silently clobber v0's files).
"""

import os

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/data"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path=None,  # train from base SVD -- see module docstring for why

        # ----- Dataset: reuse vidwm's existing droid_ctrl_world data -----
        dataset_root_path=data_root,
        dataset_meta_info_path=os.path.join(data_root, "dataset_meta_info"),
        dataset_names="droid_ctrl_world",
        dataset_cfgs="dataset_meta_info/droid_ctrl_world",
        prob=(1.0,),
        annotation_name="annotation",

        # ----- Compute -----
        train_batch_size=3,
        gradient_accumulation_steps=2,
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
        down_sample=3,  # corrected value -- build against the fix, don't propagate v0's bug

        # ----- Loss / sampling defaults -----
        flow_map_type="flow_matching",
        distance_conditioning=False,

        # ----- UQ head -----
        predict_uncertainty=True,
        uncertainty_weight=0.01,

        # ----- History-future-overlap augmentation -----
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        # Bumped 0.05 -> 0.3 (matches the noise cap already used for true-past
        # history) after checkpoint-190000 showed the logvar head collapsing to
        # near-uniform overconfidence on overlap slots (mean_pdf_diff ~2.68,
        # stdev 0.13 across 739 chunks -- a near-constant global shift, not
        # content-driven epistemic signal; see replay_epi_future_overlap).
        # Near-clean overlap context was letting the model shortcut to "trivially
        # certain" instead of learning graded confidence.
        history_overlap_noise_scale=0.3,

        tag="droid_flow_matching_uq_future_overlap_v1",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
