"""DROID world-model training config: v2 of the false-future/UQ line.

Builds on droid_flow_matching_uq_false_future_v1
(configs/training/droid_wm_uq_false_future_v1.py). v1 already fixed the
"peek == answer" content-independent-confidence shortcut by injecting
mismatched-episode futures 50% of the time the history-future-overlap branch
fires (p_false_future=0.5) -- it cheats noticeably less than
droid_flow_matching_uq_future_overlap_v1, but two things observed on v1's
replay evals motivate this run:

1. Predictions conditioned on a *predicted* (pass-1) future are sometimes
   worse than the first-pass prediction itself, suggesting the overlap/UQ
   objective needs more training signal per step, not just a different
   augmentation mix.
2. v1's overlap_k ~ randint(1, num_frames-1) means training only sometimes
   peeks the full future window, while eval (uq_epi_mode=future_overlap)
   always uses the max (epi_overlap_k=0 -> num_frames-1) -- a train/eval
   mismatch that dilutes the self-refinement signal from (1).

Three changes on top of v1, all implemented generically in config.py /
dataset.py / flow_map_ctrl_world.py (no v2-specific code paths):

1. ckpt_path = the officially released pretrained Ctrl-World checkpoint
   (yjguo/Ctrl-World checkpoint-10000.pt, SVD-based, action-conditioned,
   trained on public DROID). Loaded via train_wm.py's existing ckpt_path
   overlay (state_dict, strict=False) on top of the SVD-scaffolded model --
   svd_model_path is unchanged and still used to build the pipeline
   skeleton (VAE/image_encoder/scheduler) before the overlay. The released
   checkpoint has no log_var_net/logvar-head keys (predict_uncertainty is a
   repo-specific addition), so the UQ head starts randomly initialized as
   usual; everything else (unet, action_encoder, vae, image_encoder,
   text_encoder) starts from Ctrl-World's DROID-pretrained weights instead
   of raw SVD, giving a much better starting point for action-conditioned
   video prediction before layering the UQ/false-future objective on top.
   Unlike v0->v1 (which deliberately trained from scratch to avoid
   inheriting a shortcut baked into a *fine-tuned-in-this-repo* checkpoint),
   this is a separately-trained upstream release with no false-future/UQ
   training at all, so there's no shortcut to inherit.

2. fixed_overlap_k=True: the history-future-overlap branch always peeks the
   full num_frames-1 future window instead of a random prefix, matching the
   eval-time default and removing the train/eval mismatch in (2) above.

3. Three-way future mix instead of v1's binary real/other-episode split:
   - p_history_future_overlap=0.5 (unchanged): overlap branch fires 50% of
     steps.
   - Conditional on that: p_shifted_future=0.25 and p_false_future=0.25 (so
     50% of overlap-firing steps keep the true peek, 25% get a
     temporally-shifted window from the SAME episode, 25% get a mismatched
     window from a DIFFERENT episode) -- i.e. 50% real / 25% same-episode
     shift / 25% other-episode overall, as requested. The same-episode
     shift is a harder distractor (same scene/robot/objects, wrong point in
     time) that should push the UQ head to key off actual plausibility
     rather than a same-episode-vs-not shortcut. shifted_future_min_gap/
     max_gap use config.py's defaults (5/30 frames beyond the true peeked
     window); revisit if replay evals show the shift is too easy/hard to
     distinguish from the true continuation.

zero_overlap_action=True is kept from v1 (still required so the model can't
use action<->frame consistency as a plausibility shortcut).

IMPORTANT -- eval/inference compatibility: same as v1, evaluate THIS
checkpoint with --overlap_zero_action / active_uq.overlap_zero_action: true
to match zero_overlap_action=True.

Tag bumped to droid_flow_matching_uq_false_future_v2 so this run's
checkpoints land in a fresh directory instead of touching v1's.
"""

import os

from openworld.training.world_model.config import LiberoWMArgs


CTRL_WORLD_CKPT = (
    "/scratch/gpfs/AM43/yx2653/projects/models/models--yjguo--Ctrl-World/"
    "snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt"
)


def get_args() -> LiberoWMArgs:
    data_root = "/scratch/gpfs/AM43/yy4041/data"
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        # Warm-start from the pretrained Ctrl-World release instead of
        # training from scratch on top of raw SVD -- see module docstring.
        ckpt_path=CTRL_WORLD_CKPT,

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
        validation_steps=10_000,
        max_grad_norm=1.0,

        # ----- Architecture (DROID-specific: 3 cams, 192x320) -----
        num_cams=3,
        height=192,
        width=320,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=3,

        # ----- Loss / sampling defaults -----
        flow_map_type="flow_matching",
        distance_conditioning=False,

        # ----- UQ head -----
        predict_uncertainty=True,
        uncertainty_weight=0.01,

        # ----- History-future-overlap augmentation -----
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        history_overlap_noise_scale=0.3,
        # Always peek the full future window (see module docstring point 2).
        fixed_overlap_k=True,

        # ----- Three-way future mix (see module docstring point 3) -----
        p_false_future=0.25,     # other-episode distractor
        p_shifted_future=0.25,   # same-episode, temporally-shifted distractor
        zero_overlap_action=True,

        tag="droid_flow_matching_uq_false_future_v2",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
