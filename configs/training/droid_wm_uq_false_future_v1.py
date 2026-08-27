"""DROID world-model training config: UQ head + history-future-overlap
augmentation + false-future injection.

Builds on droid_flow_matching_uq_future_overlap_v1
(configs/training/droid_wm_uq_future_overlap.py), which enables
p_history_future_overlap=0.5: with 50% probability, the history window is
grown by k in [1, num_frames-1] extra slots holding frame_now+1..frame_now+k
-- the SAME future frames that also appear in the noised target block --
teaching a self-refinement objective (reconstruct frames the model is
simultaneously shown as near-clean context).

v1's replay eval (uq_epi_mode=future_overlap, see
checkpoints/wm_droid/droid_flow_matching_uq_future_overlap_v1/replay_epi_future_overlap/)
showed the logvar head still collapsing toward near-uniform overconfidence on
overlap slots, not clearly content-driven: because training *always* shows the
literal correct future when the overlap augmentation fires, the model can
learn "a peek matching the target => be confident" as a content-independent
shortcut, rather than actually checking whether the peeked future is
*plausible given the real history/action*. At inference the peek (pass-1's
own prediction) is not guaranteed correct, so the model stays overconfident
regardless of whether pass-1 was actually right.

This config adds two changes on top of v1, both implemented in
CrtlWorld.forward() (flow_map_ctrl_world.py) and openworld/training/world_model/dataset.py:

1. p_false_future=0.5: conditional on the overlap branch firing, with 50%
   probability the peeked frames are replaced with a MISMATCHED future
   sampled from a different episode (dataset.py picks the distractor; the
   splice happens in forward()). The diffusion loss still supervises against
   the TRUE target future regardless, so a false peek naturally produces
   higher prediction error and, via the existing NLL uq_loss, higher
   predicted uncertainty -- no new loss term needed. This breaks the
   "peek == answer" tautology.

2. zero_overlap_action=True: the overlap slot's action conditioning is
   zeroed (both true- and false-peek cases), not left as the real
   trajectory's action. The correct future action is already present,
   unmodified, at the true target position regardless of the overlap splice,
   so a real action at the overlap slot is redundant and gives the model an
   action<->frame consistency shortcut that bypasses judging the peeked
   frame's own plausibility -- we want confidence in the peek to come purely
   from visual/temporal content.

IMPORTANT -- eval/inference compatibility: because (2) changes what the
overlap slot's action conditioning looks like relative to v1, any evaluation
of THIS checkpoint via scripts/replay_libero_wm_traj.py or
scripts/run_droid_hardware_active_uq.py MUST set --overlap_zero_action (or
the YAML `active_uq.overlap_zero_action: true`) to match. Evaluating v1 (or
any earlier checkpoint) must NOT set that flag -- those were trained with a
real action at the overlap slot. See config.py's zero_overlap_action
docstring.

Trains from scratch (ckpt_path=None), matching the v0->v1 convention already
used for augmentation changes in this project (droid_wm_uq_future_overlap.py's
own docstring): a fresh run avoids any chance that v1's weights have already
baked in the "peek == answer" shortcut in a way that resists correction.

Tag bumped to droid_flow_matching_uq_false_future_v1 so this run's
checkpoints land in a fresh directory instead of touching v1's.
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
        validation_steps=10_000,
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

        # ----- History-future-overlap augmentation (unchanged from v1) -----
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        history_overlap_noise_scale=0.3,

        # ----- False-future injection (new -- see module docstring) -----
        p_false_future=0.5,
        zero_overlap_action=True,

        tag="droid_flow_matching_uq_false_future_v1",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
