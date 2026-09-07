"""DROID world-model training config: v3 of the false-future/UQ line --
fixed-length, masked temporal context for the history-future-overlap peek.

Builds on droid_flow_matching_uq_false_future_v2
(configs/training/droid_wm_uq_false_future_v2.py). v2's own comparison
against pass-1 (no peek) is confounded by two things that have nothing to do
with the peek's content, both stemming from the peek-present and peek-absent
cases using different total frame-sequence lengths in the UNet:

1. Attention renormalization: temporal self-attention is fully bidirectional
   and unmasked, so true-history's own token representations get recomputed
   differently depending on how many extra keys/values exist in the
   sequence, independent of what's actually in the peek.
2. Positional-embedding shift: TransformerSpatioTemporalModel's per-frame
   position embedding is built from torch.arange(num_frames) where
   num_frames is the *current* total sequence length -- so the same
   conceptual target frame gets a different absolute position embedding
   depending on whether the peek is present.

One change on top of v2, implemented generically in config.py /
flow_map_ctrl_world.py / the new openworld/world_models/ctrl_world/attention_masks.py
(no v3-specific code paths):

mask_history_from_peek=True: always reserve num_frames-1 peek slots in the
frame sequence -- whether or not the overlap branch fires this step -- so
total sequence length (and therefore every frame's position embedding) is
identical regardless. When the branch doesn't fire, those slots are
zero-filled and fully masked from every other frame (functionally
equivalent to "the slots don't exist"). When it does fire, temporal
attention is masked so true-history and peek are each sealed to their own
block -- neither can attend to the other -- while the target attends to
everything, same as an unmasked model would for that segment. Neither
conditioning segment needs the other's content (peek doesn't need to
reconcile itself against history via attention; it's presented as a fixed
candidate future); only target needs to compare them. See
attention_masks.py's module docstring for the full design and
config.py's mask_history_from_peek docstring for the exact masking rules.

Trains from scratch (ckpt_path=None), matching the v0->v1 convention
(droid_wm_uq_future_overlap.py's and droid_wm_uq_false_future_v1.py's own
docstrings) rather than v2's warm-start from the released Ctrl-World
checkpoint: masking is itself a distribution shift for temporal-attention
weights that have never seen a masked, fixed-length input, and this
project's history already shows a warm-started checkpoint (v0) baking in a
shortcut in a way that resisted correction (motivating v0->v1's fresh-start
choice for that case) -- a fresh run avoids the analogous risk here rather
than needing to first establish whether v2's (not yet checkpointed as of
this writing) or v1's weights would adapt cleanly to the new masked shape.
Also isolates masking as a standalone architectural change, evaluable on its
own footing rather than compounded with either warm-start's own effects.

IMPORTANT -- eval/inference compatibility: evaluate THIS checkpoint with
--mask_history_from_peek (or the YAML `active_uq.mask_history_from_peek:
true`) to match. v1/v2 (and any earlier checkpoint) must NOT set that flag --
they were trained with a variable-length, unmasked sequence.

Tag bumped to droid_flow_matching_uq_false_future_v3 so this run's
checkpoints land in a fresh directory instead of touching v1's/v2's.
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
        down_sample=3,

        # ----- Loss / sampling defaults -----
        flow_map_type="flow_matching",
        distance_conditioning=False,

        # ----- UQ head -----
        predict_uncertainty=True,
        uncertainty_weight=0.01,

        # ----- History-future-overlap augmentation (unchanged from v2) -----
        p_future_in_history=0.0,
        p_history_future_overlap=0.5,
        history_overlap_noise_scale=0.3,
        fixed_overlap_k=True,

        # ----- Three-way future mix (unchanged from v2) -----
        p_false_future=0.25,     # other-episode distractor
        p_shifted_future=0.25,   # same-episode, temporally-shifted distractor
        zero_overlap_action=True,

        # ----- Fixed-length masked peek (new -- see module docstring) -----
        mask_history_from_peek=True,

        tag="droid_flow_matching_uq_false_future_v3",
        wandb_project_name="droid_world_model",
    )
    # Override the config.py default of checkpoints/wm_libero/<tag> so DROID
    # runs don't land under a misleadingly-named "wm_libero" directory.
    args.output_dir = f"checkpoints/wm_droid/{args.tag}"
    return args
