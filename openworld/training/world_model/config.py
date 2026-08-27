"""LIBERO world-model training config.

This is the LIBERO analog of ``Fast-Control-World/config_flow_map.py``.
Defaults mirror the DROID values from that file unless noted in
``docs/LIBERO.md``.

Override fields by writing a new file in ``configs/training/`` that
subclasses :class:`LiberoWMArgs` (or just instantiates it with overrides) and
points the training entrypoint at it via ``--config configs/training/foo.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


@dataclass
class LiberoWMArgs:
    # ---------------- training infra ----------------
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    train_batch_size: int = 1
    shuffle: bool = True
    num_train_epochs: int = 100
    max_train_steps: int = 500_000
    checkpointing_steps: int = 20_000
    validation_steps: int = 2_500
    max_grad_norm: float = 1.0
    video_num: int = 10
    debug: bool = False

    # ---------------- model paths -------------------
    svd_model_path: str = "external/stable-video-diffusion-img2vid"
    clip_model_path: str = "external/clip-vit-base-patch32"
    ckpt_path: str | None = None  # initial WM weights, e.g. checkpoint-10000.pt
    # Optional AdamW optimizer state saved by a previous run (see train_wm.py
    # `optimizer-<step>.pt`). Carries momentum/variance across orchestrated
    # FT cycles so each cycle isn't ~hundreds of steps of warm-up noise.
    optimizer_state_path: str | None = None

    # ---------------- dataset -----------------------
    dataset_root_path: str = "data/libero_processed"
    # Mix all 5 suites by default (equal probability). Override to specialize.
    dataset_names: str = "libero_spatial+libero_object+libero_goal+libero_10+libero_90"
    dataset_meta_info_path: str = "dataset_meta_info"
    # By default the per-suite normalization stat lives at
    # ``dataset_meta_info/<suite>/stat.json``. If a per-suite file is
    # missing, the loader falls back to ``dataset_meta_info/libero/stat.json``.
    dataset_cfgs: str = "libero_spatial+libero_object+libero_goal+libero_10+libero_90"
    prob: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    annotation_name: str = "annotation"
    num_workers: int = 4
    # Preprocessed LIBERO data has T_state == T_latent (latents already at
    # 20 Hz, see scripts/preprocess_libero_for_wm.py), so the natural
    # pairing is down_sample=1: state_id = rgb_id, frame_len = T_state.
    # The historical default of 4 (from a DROID analog "5 Hz WM rate")
    # caused dataset.py to clip rgb_id to T_state//4 -- training only on
    # the first 25% of each trajectory, with state lookup at state[rgb_id*4]
    # which paired a latent at time t/20s with state at time t/5s.
    # Legacy LIBERO checkpoints (libero_0429/checkpoint-36000.pt) were
    # trained with the old default; FT'ing them under the new default will
    # show a transient loss bump as the model adapts to the corrected pairing.
    down_sample: int = 1
    skip_step: int = 1

    # ---------------- logging -----------------------
    tag: str = "libero_flow_matching"
    output_dir: str = field(init=False)
    wandb_project_name: str = "libero_world_model"
    wandb_run_name: str = field(init=False)

    # ---------------- model arch --------------------
    motion_bucket_id: int = 127
    fps: int = 7
    guidance_scale: float = 2.0
    num_inference_steps: int = 50
    test_num_inference_steps: tuple[int, ...] = ()
    decode_chunk_size: int = 7
    width: int = 320
    height: int = 320

    num_frames: int = 5  # future frames per WM rollout
    num_history: int = 6
    action_dim: int = 7  # 6 EEF (xyz + axis-angle) + 1 gripper
    num_cams: int = 2  # agentview + wrist

    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    dtype: torch.dtype = torch.bfloat16

    # ---------------- flow / shortcut ---------------
    SIGMA_MIN: float = 0.02
    SIGMA_MAX: float = 700.0
    flow_map_type: str = "flow_matching"  # one of {flow_matching, shortcut, flow_map}
    distance_conditioning: bool = False
    use_train_set_for_val: bool = False
    use_weights: bool = False
    predict_uncertainty: bool = False
    uncertainty_weight: float = 0.01
    uq_vis_t_targets: tuple[float, ...] = (0.9, 0.5, 0.1)
    # With this probability, shift the training anchor forward by a random
    # delta in [1, num_frames-1]*skip steps so that recent history slots
    # contain GT frames from the prior chunk's prediction window.  This
    # simulates the iterative test-time scenario where rolled2 history
    # includes model-predicted frames.  0.0 = disabled (backward-compatible).
    p_future_in_history: float = 0.0
    # With this probability, replace the full history window with the most-recent
    # history frame repeated for all slots (h6, h6, ..., h6 instead of h1..h6).
    # Simulates scenarios where only the immediately preceding frame is available.
    # 0.0 = disabled (backward-compatible).
    p_single_history: float = 0.0
    # With this probability (drawn once per training step / micro-batch, not
    # per-sample), grow the history window by k in [1, num_frames-1] extra
    # slots holding frame_now+1..frame_now+k -- the SAME future frames that
    # also appear in the noised target block. Teaches a self-refinement
    # objective: reconstruct frames the model is simultaneously shown as
    # (noise-augmented) clean context. Implemented entirely inside
    # CrtlWorld.forward() -- no dataset.py changes, since the future frames
    # needed are already loaded for every sample. Makes the "run twice,
    # splice pass-1 predictions into history, run again" inference-time
    # self-consistency check (uq_epi_mode="future_overlap") in-distribution
    # for the checkpoint. 0.0 = disabled (backward-compatible).
    p_history_future_overlap: float = 0.0
    # Noise-augmentation std-dev cap applied to the overlap history slots
    # specifically (separate from the fixed 0.3 used for true-past history
    # in forward()). Kept small so overlap context stays a strong/near-clean
    # signal, matching the near-clean self-generated frames the checkpoint
    # will be fed at inference-time pass 2.
    history_overlap_noise_scale: float = 0.05
    # Conditional on the history-future-overlap branch firing: with this
    # probability, replace the peeked future frames with a mismatched future
    # drawn from a different episode instead of the true continuation. Breaks
    # the "peek == answer" shortcut that let the logvar head learn
    # content-independent confidence (see droid_flow_matching_uq_future_overlap_v1's
    # known collapse -- near-uniform overconfidence regardless of whether the
    # peek is actually plausible). Implemented in dataset.py (samples a
    # distractor episode's future latents) and spliced in inside
    # CrtlWorld.forward(), which still supervises the diffusion loss against
    # the TRUE target future regardless -- so a false peek naturally produces
    # higher prediction error and, via the existing NLL uq_loss, higher
    # predicted uncertainty, with no new loss term needed.
    # Only meaningful when p_history_future_overlap > 0. 0.0 = disabled.
    p_false_future: float = 0.0
    # If True, zero the action conditioning at the overlap slot (both true-
    # and false-peek cases) instead of leaving the real trajectory's action
    # there. The correct future action is always separately available at the
    # true target position regardless of the overlap splice, so a real
    # action at the overlap slot is redundant and gives the model an
    # action<->frame consistency shortcut that bypasses judging the peeked
    # frame's own plausibility. Default False preserves
    # droid_flow_matching_uq_future_overlap_v1's exact behavior --
    # scripts/replay_libero_wm_traj.py and run_droid_hardware_active_uq.py
    # must be evaluated with a matching --overlap_zero_action flag, since v1
    # was trained with a real action there.
    zero_overlap_action: bool = False

    flow_map_loss_type: str = "lsd"
    psd_sample_mode: str = "uniform"
    bias_prob: float = -1
    one_step_prob: float = 0.0
    one_step_sample: bool = False

    # ---------------- shortcut ----------------------
    bootstrap_bs: int = 1
    DENOISE_TIMESTEPS: int = 128
    single_bs_mode: bool = False

    def __post_init__(self) -> None:
        _n_exclusive = sum(
            p > 0.0
            for p in (self.p_future_in_history, self.p_single_history, self.p_history_future_overlap)
        )
        if _n_exclusive > 1:
            raise ValueError(
                "p_future_in_history, p_single_history, and p_history_future_overlap are "
                "mutually exclusive for now; set at most one to a non-zero value. "
                "(p_history_future_overlap is implemented independently in CrtlWorld.forward() "
                "rather than dataset.py, so relaxing this to allow composition later is a "
                "one-line change -- remove this check -- once each effect is validated alone.)"
            )
        if self.p_false_future > 0.0 and self.p_history_future_overlap <= 0.0:
            raise ValueError(
                "p_false_future is only meaningful when p_history_future_overlap > 0 "
                "(it replaces the overlap-peek content conditionally on that branch firing)."
            )
        self.output_dir = f"checkpoints/wm_libero/{self.tag}"
        self.wandb_run_name = self.tag
        # Per-camera latent shape after SVD VAE 8x downsample.
        self.latent_h_per_cam = self.height // 8
        self.latent_h_total = self.latent_h_per_cam * self.num_cams
        self.latent_w = self.width // 8
