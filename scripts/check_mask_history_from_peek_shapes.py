"""Verification 2 (plan: fixed-length masked future-overlap peek):
confirm the total frame-sequence length fed to the UNet's temporal
attention -- and therefore `torch.arange(num_frames)`-based position
embedding, see attention_masks.py's module docstring -- is IDENTICAL
whether the history-future-overlap branch fires or not, when
`mask_history_from_peek=True`. This is the concrete, checkable claim the
positional part of the plan rests on.

Runs `CrtlWorld.forward()` twice on the same fake batch, forcing the
overlap-firing coin flip (`torch.rand(1).item() < p_hfo` in
flow_map_ctrl_world.py) to True and False respectively, and asserts the
`sample` tensor's frame dim passed into the UNet is the same in both cases
(captured via a forward pre-hook on `self.unet`, so it directly observes
what the UNet actually receives rather than re-deriving it independently).

Usage:
    uv run python scripts/check_mask_history_from_peek_shapes.py
"""

from unittest.mock import patch

import torch

from openworld.training.world_model.config import LiberoWMArgs
from openworld.world_models.ctrl_world.flow_map_ctrl_world import CrtlWorld


def _real_rand_except_coin_flip(force: bool):
    """Wraps torch.rand: intercepts exactly the coin-flip's `torch.rand(1)`
    call (the only call site reached at shape (1,) for flow_map_type=
    'flow_matching') and forces it to `0.0` (True) or `1.0` (False);
    delegates every other shape to the real `torch.rand` unchanged.
    """
    real_rand = torch.rand

    def _wrapped(*args, **kwargs):
        if len(args) == 1 and args[0] == 1 and not kwargs:
            return torch.tensor([0.0 if force else 1.0])
        return real_rand(*args, **kwargs)

    return _wrapped


def _fake_batch(args: LiberoWMArgs, batch_size: int, device: torch.device) -> dict:
    total_frames = args.num_history + args.num_frames
    latents = torch.randn(batch_size, total_frames, 4, args.latent_h_total, args.latent_w, device=device)
    action = torch.randn(batch_size, total_frames, args.action_dim, device=device)
    false_future_latent = torch.zeros(batch_size, args.num_frames - 1, 4, args.latent_h_total, args.latent_w, device=device)
    use_false_future = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return {
        "latent": latents,
        "action": action,
        "text": ["pick up the cup"] * batch_size,
        "false_future_latent": false_future_latent,
        "use_false_future": use_false_future,
    }


def main() -> None:
    # Shape/position-embedding correctness doesn't depend on model size or
    # resolution -- use a small spatial size (see height/width below) so the
    # full-size (default block_out_channels) UNet CrtlWorld.__init__ always
    # builds still fits comfortably in memory and runs fast.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = LiberoWMArgs(
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        num_cams=3, height=64, width=64, num_frames=5, num_history=6, action_dim=7, down_sample=3,
        flow_map_type="flow_matching",
        predict_uncertainty=True,
        p_history_future_overlap=0.5,
        fixed_overlap_k=True,
        p_false_future=0.25,
        p_shifted_future=0.25,
        zero_overlap_action=True,
        mask_history_from_peek=True,
        tag="check_mask_history_from_peek_shapes",
    )

    model = CrtlWorld(args).to(device)
    model.train()

    observed_num_frames = {}

    def _capture_num_frames(_module, args_):
        sample = args_[0]
        observed_num_frames["last"] = sample.shape[1]

    model.unet.register_forward_pre_hook(_capture_num_frames)

    batch = _fake_batch(args, batch_size=1, device=device)

    with patch("torch.rand", side_effect=_real_rand_except_coin_flip(force=True)) as _:
        model.forward(batch)
        num_frames_active = observed_num_frames["last"]

    with patch("torch.rand", side_effect=_real_rand_except_coin_flip(force=False)) as _:
        model.forward(batch)
        num_frames_inactive = observed_num_frames["last"]

    print(f"overlap_active=True  -> UNet sees num_frames={num_frames_active}")
    print(f"overlap_active=False -> UNet sees num_frames={num_frames_inactive}")

    assert num_frames_active == num_frames_inactive, (
        "Sequence length fed to the UNet differs between overlap-active and overlap-inactive "
        "steps under mask_history_from_peek=True -- the fixed-length invariant this plan "
        "depends on for consistent position embeddings does NOT hold."
    )
    expected = args.num_history + (args.num_frames - 1) + args.num_frames
    assert num_frames_active == expected, (
        f"Expected fixed length {expected} (num_history + max_overlap_slots + num_frames), "
        f"got {num_frames_active}."
    )

    print("PASS: UNet receives an identical, fixed sequence length regardless of overlap_active.")


if __name__ == "__main__":
    main()
