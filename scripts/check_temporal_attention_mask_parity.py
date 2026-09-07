"""Regression parity check for `attention_masks.py`'s class-swap patch.

Confirms `patch_temporal_attention_masking` + `MaskableTemporalBasicTransformerBlock`
introduce ZERO behavior change when no mask is applied (the default state
before any `apply_temporal_attention_mask` call, and the state whenever
`apply_temporal_attention_mask(unet, None)` is used) -- this is the safety
net for Verification item 1 in the false_future attention-masking plan.

Builds a tiny (fast, CPU-friendly) `UNetSpatioTemporalConditionModel` --
channel widths/heads/cross_attention_dim are shrunk purely for speed; none
of that affects whether the patch changes behavior, since the patch only
touches `TemporalBasicTransformerBlock.forward`'s control flow, not any
learned computation.

Usage:
    uv run python scripts/check_temporal_attention_mask_parity.py
"""

import torch

from openworld.world_models.ctrl_world.attention_masks import (
    apply_temporal_attention_mask,
    build_temporal_attention_mask,
    patch_temporal_attention_masking,
)
from openworld.world_models.ctrl_world.flow_map_unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
)


def _build_tiny_unet() -> UNetSpatioTemporalConditionModel:
    return UNetSpatioTemporalConditionModel(
        in_channels=8,
        out_channels=4,
        block_out_channels=(32, 32, 32, 32),  # must each be divisible by GroupNorm's default 32 groups
        layers_per_block=1,
        cross_attention_dim=8,
        num_attention_heads=(1, 1, 2, 2),
        num_frames=11,
    )


def _random_inputs(batch: int, num_frames: int, cross_attention_dim: int, size: int, device: torch.device):
    torch.manual_seed(0)
    sample = torch.randn(batch, num_frames, 8, size, size, device=device)
    timestep = torch.rand(batch, device=device)
    encoder_hidden_states = torch.randn(batch, num_frames, cross_attention_dim, device=device)
    added_time_ids = torch.zeros(batch, 3, device=device)  # (fps, motion_bucket_id, noise_aug_strength)-shaped
    return sample, timestep, encoder_hidden_states, added_time_ids


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, num_frames, size = 2, 11, 16  # size must be a multiple of 2**(len(block_out_channels)-1) = 8

    unet_before = _build_tiny_unet().to(device).eval()
    unet_after = _build_tiny_unet().to(device).eval()
    unet_after.load_state_dict(unet_before.state_dict())  # identical weights, only `unet_after` gets patched

    sample, timestep, encoder_hidden_states, added_time_ids = _random_inputs(
        batch, num_frames, cross_attention_dim=8, size=size, device=device
    )

    with torch.no_grad():
        out_before = unet_before(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids, frame_level_cond=True,
        ).sample

        patch_temporal_attention_masking(unet_after)
        # Default state (no apply_temporal_attention_mask call yet): every
        # block's `_temporal_attention_mask` attribute is simply absent, so
        # `MaskableTemporalBasicTransformerBlock.forward`'s `getattr(...,
        # None)` check should route straight to `super().forward(...)`.
        out_after_unset = unet_after(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids, frame_level_cond=True,
        ).sample

        # Explicitly passing mask=None should be identical too.
        apply_temporal_attention_mask(unet_after, None)
        out_after_none = unet_after(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids, frame_level_cond=True,
        ).sample

        # A genuine mask should change the output (sanity check the mask
        # path is actually wired up and not silently a no-op).
        mask_active = build_temporal_attention_mask(
            batch_size=batch, true_hist_len=4, max_overlap_slots=2,
            overlap_active=True, num_target=num_frames - 6, device=device,
        )
        apply_temporal_attention_mask(unet_after, mask_active)
        out_after_masked = unet_after(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids, frame_level_cond=True,
        ).sample

        # overlap_active=False must NOT produce NaN/Inf anywhere in the
        # output -- this is the degenerate-all-masked-row case (peek rows
        # get only a diagonal self-attention fallback; see
        # attention_masks.py's build_temporal_attention_mask docstring).
        mask_inactive = build_temporal_attention_mask(
            batch_size=batch, true_hist_len=4, max_overlap_slots=2,
            overlap_active=False, num_target=num_frames - 6, device=device,
        )
        apply_temporal_attention_mask(unet_after, mask_inactive)
        out_after_inactive = unet_after(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids, frame_level_cond=True,
        ).sample

    assert torch.equal(out_before, out_after_unset), (
        "Patched unet with no mask ever applied diverged from the unpatched baseline -- "
        "the class-swap is not behavior-preserving by default."
    )
    assert torch.equal(out_before, out_after_none), (
        "apply_temporal_attention_mask(unet, None) diverged from the unpatched baseline."
    )
    max_abs_diff = (out_after_masked - out_before).abs().max().item()
    assert max_abs_diff > 0, (
        "A real mask produced output identical to the unmasked baseline -- "
        "the mask is not actually taking effect."
    )
    assert torch.isfinite(out_after_inactive).all(), (
        "overlap_active=False produced NaN/Inf -- the degenerate all-masked peek-row case "
        "is not actually being caught by the diagonal self-attention fallback."
    )

    print("PASS: patch is a no-op when disabled (mask=None / never applied).")
    print(f"PASS: a real (active) mask changes output as expected (max abs diff = {max_abs_diff:.6f}).")
    print("PASS: an inactive-peek mask produces finite output (no NaN from the degenerate all-masked row).")


if __name__ == "__main__":
    main()
