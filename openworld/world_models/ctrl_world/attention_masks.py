"""Frame-level temporal attention masking for CrtlWorld's SVD-based UNet.

Motivation (see the "false_future"/"future_overlap" UQ line, e.g.
``configs/training/droid_wm_uq_false_future_v2.py``): ``uq_epi_mode=
future_overlap`` estimates epistemic uncertainty by comparing two forward
passes over the same target frames -- pass-1 conditions on true history
only, pass-2 additionally splices a "peeked" future (true, temporally
shifted, or from a different episode) into extra history slots. Two
confounds ride along with that comparison purely because the two passes use
different total frame-sequence lengths:

1. Attention renormalization: today's temporal self-attention
   (``TemporalBasicTransformerBlock``, diffusers ``attention.py``) is fully
   bidirectional and unmasked, so true-history's own token representations
   get recomputed differently in pass-2 (more keys/values in the softmax)
   than in pass-1, even when the peek carries no new information.
2. Positional-embedding shift: ``TransformerSpatioTemporalModel.forward``
   (diffusers ``transformer_temporal.py``) builds a per-frame position
   embedding from ``torch.arange(num_frames)`` where ``num_frames`` is the
   *current* total sequence length -- so the same conceptual target frame
   gets a different absolute position embedding depending on whether the
   peek is present.

The fix used throughout this repo's history-future-overlap augmentation
(``LiberoWMArgs.mask_history_from_peek``) is to always reserve a fixed
number of peek slots in the frame sequence, whether or not a real peek is
available this step, and mask temporal attention so that:

  - true-history frames can never attend to the peek segment (real or
    zero-padded placeholder) -- this keeps true-history's own contribution
    identical between pass-1 and pass-2 for the same reason it keeps it
    identical between the "overlap fired" and "overlap didn't fire" training
    branches.
  - symmetrically, the peek segment (when active/real) can never attend to
    true-history either -- it's sealed to itself, just like true-history is.
    Peek doesn't need history to serve its role (it's presented to the model
    as a fixed candidate future, not something that needs to reconcile
    itself against the past via attention); only target needs to compare the
    two, and target is the one segment that attends to everything.
  - when the peek segment is inactive (zero-padded placeholder, i.e. no real
    peek this step), it is fully masked from *every* other frame, including
    the target -- functionally equivalent to "the peek slots don't exist,"
    since a fully-masked key contributes zero softmax weight to every other
    query.
  - when the peek segment is active, the target may attend to it (that's the
    whole point -- self-consistency conditioning); the peek segment attends
    only to itself, exactly like true-history does.

Because total sequence length is now fixed regardless of peek-active state,
every frame's position embedding is identical across all four cases
(training overlap-active / overlap-inactive, inference pass-1 / pass-2).

Implementation note: the underlying ``Attention``/``AttnProcessor2_0``
machinery (diffusers ``attention_processor.py``) already fully supports an
``attention_mask`` argument all the way down to
``torch.nn.functional.scaled_dot_product_attention`` -- the only gap is that
nothing between ``UNetSpatioTemporalConditionModel.forward`` and
``TemporalBasicTransformerBlock.forward`` threads a mask parameter through
(and the intermediate blocks live in vendored site-packages diffusers, not
this repo). Rather than vendor and thread a new kwarg through ~5 block
classes, we patch the one class that actually does frame-axis self-attention
via a class-swap (``module.__class__ = MaskableTemporalBasicTransformerBlock``)
and communicate the mask via a plain attribute set immediately before each
``unet(...)`` call. Pinned against diffusers==0.34.0's
``TemporalBasicTransformerBlock.forward`` -- re-diff against that method if
upgrading diffusers.

Caveats (both confirmed non-issues for the ``flow_matching``-type training
this targets -- re-verify if reused elsewhere):
  - Gradient checkpointing is only enabled in ``CrtlWorld.__init__`` for
    ``flow_map_type == 'flow_map'`` (not used by the false_future line,
    which uses ``flow_map_type == 'flow_matching'``).
  - The ``torch.func.jvp`` path in ``flow_map_ctrl_world.py`` (used by
    ``_model_partial_t``/``_flow_map``) is likewise only exercised by
    ``flow_map_type == 'flow_map'``.
"""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.models.attention import TemporalBasicTransformerBlock, _chunked_feed_forward


class MaskableTemporalBasicTransformerBlock(TemporalBasicTransformerBlock):
    """Drop-in replacement for diffusers' ``TemporalBasicTransformerBlock``
    that additionally applies an externally-stashed frame-level mask
    (``self._temporal_attention_mask``) to the frame-axis self-attention
    (``attn1``). Installed on existing instances via
    ``patch_temporal_attention_masking`` (a ``__class__`` swap -- no weights
    are touched), so this only changes ``forward()``.

    With ``self._temporal_attention_mask`` absent or ``None`` (the default,
    and the only state before ``patch_temporal_attention_masking`` sets
    anything), ``forward()`` is byte-for-byte identical to the base class --
    verified by ``scripts/check_temporal_attention_mask_parity.py``.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = getattr(self, "_temporal_attention_mask", None)
        if mask is None:
            return super().forward(hidden_states, num_frames, encoder_hidden_states=encoder_hidden_states)

        # ---- identical to TemporalBasicTransformerBlock.forward (diffusers
        # 0.34.0, attention.py:713-768) up to and including the attn1 call,
        # which is the one line that differs (attention_mask=... added). ----
        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)

        # `mask` is (batch_size, num_frames, num_frames) -- the same mask
        # applies at every spatial position, so expand it across the
        # spatially-folded `seq_length` dim (height*width at this UNet
        # resolution level, which varies per block instance) to match the
        # (batch_size * seq_length, num_frames, ...) shape attn1 sees here.
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"temporal_attention_mask batch dim {mask.shape[0]} != hidden_states batch dim {batch_size}"
            )
        expanded_mask = (
            mask.unsqueeze(1)
            .expand(batch_size, seq_length, num_frames, num_frames)
            .reshape(batch_size * seq_length, num_frames, num_frames)
        )
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None, attention_mask=expanded_mask)
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states


def patch_temporal_attention_masking(unet: torch.nn.Module) -> None:
    """Swap every ``TemporalBasicTransformerBlock`` inside ``unet`` to
    ``MaskableTemporalBasicTransformerBlock`` in place (no weights touched),
    and cache the resulting list on ``unet._temporal_attention_blocks`` so
    ``apply_temporal_attention_mask`` can reach them without walking
    ``unet.modules()`` again on every call. Idempotent -- safe to call more
    than once (e.g. if both ``CrtlWorld.__init__`` and an inference-side
    loader independently hold/patch the same unet instance).

    Both ``CrtlWorld`` (training) and ``CtrlWorldDiffusionPipeline``
    (inference) read off the *same* ``unet`` object, so patching once at
    construction time is enough for both code paths to use
    ``apply_temporal_attention_mask`` later.
    """
    if getattr(unet, "_temporal_attention_blocks", None) is not None:
        return  # already patched
    blocks = []
    for module in unet.modules():
        if isinstance(module, TemporalBasicTransformerBlock):
            module.__class__ = MaskableTemporalBasicTransformerBlock
            blocks.append(module)
    unet._temporal_attention_blocks = blocks


def apply_temporal_attention_mask(unet: torch.nn.Module, mask: Optional[torch.Tensor]) -> None:
    """Stash ``mask`` (or ``None`` to disable) on every patched temporal
    attention block in ``unet``, to take effect on the next ``unet(...)``
    call. Call this immediately before every such call in this code path
    (training and inference alike) -- always pass an explicit value
    (including ``None``) rather than relying on a previous call's leftover
    state.

    Raises if ``unet`` was never patched via ``patch_temporal_attention_masking``
    -- that's a wiring bug at the call site, not something to silently no-op.
    """
    blocks = getattr(unet, "_temporal_attention_blocks", None)
    if blocks is None:
        raise RuntimeError(
            "apply_temporal_attention_mask called on a unet that was never patched -- "
            "call patch_temporal_attention_masking(unet) once at construction time first."
        )
    for block in blocks:
        block._temporal_attention_mask = mask


def build_temporal_attention_mask(
    batch_size: int,
    true_hist_len: int,
    max_overlap_slots: int,
    overlap_active: bool,
    num_target: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the boolean frame-level attention mask (True = may attend) for
    the fixed-length sequence ``[true_history | peek | target]`` of total
    length ``true_hist_len + max_overlap_slots + num_target``, shape
    ``(batch_size, F, F)`` where ``F`` is that total length (broadcast
    identically across the batch -- every example in a batch shares the same
    segment boundaries and the same ``overlap_active`` state, matching the
    existing per-batch, not per-sample, coin flip in ``forward()``'s
    history-future-overlap augmentation).

    Both conditioning segments (true-history and peek) are "sealed" -- each
    only ever attends within its own block, refining its representation
    solely from its own raw content (via the OTHER attention/conv layers --
    spatial self+cross-attention, action cross-attention -- untouched by this
    mask) and never from each other. Target is the only segment that reads
    across blocks; it's where the model actually compares "does the peeked
    future look consistent with real history" -- neither conditioning
    segment needs to do that reasoning itself, so neither needs the other's
    content flowing into its own representation.

    - ``overlap_active=False`` (peek segment is a zero-padded placeholder,
      i.e. no real peek this step / this is inference pass-1): peek columns
      are masked from every row, including target's and its own; true
      history and target otherwise attend as they would with no peek segment
      at all (true history to itself, target to true-history + itself).
    - ``overlap_active=True`` (peek segment holds real content -- true,
      shifted, or false-future -- i.e. this is inference pass-2 or a
      training step where the overlap branch fired): true-history rows and
      peek rows are each sealed to their own block (symmetric with each
      other -- neither can see the other); target rows attend to everything
      (unchanged from the unmasked baseline for this segment).
    """
    total = true_hist_len + max_overlap_slots + num_target
    hist_end = true_hist_len
    peek_end = true_hist_len + max_overlap_slots

    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    # true-history rows: may attend to true-history columns only, ever.
    mask[:hist_end, :hist_end] = True
    # target rows: may always attend to true-history + itself.
    mask[peek_end:, :hist_end] = True
    mask[peek_end:, peek_end:] = True

    if overlap_active:
        # peek rows: sealed to the peek block only -- symmetric with
        # true-history's own sealing above, NOT true-history + peek (peek
        # doesn't need to reason about history; only target does).
        mask[hist_end:peek_end, hist_end:peek_end] = True
        # target rows: may additionally attend to the (now-real) peek.
        mask[peek_end:, hist_end:peek_end] = True
    else:
        # Placeholder case: peek columns stay all-False for every OTHER
        # row (true-history, target), fully excluding the placeholder from
        # everyone else's attention. But peek's own rows must still get
        # *some* True entry -- an all-False row makes softmax divide by
        # zero (every score is -inf, so every weight is 0/0 = NaN). That
        # NaN never reaches the loss (only `target`'s output slice is used
        # downstream, and self-attention doesn't mix across frame/query
        # positions -- each query's output depends only on the keys *it*
        # attends to), but relying on NaN staying perfectly confined across
        # every downstream layer is fragile. Give each peek row a trivial
        # self-attention fallback instead: diagonal-only, so the row is
        # well-defined (attends only to itself) without granting it, or
        # granting anyone else, any real connectivity.
        diag = torch.eye(max_overlap_slots, dtype=torch.bool, device=device)
        mask[hist_end:peek_end, hist_end:peek_end] = diag

    return mask.unsqueeze(0).expand(batch_size, total, total)
