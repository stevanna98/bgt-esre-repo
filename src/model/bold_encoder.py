"""
Parallel Region 1D-CNN encoder for fMRI BOLD signals.

Each brain region's time series is processed independently through a stack of
grouped 1D convolutional layers.  Independence is enforced via PyTorch's
``groups`` parameter: setting ``groups=num_regions`` with a contiguous
channel layout guarantees that no cross-region information is mixed at any
convolutional stage — without explicit loops or per-region module lists.

Memory optimisation
-------------------
fMRI BOLD signals are temporally oversampled relative to the haemodynamic
response function (HRF), which peaks at ~6 seconds.  With typical TRs of
0.7–2 s this yields 1000+ timepoints per scan, leading to large intermediate
tensors (B × R·d × T) that dominate GPU memory during training.

This module applies **early temporal downsampling** before the expensive
grouped convolutions, reducing T by a configurable factor (default 4×).
For a batch of 8 subjects × 360 regions × 64 features × 1200 timepoints,
this cuts peak activation memory from ~3.4 GB to ~850 MB with negligible
information loss given the slow dynamics of the HRF.

Tensor shape conventions used throughout
-----------------------------------------
B  — batch size
R  — number of brain regions  (``num_regions``)
T  — number of BOLD time steps (variable)
d  — feature channels extracted per region (``d``)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Internal building blocks ───────────────────────────────────────────────────

class _GroupedConvBlock(nn.Module):
    """Single grouped-convolution stage.

    Applies one grouped ``Conv1d`` layer followed by batch normalisation,
    a GELU activation, and channel dropout.  All operations respect the
    per-region grouping: no weights are shared across regions and no
    activations cross region boundaries.

    Args:
        in_channels:  Total input channels  (R × in_features_per_region).
        out_channels: Total output channels (R × out_features_per_region).
        kernel_size:  Temporal kernel width.  Odd values are recommended so
                      that ``padding='same'`` is symmetric.
        groups:       Number of independent groups — must equal ``num_regions``
                      and evenly divide both ``in_channels`` and
                      ``out_channels``.
        dropout:      Dropout probability applied to channels after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # Grouped convolution: each region's channels are processed by its
            # own independent filter bank — equivalent to R separate Conv1d
            # layers but fully vectorised on the GPU.
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
                padding="same",   # preserves temporal length (requires PyTorch ≥ 1.9)
                bias=False,       # BN absorbs the bias
            ),
            # Normalise across the full channel dimension (all R × features).
            nn.BatchNorm1d(out_channels),
            # GELU performs well on smooth continuous signals such as BOLD.
            nn.GELU(),
            # Channel dropout regularises inter-feature correlations.
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, in_channels, T)``

        Returns:
            ``(B, out_channels, T)``  — temporal length preserved by
            ``padding='same'``.
        """
        return self.block(x)


class _ResidualGroupedBlock(nn.Module):
    """Grouped conv block with an additive skip connection.

    The skip allows gradients to bypass this block entirely, stabilising
    training when ``kernel_sizes`` has many entries.  Both the main path and
    the skip operate within the same grouped-channel layout so no cross-region
    mixing occurs.

    Args:
        channels:    Total channels  (R × d) — same on input and output.
        kernel_size: Temporal kernel width.
        groups:      Number of independent groups (``num_regions``).
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                groups=groups,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        # Post-skip activation keeps non-linearity after the addition.
        self.post_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, R·d, T)``

        Returns:
            ``(B, R·d, T)``
        """
        return self.post_act(x + self.conv(x))


# ── Public encoder ─────────────────────────────────────────────────────────────

class ParallelRegionEncoder(nn.Module):
    """Region-level feature encoder for fMRI BOLD signals.

    Processes each brain region's BOLD time series in parallel using grouped
    1D convolutions, producing a fixed-length ``d``-dimensional embedding per
    region regardless of the original scan length.

    Architecture
    ------------
    ::

        (B, R, T)                                      ← input
          │
          ├─ Temporal downsampling ─────────────────── AvgPool1d(factor)
          │    (B, R, T // factor)                       (optional, default 4×)
          │
          ├─ Projection block  ─────────────────────── Conv1d(R → R·d, groups=R)
          │    (B, R·d, T')                              + BN + GELU + Dropout
          │
          ├─ Residual block × (len(kernel_sizes) - 1)  Conv1d(R·d → R·d, groups=R)
          │    (B, R·d, T')                              + BN + GELU + Dropout
          │                                              + additive skip
          │
          ├─ AdaptiveAvgPool1d(1)
          │    (B, R·d, 1)
          │
          ├─ squeeze(-1)
          │    (B, R·d)
          │
          └─ view(B, R, d)                             ← output
               (B, R, d)

    Grouped Conv1d channel ordering
    --------------------------------
    PyTorch lays out grouped-convolution output channels contiguously per
    group::

        [r₀·f₀, r₀·f₁, …, r₀·f_{d-1},
         r₁·f₀, …, r₁·f_{d-1},
         …,
         r_{R-1}·f₀, …, r_{R-1}·f_{d-1}]

    This means a plain ``.view(B, R, d)`` after the pool correctly maps the
    ``d`` contiguous channels back to their originating region — no permutation
    or gather is required.

    Args:
        num_regions:  Number of brain parcels / ROIs  (R).  Must match the
                      second dimension of the input tensor.
        d:            Feature channels to extract per region.  Default: ``64``.
        kernel_sizes: Temporal kernel widths for each convolutional layer.
                      The first entry is used for the projection layer (which
                      expands ``1 → d`` features per region); every subsequent
                      entry adds a residual block that preserves ``d`` features
                      per region.  At least one value is required.
                      Default: ``(7, 5, 3)``.
        dropout:      Dropout probability applied after each activation.
                      Default: ``0.1``.
        temporal_downsample_factor:
                      Factor by which to reduce the temporal dimension before
                      the projection layer.  Set to ``1`` to disable.
                      Default: ``4`` (e.g. 1200 → 300 timepoints).

                      **Rationale:** The haemodynamic response function (HRF)
                      peaks at ~6 s, so BOLD signals sampled at TR ≤ 1 s are
                      temporally redundant.  Downsampling 4–8× before the
                      expensive grouped convolutions cuts memory by the same
                      factor with negligible information loss.

    Raises:
        ValueError: If ``kernel_sizes`` is empty, ``d < 1``, ``num_regions < 1``,
                    ``dropout`` is outside ``[0, 1)``, or
                    ``temporal_downsample_factor < 1``.

    Example::

        encoder = ParallelRegionEncoder(num_regions=360, d=64)
        bold    = torch.randn(8, 360, 1200)   # (B, R, T)
        feats   = encoder(bold)               # → (8, 360, 64)

        # Memory comparison (approx. peak activation for batch=8, R=360, d=64):
        #   temporal_downsample_factor=1  → ~3.4 GB  (T=1200)
        #   temporal_downsample_factor=4  → ~850 MB  (T=300)
        #   temporal_downsample_factor=8  → ~425 MB  (T=150)
    """

    def __init__(
        self,
        num_regions: int,
        d: int = 64,
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        dropout: float = 0.1,
        temporal_downsample_factor: int = 8,
    ) -> None:
        super().__init__()

        # ── Validation ────────────────────────────────────────────────────
        if num_regions < 1:
            raise ValueError(f"num_regions must be ≥ 1, got {num_regions}.")
        if d < 1:
            raise ValueError(f"d must be ≥ 1, got {d}.")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one element.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if temporal_downsample_factor < 1:
            raise ValueError(
                f"temporal_downsample_factor must be ≥ 1, got {temporal_downsample_factor}."
            )

        self.num_regions = num_regions
        self.d = d
        self.temporal_downsample_factor = temporal_downsample_factor

        # ------------------------------------------------------------------
        # Temporal downsampling (memory optimisation)
        # Applied before the channel-expanding projection to minimise peak
        # memory.  Uses average pooling to preserve signal smoothness.
        # ------------------------------------------------------------------
        if temporal_downsample_factor > 1:
            self.temporal_downsample: nn.Module = nn.AvgPool1d(
                kernel_size=temporal_downsample_factor,
                stride=temporal_downsample_factor,
                # ceil_mode=True ensures we don't drop trailing frames if
                # T is not exactly divisible by the factor.
                ceil_mode=True,
            )
        else:
            self.temporal_downsample = nn.Identity()

        # ------------------------------------------------------------------
        # Projection block  (B, R, T') → (B, R·d, T')
        # Groups: R groups, each with 1 input channel and d output channels.
        # ------------------------------------------------------------------
        self.projection = _GroupedConvBlock(
            in_channels=num_regions,
            out_channels=num_regions * d,
            kernel_size=kernel_sizes[0],
            groups=num_regions,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Residual blocks  (B, R·d, T') → (B, R·d, T')
        # Groups: R groups, each with d input channels and d output channels.
        # A learned skip connection is added after the convolution + BN so
        # that gradients can bypass deep stacks without degradation.
        # ------------------------------------------------------------------
        residual_blocks: list[nn.Module] = []
        for ks in kernel_sizes[1:]:
            residual_blocks.append(
                _ResidualGroupedBlock(
                    channels=num_regions * d,
                    kernel_size=ks,
                    groups=num_regions,
                    dropout=dropout,
                )
            )
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # ------------------------------------------------------------------
        # Temporal pooling: collapses variable-length T' → 1 via averaging.
        # ------------------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool1d(1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode per-region BOLD time series into fixed-length embeddings.

        Args:
            x: BOLD input tensor of shape ``(B, R, T)``.
               ``R`` must equal ``self.num_regions``.
               ``T`` can be any positive integer (variable scan length is
               handled by ``AdaptiveAvgPool1d``).

        Returns:
            Feature tensor of shape ``(B, R, d)``, where position ``[:, r, :]``
            contains the ``d``-dimensional embedding of region ``r``.

        Raises:
            RuntimeError: If ``x.shape[1] != self.num_regions``.
        """
        if x.shape[1] != self.num_regions:
            raise RuntimeError(
                f"Expected x.shape[1] == {self.num_regions} (num_regions), "
                f"got {x.shape[1]}."
            )

        # x: (B, R, T)
        # Conv1d treats dimension 1 as channels and dimension 2 as the
        # sequence length, so the input is already correctly shaped.

        # ── Early temporal downsampling (memory optimisation) ─────────────
        # Reduces T → T // factor BEFORE the projection expands channels,
        # cutting peak memory by ~factor× with minimal information loss.
        x = self.temporal_downsample(x)
        # x: (B, R, T')  where T' ≈ T // temporal_downsample_factor

        out = self.projection(x)
        # out: (B, R·d, T')   — projection from 1 to d features per region

        out = self.residual_blocks(out)
        # out: (B, R·d, T')   — residual refinement; T' unchanged (padding='same')

        out = self.pool(out)
        # out: (B, R·d, 1)   — temporal dimension collapsed to a single value

        out = out.squeeze(-1)
        # out: (B, R·d)

        # Reshape: grouped conv output is contiguous per region, so a plain
        # view maps d consecutive channels back to their originating region.
        batch_size = out.size(0)
        out = out.view(batch_size, self.num_regions, self.d)
        # out: (B, R, d)

        return out

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"num_regions={self.num_regions}, d={self.d}, "
            f"n_residual_blocks={len(self.residual_blocks)}, "
            f"temporal_downsample_factor={self.temporal_downsample_factor}"
        )