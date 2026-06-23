from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.config import BGTESREConfig

class BGTCCRELoss(nn.Module):
    def __init__(self,
                 cfg: BGTESREConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        
    def forward(
            self,
            logits: Tensor,
            y: Tensor
    ) -> Tensor:
        loss = F.cross_entropy(
            logits,
            y,
            label_smoothing=self.cfg.loss.label_smoothing
        )
        return loss