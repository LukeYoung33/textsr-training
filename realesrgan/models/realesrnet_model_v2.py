"""Independent lightweight BasicSR model wrapper for BASIC 2.0."""

from __future__ import annotations

from copy import deepcopy

import torch
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.losses import build_loss


@MODEL_REGISTRY.register()
class RealESRNetModelV2(SRModel):
    """Fixed-shape V2 model with explicit valid/text/blend masks."""

    @staticmethod
    def _build_ema_network(net, device):
        """@shanlan-comment Clone a generator for frozen EMA inference without changing legacy models."""
        ema = deepcopy(net.module if hasattr(net, "module") else net).to(device)
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        return ema

    def init_training_settings(self):
        """@shanlan-comment Build optimizer and V2 losses without legacy queue/degradation behavior."""
        train_opt = self.opt.get("train", {})
        self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device) if train_opt.get("pixel_opt") else None
        self.cri_y = build_loss(train_opt["y_charbonnier_opt"]).to(self.device) if train_opt.get("y_charbonnier_opt") else None
        self.net_g.train()
        self.ema_decay = float(train_opt.get("ema_decay", 0.0))
        if self.ema_decay > 0:
            self.net_g_ema = self.__class__._build_ema_network(self.net_g, self.device)
            self.net_g_ema.eval()
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        """@shanlan-comment Move fixed tensors and independent masks to the active device."""
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)
        self.valid_mask = data.get("valid_mask", torch.ones_like(self.gt[:, :1])).to(self.device)
        self.text_mask = data.get("text_mask", torch.zeros_like(self.valid_mask)).to(self.device)
        self.blend_mask = data.get("blend_mask", self.valid_mask).to(self.device)

    def optimize_parameters(self, current_iter):
        """@shanlan-comment Run one padding-aware V2 optimization step and expose scalar losses."""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        total = self.output.sum() * 0.0
        loss_dict = {}
        if self.cri_pix is not None:
            pix = self.cri_pix(self.output, self.gt, valid_mask=self.valid_mask, text_mask=self.text_mask)
            total = total + pix; loss_dict["l_pix"] = pix
        if self.cri_y is not None:
            yloss = self.cri_y(self.output, self.gt, valid_mask=self.valid_mask, text_mask=self.text_mask)
            total = total + yloss; loss_dict["l_y"] = yloss
        total.backward(); self.optimizer_g.step()
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        self.log_dict = self.reduce_loss_dict(loss_dict)
