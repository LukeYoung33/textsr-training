"""Keep the 0921 optimization path while selecting native paired/degraded inputs."""
import torch
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
from .basic_text_line_y4uv_v4_gtdeg_model import BasicTextLineY4UVV4GTDegradationTrainingModel


@MODEL_REGISTRY.register()
class NativeRegion0921Model(BasicTextLineY4UVV4GTDegradationTrainingModel):
    @torch.no_grad()
    def feed_data(self, data):
        """@luna-comment
        Input: native RGB/mask batch with optional standard degradation kernels.
        Output: original packed model state, with selected paired samples substituted.
        Key Steps: invoke baseline feed, optionally replace synthesized inputs by real
        paired RGB, then repack. Validation unconditionally uses paired data.
        Key Params: paired_probability in [0,1]; separate CPU generator isolates selection.
        State/Dependencies: historical wrapper/loss state, local reproducible mixture RNG.
        """
        probability = float(self.opt.get('paired_probability', 0))
        if not 0 <= probability <= 1:
            raise ValueError('paired_probability must be within [0,1]')
        paired_only = not self.is_train or 'kernel1' not in data or probability == 1
        feed = {k: v for k, v in data.items() if k not in ('kernel1', 'kernel2', 'sinc_kernel')} if paired_only else data
        super().feed_data(feed)
        batch = self.lq.shape[0]
        if not hasattr(self, 'mixture_rng'):
            self.mixture_rng = torch.Generator().manual_seed(int(self.opt.get('manual_seed', 20260828)) + 75025)
        if paired_only:
            selection = torch.ones(batch, dtype=torch.bool)
        elif probability == 0:
            selection = torch.zeros(batch, dtype=torch.bool)
        else:
            selection = torch.rand(batch, generator=self.mixture_rng) < probability
            self.lq_rgb = torch.where(selection.to(self.device)[:, None, None, None],
                                      data['lq_rgb'].to(self.device), self.lq_rgb)
            self.lq = rgb_to_y4uv(self.lq_rgb)
        self.last_paired_selection = selection

    def optimize_parameters(self, current_iter):
        """@luna-comment Input: current iteration. Output: unchanged 0921 update plus routing log.
        Key Steps: delegate all losses/optimizer; append observed input-source fraction.
        Key Params: inherited schedule unchanged. State/Dependencies: historical optimizer/log.
        """
        super().optimize_parameters(current_iter)
        self.log_dict['input_paired_fraction'] = float(self.last_paired_selection.float().mean())
