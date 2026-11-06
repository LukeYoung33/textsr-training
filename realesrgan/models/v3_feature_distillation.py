"""Training-only feature objective and resumable single-device step engine.

Base losses are supplied by the selected experiment; no legacy model or loss
recipe is replaced. Ordinary V3M checkpoints contain no adapters or teacher.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from realesrgan.archs.basic_text_line_r4_teacher_y4uv_v3l_arch import FEATURE_TAPS


def _tap_widths(network):
    """@luna-comment
    Input: bare V3M/V3L module. Output: ordered channel counts for eight stage taps.
    Key Steps: resolve each stage and its terminal ECB channel count.
    Key Params: FEATURE_TAPS fixes post-activation/post-skip correspondence.
    State/Dependencies: existing backbone module structure; no forward or mutations.
    """
    widths = {}
    for name in FEATURE_TAPS:
        stage = getattr(network.backbone, name)
        block = stage.module_list[-1] if hasattr(stage, 'module_list') else stage
        widths[name] = block.out_planes
    return widths


@contextmanager
def _capture(network):
    """@luna-comment
    Input: one network. Output: scoped mapping of live stage tensors.
    Key Steps: install eight hooks; reject duplicates; remove hooks and clear on all exits.
    Key Params: fixed stage tap names. State/Dependencies: temporary forward hooks only.
    """
    features, handles = {}, []

    def make_hook(name):
        """@luna-comment
        Input: stage name. Output: hook retaining the differentiable tensor once.
        Key Steps: close over the scoped mapping. Key Params: unique stage name.
        State/Dependencies: mapping is cleared by the enclosing context manager.
        """
        def hook(module, inputs, output):
            """@luna-comment
            Input: torch forward-hook arguments. Output: captured tensor or explicit error.
            Key Steps: reject duplicates and non-tensor stage outputs before retaining.
            Key Params: stage name. State/Dependencies: context-local feature mapping.
            """
            if name in features or not isinstance(output, torch.Tensor):
                raise RuntimeError(f'Duplicate or invalid feature tap: {name}')
            features[name] = output
        return hook

    try:
        for name in FEATURE_TAPS:
            handles.append(getattr(network.backbone, name).register_forward_hook(make_hook(name)))
        yield features
    finally:
        for handle in handles:
            handle.remove()
        features.clear()


class V3FeatureDistillation(nn.Module):
    """Independent per-stage adapters; teacher tensors never receive gradients."""

    def __init__(self, student, teacher, stage_weights=None):
        """@luna-comment
        Input: student/teacher structures and optional eight positive stage weights.
        Output: trainable channel adapters, without registering either network as a child.
        Key Steps: inspect tap widths; validate weights; initialize independent 1x1 projections.
        Key Params: per-stage means and normalized weights prevent channel-count domination.
        State/Dependencies: torch initialization; only adapters and weights enter state_dict.
        """
        super().__init__()
        sw, tw = _tap_widths(student), _tap_widths(teacher)
        weights = [1.0] * len(FEATURE_TAPS) if stage_weights is None else list(stage_weights)
        if len(weights) != len(FEATURE_TAPS) or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError('stage_weights must contain eight finite positive values')
        self.adapters = nn.ModuleDict({name: nn.Conv2d(sw[name], tw[name], 1) for name in FEATURE_TAPS})
        self.register_buffer('stage_weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, student_features, teacher_features):
        """@luna-comment
        Input: complete ordered/mapped BCHW stage features. Output: mean loss and detached tap logs.
        Key Steps: validate keys and spatial sizes; project student; detach teacher; average L1.
        Key Params: normalized positive stage weights. State/Dependencies: adapter parameters.
        """
        if set(student_features) != set(FEATURE_TAPS) or set(teacher_features) != set(FEATURE_TAPS):
            raise ValueError('Missing or unexpected feature taps')
        losses = []
        for name in FEATURE_TAPS:
            student, teacher = student_features[name], teacher_features[name]
            if student.ndim != 4 or teacher.ndim != 4 or student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
                raise ValueError(f'Feature spatial/batch mismatch at {name}')
            losses.append(F.l1_loss(self.adapters[name](student), teacher.detach()))
        values = torch.stack(losses)
        loss = (values * self.stage_weights).sum() / self.stage_weights.sum()
        return loss, {f'feature/{name}': value.detach() for name, value in zip(FEATURE_TAPS, losses)}


class V3DistillationTrainer:
    """Single-device engine accepting the experiment's unchanged base-loss callable.

    A configured teacher is required to be strict-loaded by the caller before
    construction. A newly initialized teacher is useful only for correctness tests.
    Distributed/AMP/EMA execution is not silently emulated by this engine.
    """

    def __init__(self, student, teacher, *, feature_weight=1.0, warmup_iters=5000,
                 stage_weights=None, optimizer_options=None):
        """@luna-comment
        Input: colocated student/teacher and Adam/feature settings. Output: joint optimizer engine.
        Key Steps: freeze teacher; build adapters on student device; create Adam with both groups.
        Key Params: nonnegative feature weight and integer warmup; default lr=1e-4.
        State/Dependencies: caller owns strict teacher loading and base-loss/data recipe.
        """
        if not math.isfinite(feature_weight) or feature_weight < 0:
            raise ValueError('feature_weight must be finite and nonnegative')
        if isinstance(warmup_iters, bool) or not isinstance(warmup_iters, int) or warmup_iters < 0:
            raise ValueError('warmup_iters must be a nonnegative integer')
        if student is teacher or {id(p) for p in student.parameters()} & {id(p) for p in teacher.parameters()}:
            raise ValueError('Teacher and student must not share parameters')
        parameter = next(student.parameters())
        if any(p.device != parameter.device or p.dtype != parameter.dtype for p in teacher.parameters()):
            raise ValueError('Teacher and student must have the same device and dtype')
        self.student, self.teacher = student, teacher
        self.teacher.requires_grad_(False).eval()
        for parameter_t in self.teacher.parameters():
            parameter_t.grad = None
        self.features = V3FeatureDistillation(student, teacher, stage_weights).to(parameter)
        options = dict(lr=1e-4) if optimizer_options is None else dict(optimizer_options)
        self.optimizer = torch.optim.Adam([
            {'params': student.parameters()}, {'params': self.features.parameters()},
        ], **options)
        self.feature_weight, self.warmup_iters = float(feature_weight), warmup_iters
        self.iteration = 0

    def step(self, lq, target, base_loss):
        """@luna-comment
        Input: packed LQ/GT and callable (prediction,target)->scalar base objective.
        Output: detached scalar logs after one joint update; teacher remains unchanged.
        Key Steps: scoped taps, no-grad teacher, student, base plus ramped feature loss, backward/step.
        Key Params: next-iteration warmup; zero feature weight skips teacher/adapters entirely.
        State/Dependencies: updates student/adapters/Adam/iteration; base-loss closure owns masks/R1.
        """
        self.student.train()
        self.teacher.eval()
        self.optimizer.zero_grad(set_to_none=True)
        next_iter = self.iteration + 1
        scale = self.feature_weight * (min(next_iter / self.warmup_iters, 1) if self.warmup_iters else 1)
        logs = {}
        if scale:
            with _capture(self.teacher) as tf, _capture(self.student) as sf:
                with torch.no_grad():
                    self.teacher(lq)
                prediction = self.student(lq)
                feature_loss, logs = self.features(sf, tf)
        else:
            prediction = self.student(lq)
            feature_loss = prediction.new_zeros(())
        base = base_loss(prediction, target)
        if base.ndim != 0:
            raise ValueError('base_loss must return a scalar')
        total = base + scale * feature_loss
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite distillation objective')
        total.backward()
        self.optimizer.step()
        self.iteration = next_iter
        logs.update(base=base.detach(), feature=feature_loss.detach(), total=total.detach(), feature_scale=scale)
        return logs

    def state_dict(self, scheduler=None):
        """@luna-comment
        Input: optional caller-owned scheduler. Output: complete engine state for torch.save.
        Key Steps: include student/teacher/adapters/Adam/iteration and feature hyperparameters.
        Key Params: scheduler state when supplied. State/Dependencies: returned tensors alias live state.
        """
        return dict(student=self.student.state_dict(), teacher=self.teacher.state_dict(),
                    features=self.features.state_dict(), optimizer=self.optimizer.state_dict(),
                    iteration=self.iteration, feature_weight=self.feature_weight,
                    warmup_iters=self.warmup_iters,
                    scheduler=None if scheduler is None else scheduler.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    def load_state_dict(self, state, scheduler=None):
        """@luna-comment
        Input: trusted engine checkpoint and matching optional scheduler. Output: restored engine.
        Key Steps: reject schedule mismatch; strict-load networks/adapters; restore Adam/iteration/RNG.
        Key Params: feature settings must agree. State/Dependencies: global torch RNG restored;
        dataset/Python/NumPy RNG remain the outer training pipeline's responsibility.
        """
        if state['feature_weight'] != self.feature_weight or state['warmup_iters'] != self.warmup_iters:
            raise ValueError('Checkpoint feature schedule differs from engine configuration')
        if (state['scheduler'] is None) != (scheduler is None):
            raise ValueError('Checkpoint and caller scheduler presence must agree')
        self.student.load_state_dict(state['student'], strict=True)
        self.teacher.load_state_dict(state['teacher'], strict=True)
        self.features.load_state_dict(state['features'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.iteration = int(state['iteration'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None:
            if not torch.cuda.is_available():
                raise ValueError('CUDA RNG checkpoint requires CUDA for exact resume')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.teacher.requires_grad_(False).eval()

    def student_checkpoint(self):
        """@luna-comment
        Input: none. Output: independent BasicSR-compatible student-only CPU weights.
        Key Steps: clone tensors under params; exclude adapters and teacher.
        Key Params: unchanged V3M state names. State/Dependencies: current student weights.
        """
        return {'params': {name: value.detach().cpu().clone() for name, value in self.student.state_dict().items()}}
