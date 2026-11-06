"""CPU structure, gradient, compatibility and resumability checks for V3L KD."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_v3m_block256 import bootstrap
bootstrap(str(ROOT))
from realesrgan.archs.basic_text_line_r4_student_y4uv_v3m_arch import BasicTextLineR4StudentY4UVV3M
from realesrgan.archs.basic_text_line_r4_teacher_y4uv_v3l_arch import BasicTextLineR4TeacherY4UVV3L, FEATURE_TAPS, TEACHER_WIDTHS
from realesrgan.archs.ecbsr_Unet_arch_server import ECB, ECB2
from realesrgan.models.v3_feature_distillation import V3DistillationTrainer, V3FeatureDistillation, _capture
from tools.profile_v3l_teachers import profile

torch.set_num_threads(1)


def pair(target=16):
    """@luna-comment
    Input: teacher target label. Output: deterministic small-canvas teacher/student.
    Key Steps: seed and construct original student and isolated teacher.
    Key Params: canvas32, CPU. State/Dependencies: torch RNG reset for reproducible fixtures.
    """
    torch.manual_seed(17)
    return BasicTextLineR4StudentY4UVV3M(canvas_hw=(32, 32)), BasicTextLineR4TeacherY4UVV3L(compute_ratio=target, canvas_hw=(32, 32))


@pytest.mark.parametrize('target', TEACHER_WIDTHS)
def test_width_only_topology_compute_and_nonzero_fusion(target):
    """@luna-comment
    Input: each candidate. Output: topology, measured MAC and nonzero fusion assertions.
    Key Steps: compare module type graphs; activate residual; compare train/eval/fused outputs.
    Key Params: 5% MAC tolerance, 2e-5 numerical tolerance. State/Dependencies: CPU torch/exporter.
    """
    from tools.export_basic_text_line_y4uv_v3_checkpoint_deploy_onnx import fuse_reparameterized_modules
    student, teacher = pair(target)
    assert [(n, type(m)) for n, m in student.backbone.named_modules()] == [(n, type(m)) for n, m in teacher.backbone.named_modules()]
    blocks = [m for m in teacher.modules() if isinstance(m, (ECB, ECB2))]
    assert len(blocks) == 9
    assert all(m.depth_multiplier == (1 if m is teacher.backbone.out else 2) for m in blocks)
    assert all(len(getattr(teacher.backbone, n).module_list) == 1 for n in ('enc1', 'enc2', 'enc3', 'dec1', 'dec2', 'dec3'))
    assert teacher.feature_widths == TEACHER_WIDTHS[target]
    ratio = profile(teacher)['convolution_macs'] / profile(student)['convolution_macs']
    assert abs(ratio / target - 1) <= .05
    x = torch.rand(1, 6, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(teacher(x), x, atol=0, rtol=0)
        teacher.residual_projection.weight.normal_(0, .01)
        training_output = teacher(x)
        assert (training_output - x).abs().max() > 1e-6
        teacher.eval()
        evaluation_output = teacher(x)
        torch.testing.assert_close(training_output, evaluation_output, atol=2e-5, rtol=2e-5)
        deployed = copy.deepcopy(teacher)
        assert fuse_reparameterized_modules(deployed) == {'ECB': 6, 'ECB2': 3}
        torch.testing.assert_close(deployed(x), evaluation_output, atol=2e-5, rtol=2e-5)


def test_stage_geometry_loss_formula_and_hook_cleanup():
    """@luna-comment
    Input: true model taps and synthetic aligned targets. Output: loss arithmetic and cleanup proof.
    Key Steps: capture eight stages; test exact zero, unit perturbation, detach and shape failures.
    Key Params: per-tap mean, equal weights. State/Dependencies: temporary scoped hooks.
    """
    student, teacher = pair()
    objective = V3FeatureDistillation(student, teacher)
    x = torch.rand(1, 6, 16, 16)
    with _capture(student) as sf, _capture(teacher) as tf:
        student(x); teacher.eval()(x)
        assert [tuple(sf[n].shape[-2:]) for n in FEATURE_TAPS] == [(16,16),(8,8),(4,4),(2,2),(2,2),(4,4),(8,8),(16,16)]
        assert all(sf[n].shape[2:] == tf[n].shape[2:] for n in FEATURE_TAPS)
        aligned = {n: objective.adapters[n](sf[n]).detach().requires_grad_() for n in FEATURE_TAPS}
        zero, _ = objective(sf, aligned)
        assert zero.item() == 0
        shifted = {n: t + 1 for n, t in aligned.items()}
        loss, logs = objective(sf, shifted)
        torch.testing.assert_close(loss, torch.tensor(1.))
        assert len(logs) == 8
        loss.backward()
        assert all(t.grad is None for t in aligned.values())
        with pytest.raises(ValueError, match='Missing'):
            objective({}, tf)
        bad = dict(tf); bad['conv0'] = bad['conv0'][:, :, :-1]
        with pytest.raises(ValueError, match='mismatch'):
            objective(sf, bad)
    assert not sf and not tf
    assert all(not m._forward_hooks for m in student.modules())
    with pytest.raises(RuntimeError, match='Duplicate'):
        with _capture(student):
            student(x); student(x)
    assert all(not m._forward_hooks for m in student.modules())


def test_joint_update_frozen_teacher_and_plain_student_export():
    """@luna-comment
    Input: real student/teacher and packed random pair. Output: frozen teacher and joint update proof.
    Key Steps: one warmup step; inspect gradients and weight changes; strict-load student-only export.
    Key Params: positive feature weight. State/Dependencies: Adam, unchanged V3M forward.
    """
    student, teacher = pair()
    original = copy.deepcopy(student.state_dict())
    frozen = copy.deepcopy(teacher.state_dict())
    engine = V3DistillationTrainer(student, teacher, warmup_iters=2)
    adapters = copy.deepcopy(engine.features.state_dict())
    logs = engine.step(torch.rand(1,6,16,16), torch.rand(1,6,16,16), F.l1_loss)
    assert logs['feature_scale'] == .5 and logs['feature'] > 0
    assert not teacher.training
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    for n, value in teacher.state_dict().items():
        torch.testing.assert_close(value, frozen[n], atol=0, rtol=0)
    assert any(not torch.equal(value, original[n]) for n, value in student.state_dict().items())
    assert any(not torch.equal(value, adapters[n]) for n, value in engine.features.state_dict().items())
    for group in (student.parameters(), engine.features.parameters()):
        gradients = [p.grad for p in group if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    clone, _ = pair()
    clone.load_state_dict(engine.student_checkpoint()['params'], strict=True)
    x = torch.rand(1,6,16,16)
    torch.testing.assert_close(clone.eval()(x), student.eval()(x), atol=0, rtol=0)
    assert all(not m._forward_hooks for m in student.modules())
    assert all(not m._forward_hooks for m in teacher.modules())


def test_disabled_feature_matches_original_optimizer_step():
    """@luna-comment
    Input: identical student weights and plain L1 objective. Output: exact disabled-KD equivalence.
    Key Steps: compare original Adam step and engine step including gradients and all weights.
    Key Params: feature_weight0, lr1e-4. State/Dependencies: CPU deterministic torch.
    """
    student, teacher = pair(); control = copy.deepcopy(student)
    optimizer = torch.optim.Adam(control.parameters(), lr=1e-4)
    engine = V3DistillationTrainer(student, teacher, feature_weight=0)
    x, target = torch.rand(1,6,16,16), torch.rand(1,6,16,16)
    optimizer.zero_grad(set_to_none=True)
    base = F.l1_loss(control(x), target); base.backward(); optimizer.step()
    logs = engine.step(x, target, F.l1_loss)
    torch.testing.assert_close(logs['total'], base.detach(), atol=0, rtol=0)
    for a, b in zip(control.parameters(), student.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert all(p.grad is None for p in engine.features.parameters())


def test_exact_next_step_resume_with_scheduler(tmp_path):
    """@luna-comment
    Input: saved full engine state after an update. Output: identical resumed next-step weights.
    Key Steps: disk round-trip teacher/student/adapters/Adam/scheduler/RNG; compare next update.
    Key Params: StepLR plus nontrivial feature warmup. State/Dependencies: temporary checkpoint only.
    """
    student, teacher = pair()
    engine = V3DistillationTrainer(student, teacher, warmup_iters=3)
    scheduler = torch.optim.lr_scheduler.StepLR(engine.optimizer, step_size=1, gamma=.9)
    x, target = torch.rand(1,6,16,16), torch.rand(1,6,16,16)
    engine.step(x,target,F.l1_loss); scheduler.step()
    path = tmp_path / 'resume.pth'; torch.save(engine.state_dict(scheduler), path)
    a, b = pair(); resumed = V3DistillationTrainer(a,b,warmup_iters=3)
    other_scheduler = torch.optim.lr_scheduler.StepLR(resumed.optimizer, step_size=1,gamma=.9)
    resumed.load_state_dict(torch.load(path, weights_only=True), other_scheduler)
    logs = engine.step(x,target,F.l1_loss); other = resumed.step(x,target,F.l1_loss)
    torch.testing.assert_close(logs['total'], other['total'], atol=0, rtol=0)
    for name, value in engine.student.state_dict().items():
        torch.testing.assert_close(value, resumed.student.state_dict()[name], atol=0, rtol=0)
    for name, value in engine.features.state_dict().items():
        torch.testing.assert_close(value, resumed.features.state_dict()[name], atol=0, rtol=0)
    assert resumed.iteration == engine.iteration == 2
    assert scheduler.get_last_lr() == other_scheduler.get_last_lr()


def test_invalid_capacity_and_feature_settings():
    """@luna-comment
    Input: unsupported capacity, depth and weight settings. Output: explicit validation failures.
    Key Steps: reject malformed configs and shared teacher/student parameters.
    Key Params: width-only matrix is fixed. State/Dependencies: no optimizer steps.
    """
    with pytest.raises(ValueError): BasicTextLineR4TeacherY4UVV3L(compute_ratio=100)
    with pytest.raises(ValueError): BasicTextLineR4TeacherY4UVV3L(depth_factor=2)
    student, teacher = pair()
    with pytest.raises(ValueError): V3FeatureDistillation(student,teacher,[1]*7)
    with pytest.raises(ValueError): V3DistillationTrainer(student,teacher,feature_weight=float('nan'))
    with pytest.raises(ValueError): V3DistillationTrainer(student,student)
