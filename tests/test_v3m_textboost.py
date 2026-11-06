"""Check V3M graph equivalence and exact foreground-only coefficient changes."""
import copy
from pathlib import Path
import sys
import os
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.train_v3m_block256 import bootstrap
bootstrap(os.environ.get('V3M_TRAINING_REPO',str(ROOT)))
from realesrgan.archs.basic_text_line_r4_student_y4uv_v3m_arch import BasicTextLineR4StudentY4UVV3M
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.uv_structure_supervision_loss import UVStrongSupervisionLoss
from realesrgan.losses.text_region_boost_loss import TextBoostNativeMaskedL1Loss,TextBoostUVStrongSupervisionLoss


def test_native_text_gradient_and_background_invariance():
    """@luna-comment Input: distinct masked foreground/background errors. Output: exact gradient assertions.
    Key Steps: compare baseline and boosted backward; exercise per-sample weights and validity masking.
    Key Params: text1.5, fixed original denominator. State/Dependencies: CPU torch only.
    """
    torch.manual_seed(7)
    pred=torch.rand(2,6,8,8,requires_grad=True);target=torch.zeros_like(pred)
    text=torch.zeros(2,1,8,8);text[:,:,:4]=1
    valid=torch.ones_like(text);valid[:,:,:,-1]=0
    weight=torch.tensor([1.,.7]);kwargs=dict(fg_weight=1.2,bg_weight=.8)
    base=Y4UVNativeMaskedL1LossV3(**kwargs)(pred,target,valid,text,weight=weight)
    boosted=TextBoostNativeMaskedL1Loss(**kwargs)(pred,target,valid,text,weight=weight)
    g0=torch.autograd.grad(base,pred,retain_graph=True)[0];g1=torch.autograd.grad(boosted,pred)[0]
    torch.testing.assert_close(g1,g0*(1+.5*text),atol=1e-8,rtol=1e-6)
    neutral=TextBoostNativeMaskedL1Loss(text_multiplier=1,**kwargs)(pred,target,valid,text,weight=weight)
    torch.testing.assert_close(neutral,base,atol=0,rtol=0)


def test_uv_preserves_outside_gradients_and_scales_all_text():
    """@luna-comment Input: structured UV errors and mixed/all-text masks. Output: regional gradient proof.
    Key Steps: compare base/boost gradients outside text; all-text objective must be exactly1.5 baseline.
    Key Params: small Gaussian scales, nonzero regional+multiscale terms. State/Dependencies: CPU torch.
    """
    torch.manual_seed(9);pred=torch.rand(1,6,16,16,requires_grad=True);target=torch.rand_like(pred)
    valid=torch.ones(1,1,16,16);text=torch.zeros_like(valid);text[:,:,4:12,4:12]=1
    opt=dict(ring_radius=2,sigmas=(.5,1.,1.5))
    baseline=UVStrongSupervisionLoss(**opt);boost=TextBoostUVStrongSupervisionLoss(**opt)
    b=baseline(pred,target,valid,text);v=boost(pred,target,valid,text)
    g0=torch.autograd.grad(b,pred,retain_graph=True)[0];g1=torch.autograd.grad(v,pred,retain_graph=True)[0]
    torch.testing.assert_close((g1-g0)*(1-text),torch.zeros_like(pred),atol=1e-8,rtol=0)
    torch.testing.assert_close(boost(pred,target,valid,valid),1.5*baseline(pred,target,valid,valid),atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(TextBoostUVStrongSupervisionLoss(text_multiplier=1,**opt)(pred,target,valid,text),b,atol=0,rtol=0)


def test_v3m_nonzero_fusion_and_fixed_widths():
    """@luna-comment Input: fresh small-canvas V3M and nonzero projection probe. Output: depth/width/fusion checks.
    Key Steps: verify initial identity, widen all four levels, activate residual, compare exact fused inference.
    Key Params: three levels, depth1, Haar enabled. State/Dependencies: existing fusion implementation, CPU torch.
    """
    from tools.export_basic_text_line_y4uv_v3_checkpoint_deploy_onnx import fuse_reparameterized_modules
    torch.manual_seed(11);model=BasicTextLineR4StudentY4UVV3M(canvas_hw=(32,32)).eval()
    assert model.feature_widths==(4,8,16,32) and model.backbone.layer_num==3
    assert [model.backbone.conv0.out_planes]+[getattr(model.backbone,'enc'+str(i)).module_list[-1].out_planes for i in (1,2,3)]==[4,8,16,32]
    data=torch.rand(2,6,16,16)
    torch.testing.assert_close(model(data),data,atol=0,rtol=0)
    with torch.no_grad():model.residual_projection.weight.normal_(0,.02)
    fused=copy.deepcopy(model);counts=fuse_reparameterized_modules(fused)
    assert counts=={'ECB':6,'ECB2':3}
    assert (model(data)-data).abs().max()>1e-5
    torch.testing.assert_close(model(data),fused(data),atol=1e-5,rtol=1e-5)
