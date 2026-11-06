# Width-only V3L teachers and corresponding-stage feature distillation

## Delivered scope
Four V3M-derived teacher architectures, isolated architecture fragments, a
training-only feature objective and single-device step engine. Existing V3M
constructors, forward outputs, inference checkpoints, datasets and BasicSR model
loops are unchanged. This is preparation for the later experiment matrix, not a
formal training launch or a validated quality result.

## Teacher matrix
| Target label | Stage widths | Measured deployed Conv2d MAC ratio |
|---|---|---|
| V3L-16x | 20/40/80/160 | 16.5466 |
| V3L-32x | 28/56/112/224 | 31.6198 |
| V3L-64x | 40/80/160/320 | 63.3544 |
| V3L-128x | 57/114/228/456 | 127.0556 |

All preserve the three encoder/decoder levels, depth_factor=1, internal
scale_factor=2, nine ECB/ECB2 blocks, activation family, skip topology and native
six-channel residual boundary. The wider channels also widen existing internal
branches proportionally; no block repetition or expansion ratio is added.
Integer stage widths retain the 1:2:4:8 pattern. Target labels are approximate,
with measured errors below 5%. The initial square-root width estimates were
replaced after accounting for fixed input/output layers and skip projections.

The reference is V3M widths4/8/16/32. Measurement uses canvas256x256, packed
1x6x128x128 FP32, with_proj=true and the actual fused deployment model. One MAC
is one multiply-accumulate. Counts include executed dense/depthwise/pointwise
Conv2d operations and residual projection, excluding fixed Haar transforms,
interpolation, activations, bias arithmetic and kernel-fusion setup. They are
not total operator FLOPs, training computation, training latency or peak memory.
The report also records training/deployed parameter counts and summed stage
output bytes; the latter is not a training-memory estimate.

Reproduce from the repository root:
```powershell
python tools/profile_v3l_teachers.py
```
Report: evaluation/v3l_feature_distillation_20261106/profiles.json.
Architecture fragments: options/v3l_feature_distillation/teacher_{16,32,64,128}x.yml.
These fragments must be merged with an approved training recipe; they are not
standalone BasicSR configs.

## Feature mapping and loss
| Tap | Packed-space resolution | Student channels | Teacher channels |
|---|---|---|---|
| conv0 | H x W | 4 | w |
| enc1 | H/2 x W/2 | 8 | 2w |
| enc2 | H/4 x W/4 | 16 | 4w |
| enc3 | H/8 x W/8 | 32 | 8w |
| encdec | H/8 x W/8 | 32 | 8w |
| dec1 | H/4 x W/4 | 16 | 4w |
| dec2 | H/2 x W/2 | 8 | 2w |
| dec3 | H x W | 4 | w |

Decoder taps are after skip addition and activation. Both models see exactly
the same augmented packed LQ tensor. No spatial resizing is used. Each tap has
its own trainable 1x1 student-to-teacher projection A[k]:

L[k] = mean(abs(A[k](student[k]) - stop_gradient(teacher[k])))
Lfeature = sum(weight[k] * L[k]) / sum(weight[k])
Ltotal = Lbase + feature_scale(iteration) * Lfeature

All eight stage weights initially equal1. Each tap uses its own element mean.
Raw stage values are logged; no implicit feature normalization is applied.
Feature weight1 and linear5K warmup are proposed pilot defaults, not tuned
quality recommendations. Warmup uses min(next_iteration/5000,1). A zero feature
weight bypasses the teacher and adapters and matches the ordinary student
optimizer update. No output distillation is enabled.

## Engine integration
Explicitly bootstrap the existing registrations before importing extensions in
an isolated local process. Example construction (the base-loss callable and
trained teacher weights must be supplied by the experiment):
```python
from tools.train_v3m_block256 import bootstrap
bootstrap(repository_root)
from realesrgan.archs.basic_text_line_r4_student_y4uv_v3m_arch import BasicTextLineR4StudentY4UVV3M
from realesrgan.archs.basic_text_line_r4_teacher_y4uv_v3l_arch import BasicTextLineR4TeacherY4UVV3L
from realesrgan.models.v3_feature_distillation import V3DistillationTrainer

student = BasicTextLineR4StudentY4UVV3M(canvas_hw=(256, 256)).to(device)
teacher = BasicTextLineR4TeacherY4UVV3L(compute_ratio=32, canvas_hw=(256, 256)).to(device)
teacher.load_state_dict(selected_teacher_state, strict=True)
engine = V3DistillationTrainer(student, teacher, feature_weight=1.0,
                               warmup_iters=5000, optimizer_options={"lr": 1e-4})
# Closure binds masks and any approved R1 inputs for this same batch.
logs = engine.step(packed_lq, packed_gt, selected_base_loss_callable)
# If configured, the outer loop steps the scheduler once after engine.step.
resume_state = engine.state_dict(scheduler)
student_only = engine.student_checkpoint()  # {'params': original V3M keys}
```

The engine does not define or alter L1-R1_ori. The later recipe integration must
provide that exact base objective as a scalar-returning callable. Freeze its
configuration and compare it against the non-distilled baseline. Teacher weights
must be trained, selected and strict-loaded with their provenance recorded;
random teacher weights used in tests do not constitute a usable teacher.

Teacher is always frozen/eval/no-grad and excluded from optimizer and EMA.
Student and adapters are both in Adam. Scoped hooks are removed and feature
mappings cleared after each capture, including exception paths. No hooks or
adapters are installed in exported student checkpoints.

Save full engine state with torch.save for resume, and reconstruct the same
architectures/settings before load_state_dict. The engine saves student,
teacher, adapters, Adam, iteration, optional scheduler and torch/CUDA RNG.
The outer data pipeline owns sampler/Python/NumPy RNG and data position. Scheduler
stepping is also owned by the outer loop. This engine is single-device FP32;
AMP, DDP, EMA, multi-device and full BasicSR data/resume integration are not
implemented or claimed. Those require a separate verified integration before
formal GPU launch.

## Validation and remaining experiments
Focused tests exercise all four topologies/MAC targets, nonzero train/eval/fused
parity, exact eight-stage geometry, feature-loss arithmetic and teacher detach,
finite gradients and joint updates, hook cleanup on errors, unchanged strict
student export, zero-feature equivalence, and exact next-step optimizer/scheduler
resume. Only CPU is available locally. Teacher convergence, student image
quality, GPU throughput and peak memory remain unvalidated.

Before formal500K runs: bind trusted GT, freeze L1-only/L1-R1_ori recipes, select
teacher checkpoints and initialization sources, integrate the approved training
pipeline, and validate actual CUDA execution and memory. This task does not
change or launch the original pretrain/scratch experiment matrix.
