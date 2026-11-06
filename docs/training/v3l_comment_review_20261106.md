# V3L Function Comment Review

Status: approved by the user on 2026-11-06 (explicit OK following the review handoff).

Core review points: width-only architecture; eight post-stage taps; teacher freeze;
per-tap mean L1; exact base-loss preservation; joint optimizer; resume versus
student-only export; profiling convention. The reused Conv2D/exporter edits are
comments only. Exporter comment refinement passed an AST comparison excluding
docstrings, confirming unchanged executable statements.

All functions below are author-reviewed. The user approved the presented contracts and comments. This document lists the actual source
docstrings; it is not a substitute for reviewing implementation semantics.

## realesrgan/archs/basic_text_line_r4_teacher_y4uv_v3l_arch.py

### BasicTextLineR4TeacherY4UVV3L.__init__ (line 22)

```text
@luna-comment
Input: one of four MAC target labels and unchanged V3 canvas/boundary settings.
Output: six-channel teacher with the original residual/Haar forward contract.
Key Steps: validate target; initialize V3 boundaries; replace only its backbone.
Key Params: fixed three levels, depth1, expansion2 and target-specific widths.
State/Dependencies: existing V3 and ECB implementations; random parameter initialization.
```

## realesrgan/models/v3_feature_distillation.py

### _tap_widths (line 16)

```text
@luna-comment
Input: bare V3M/V3L module. Output: ordered channel counts for eight stage taps.
Key Steps: resolve each stage and its terminal ECB channel count.
Key Params: FEATURE_TAPS fixes post-activation/post-skip correspondence.
State/Dependencies: existing backbone module structure; no forward or mutations.
```

### _capture (line 32)

```text
@luna-comment
Input: one network. Output: scoped mapping of live stage tensors.
Key Steps: install eight hooks; reject duplicates; remove hooks and clear on all exits.
Key Params: fixed stage tap names. State/Dependencies: temporary forward hooks only.
```

### make_hook (line 40)

```text
@luna-comment
Input: stage name. Output: hook retaining the differentiable tensor once.
Key Steps: close over the scoped mapping. Key Params: unique stage name.
State/Dependencies: mapping is cleared by the enclosing context manager.
```

### V3FeatureDistillation.__init__ (line 70)

```text
@luna-comment
Input: student/teacher structures and optional eight positive stage weights.
Output: trainable channel adapters, without registering either network as a child.
Key Steps: inspect tap widths; validate weights; initialize independent 1x1 projections.
Key Params: per-stage means and normalized weights prevent channel-count domination.
State/Dependencies: torch initialization; only adapters and weights enter state_dict.
```

### V3FeatureDistillation.forward (line 86)

```text
@luna-comment
Input: complete ordered/mapped BCHW stage features. Output: mean loss and detached tap logs.
Key Steps: validate keys and spatial sizes; project student; detach teacher; average L1.
Key Params: normalized positive stage weights. State/Dependencies: adapter parameters.
```

### V3DistillationTrainer.__init__ (line 113)

```text
@luna-comment
Input: colocated student/teacher and Adam/feature settings. Output: joint optimizer engine.
Key Steps: freeze teacher; build adapters on student device; create Adam with both groups.
Key Params: nonnegative feature weight and integer warmup; default lr=1e-4.
State/Dependencies: caller owns strict teacher loading and base-loss/data recipe.
```

### V3DistillationTrainer.step (line 142)

```text
@luna-comment
Input: packed LQ/GT and callable (prediction,target)->scalar base objective.
Output: detached scalar logs after one joint update; teacher remains unchanged.
Key Steps: scoped taps, no-grad teacher, student, base plus ramped feature loss, backward/step.
Key Params: next-iteration warmup; zero feature weight skips teacher/adapters entirely.
State/Dependencies: updates student/adapters/Adam/iteration; base-loss closure owns masks/R1.
```

### V3DistillationTrainer.state_dict (line 177)

```text
@luna-comment
Input: optional caller-owned scheduler. Output: complete engine state for torch.save.
Key Steps: include student/teacher/adapters/Adam/iteration and feature hyperparameters.
Key Params: scheduler state when supplied. State/Dependencies: returned tensors alias live state.
```

### V3DistillationTrainer.load_state_dict (line 191)

```text
@luna-comment
Input: trusted engine checkpoint and matching optional scheduler. Output: restored engine.
Key Steps: reject schedule mismatch; strict-load networks/adapters; restore Adam/iteration/RNG.
Key Params: feature settings must agree. State/Dependencies: global torch RNG restored;
dataset/Python/NumPy RNG remain the outer training pipeline's responsibility.
```

### V3DistillationTrainer.student_checkpoint (line 216)

```text
@luna-comment
Input: none. Output: independent BasicSR-compatible student-only CPU weights.
Key Steps: clone tensors under params; exclude adapters and teacher.
Key Params: unchanged V3M state names. State/Dependencies: current student weights.
```

### hook (line 46)

```text
@luna-comment
Input: torch forward-hook arguments. Output: captured tensor or explicit error.
Key Steps: reject duplicates and non-tensor stage outputs before retaining.
Key Params: stage name. State/Dependencies: context-local feature mapping.
```

## tools/profile_v3l_teachers.py

### profile (line 14)

```text
@luna-comment
Input: CPU V3M/V3L model at its declared canvas. Output: parameter/MAC/tap-byte report.
Key Steps: fuse a copy; count executed Conv2d MACs by output shape; run all stage taps.
Key Params: batch1 FP32, one multiply-accumulate=one MAC; excludes Haar, fusion setup,
interpolation, bias and activation arithmetic. State/Dependencies: existing exact ECB exporter.
```

### main (line 62)

```text
@luna-comment
Input: CLI canvas and report path. Output: measured JSON for baseline and four teachers.
Key Steps: explicit accepted bootstrap; profile actual fused graphs; assert target tolerance.
Key Params: square canvas256 default and 5% target tolerance, CPU thread1.
State/Dependencies: local repository only; creates report parent and JSON, no checkpoints.
```

### count (line 27)

```text
@luna-comment
Input: executed Conv2d hook. Output: append exact dense/grouped convolution MAC count.
Key Steps: multiply output elements by per-output kernel size.
Key Params: actual groups/kernel dimensions. State/Dependencies: local accumulation list.
```

### feature_size (line 35)

```text
@luna-comment
Input: stage output hook. Output: append output storage bytes (not peak training memory).
Key Steps: use tensor element count and size. Key Params: batch1 FP32.
State/Dependencies: local list; no retained tensors.
```

## tests/test_v3l_feature_distillation.py

### pair (line 23)

```text
@luna-comment
Input: teacher target label. Output: deterministic small-canvas teacher/student.
Key Steps: seed and construct original student and isolated teacher.
Key Params: canvas32, CPU. State/Dependencies: torch RNG reset for reproducible fixtures.
```

### test_width_only_topology_compute_and_nonzero_fusion (line 34)

```text
@luna-comment
Input: each candidate. Output: topology, measured MAC and nonzero fusion assertions.
Key Steps: compare module type graphs; activate residual; compare train/eval/fused outputs.
Key Params: 5% MAC tolerance, 2e-5 numerical tolerance. State/Dependencies: CPU torch/exporter.
```

### test_stage_geometry_loss_formula_and_hook_cleanup (line 64)

```text
@luna-comment
Input: true model taps and synthetic aligned targets. Output: loss arithmetic and cleanup proof.
Key Steps: capture eight stages; test exact zero, unit perturbation, detach and shape failures.
Key Params: per-tap mean, equal weights. State/Dependencies: temporary scoped hooks.
```

### test_joint_update_frozen_teacher_and_plain_student_export (line 99)

```text
@luna-comment
Input: real student/teacher and packed random pair. Output: frozen teacher and joint update proof.
Key Steps: one warmup step; inspect gradients and weight changes; strict-load student-only export.
Key Params: positive feature weight. State/Dependencies: Adam, unchanged V3M forward.
```

### test_disabled_feature_matches_original_optimizer_step (line 130)

```text
@luna-comment
Input: identical student weights and plain L1 objective. Output: exact disabled-KD equivalence.
Key Steps: compare original Adam step and engine step including gradients and all weights.
Key Params: feature_weight0, lr1e-4. State/Dependencies: CPU deterministic torch.
```

### test_exact_next_step_resume_with_scheduler (line 150)

```text
@luna-comment
Input: saved full engine state after an update. Output: identical resumed next-step weights.
Key Steps: disk round-trip teacher/student/adapters/Adam/scheduler/RNG; compare next update.
Key Params: StepLR plus nontrivial feature warmup. State/Dependencies: temporary checkpoint only.
```

### test_invalid_capacity_and_feature_settings (line 175)

```text
@luna-comment
Input: unsupported capacity, depth and weight settings. Output: explicit validation failures.
Key Steps: reject malformed configs and shared teacher/student parameters.
Key Params: width-only matrix is fixed. State/Dependencies: no optimizer steps.
```

## realesrgan/archs/ecbsr_Unet_arch_server.py

### Conv2D (line 15)

```text
@luna-comment
Input: convolution dimensions and optional separable/activation switches.
Output: ordered torch Sequential preserving the requested channel geometry.
Key Steps: build depthwise/pointwise or dense convolution; append optional LeakyReLU.
Key Params: kernel_size, stride, padding, is_seperable and has_relu.
State/Dependencies: torch module initialization; no external state.
```

## tools/export_basic_text_line_y4uv_v3_checkpoint_deploy_onnx.py

### _checkpoint_state (line 43)

```text
@luna-comment
Input: a checkpoint dictionary with params_ema, params, state_dict or raw tensor entries.
Output: a normalized generator mapping with known wrapper prefixes removed.
Key Steps: select the first recognized nested mapping, validate its type, normalize parameter names.
Key Params: params_ema takes precedence over params and state_dict.
State/Dependencies: mapping values are reused; no model or checkpoint file is modified.
```

### _load_checkpoint (line 70)

```text
@luna-comment
Input: target module and a trusted checkpoint file path.
Output: SHA256 of the loaded file; target module parameters are replaced in place.
Key Steps: load on CPU, normalize state, strict-load the module, stream file bytes into SHA256.
Key Params: strict loading rejects missing or unexpected parameter keys.
State/Dependencies: PyTorch deserialization and file reads; legacy trusted-checkpoint semantics preserved.
```

### _activation (line 91)

```text
@luna-comment
Input: source ECB or ECB2 module.
Output: Identity for linear mode, otherwise an independent activation copy.
Key Steps: inspect act_type and deepcopy the existing activation with learned parameters.
Key Params: linear mode has no activation; PReLU retains its trained slopes.
State/Dependencies: source activation state is read but not changed.
```

### _fused_kernel (line 104)

```text
@luna-comment
Input: source ECB or ECB2 module with learned branch parameters.
Output: detached cloned merged 3x3 kernel and bias on the source device.
Key Steps: use the direct kernel in train_repParam mode, otherwise merge branches under no_grad.
Key Params: train_repParam selects the existing direct-convolution semantics.
State/Dependencies: source rep_params and current weights; no source parameter mutation.
```

### fuse_reparameterized_modules (line 206)

```text
@luna-comment
Input: model tree containing ECB and ECB2 modules.
Output: counts of replaced ECB/ECB2 blocks; model children are replaced in place.
Key Steps: recurse through children, replacing decoder blocks first and accumulating replacement counts.
Key Params: isinstance dispatch preserves ECB2 versus ECB replacement types.
State/Dependencies: mutates the supplied model; callers copy the model first when preserving training state.
```

### _export_kwargs (line 229)

```text
@luna-comment
Input: integer ONNX opset version.
Output: static-axis ONNX export keyword dictionary.
Key Steps: set input/output names and constant folding; disable dynamo when supported.
Key Params: opset controls ONNX operator compatibility; no dynamic_axes are introduced.
State/Dependencies: torch.onnx.export signature inspection; no model or file mutation.
```

### FusedECB.__init__ (line 124)

```text
@luna-comment
Input: an existing encoder or same-scale ECB source.
Output: deployment Conv2d and independent original activation initialized in this module.
Key Steps: materialize kernel, construct stride-selected convolution, copy weights and activation.
Key Params: Encoder uses stride2; other supported phases use stride1 with padding1.
State/Dependencies: source channel/phase settings and branch parameters are read only.
```

### FusedECB.forward (line 148)

```text
@luna-comment
Input: BCHW tensor matching the source channel geometry.
Output: activated convolution result at the original phase-selected resolution.
Key Steps: apply the stored fused Conv2d then activation.
Key Params: stride and padding were fixed from the source at construction.
State/Dependencies: this module stores fused parameters; source branch modules are not called.
```

### FusedECB2.__init__ (line 162)

```text
@luna-comment
Input: an existing decoder ECB2 source.
Output: deployment upsampler, fused convolution, optional independent skip projection and activation.
Key Steps: reject unsupported direct mode, materialize kernel and copy projection/activation.
Key Params: upsample2 uses bilinear align_corners=False; with_proj preserves skip transformation.
State/Dependencies: source weights and settings are read; generated modules own independent parameters.
```

### FusedECB2.forward (line 190)

```text
@luna-comment
Input: decoder BCHW tensor and spatially compatible encoder skip tuple.
Output: activated decoder feature after skip addition.
Key Steps: optionally project skip, upsample decoder, convolve, add skip and activate.
Key Params: with_proj and the fixed2x bilinear upsampling preserve source semantics.
State/Dependencies: deployment modules and input tensors only; no branch-fusion work per call.
```
