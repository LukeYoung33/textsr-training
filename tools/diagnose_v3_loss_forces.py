"""Bounded exact-loss gradient and restored-Adam attribution on frozen bank patches."""
import copy
import json
import linecache
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE = DATA/'v3pre_v3m_three_20261027/pretrain'
OUT = DATA/'v3_loss_forces_20261028'


class Captured(Exception):
    """Control-flow sentinel; no production backward or step is executed."""


def capture(model, iteration):
    """@luna-comment Input:model/iteration. Output:live loss locals. Key Steps:stop before original backward. Key Params:exact optimizer source suffix. State/Dependencies:isolated sys trace restored in finally."""
    found = {}

    def trace(frame, event, arg):
        """@luna-comment Input:trace event. Output:trace callback or capture sentinel. Key Steps:match exact backward source line. Key Params:no production step. State/Dependencies:enclosing local dictionary."""
        if event == 'line' and frame.f_code.co_name == 'optimize_parameters' and frame.f_code.co_filename.endswith('basic_text_line_y4uv_training_model_v3.py'):
            if linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() == 'total.backward()':
                found.update(frame.f_locals)
                raise Captured()
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        model.optimize_parameters(iteration)
    except Captured:
        pass
    finally:
        sys.settrace(previous)
    assert 'total' in found, 'Did not intercept effective pre-backward branch'
    return found


def regions(model):
    """@luna-comment Input:fixed paired model batch. Output:input-defined core/ring/far masks. Key Steps:local darkness then morphology and valid interior. Key Params:45/255,15px envelope,3px ring,16px border. State/Dependencies:detached native Y."""
    import torch.nn.functional as F
    x = F.pixel_shuffle(model.lq[:, :4].detach(), 2)
    ink = (F.max_pool2d(x, 15, 1, 7)-x > 45/255).float()
    core = 1-F.max_pool2d(1-ink, 3, 1, 1)
    ring = F.max_pool2d(ink, 7, 1, 3)*(1-ink)
    far = 1-F.max_pool2d(ink, 15, 1, 7)
    valid = model.valid_mask_rgb.detach().clone()
    valid[:, :, :16] = 0; valid[:, :, -16:] = 0
    valid[:, :, :, :16] = 0; valid[:, :, :, -16:] = 0
    return {k: v*valid for k, v in [('core', core), ('ring', ring), ('far', far)]}


def measure(value, masks):
    """@luna-comment Input:full-resolution luma effect/fixed masks. Output:per-region mean and counts. Key Steps:masked average. Key Params:empty support explicit. State/Dependencies:detached tensors."""
    result = {}
    for name, mask in masks.items():
        count = float(mask.sum())
        result[name] = dict(count=count, mean=float((value*mask).sum()/mask.sum().clamp_min(1)))
    result['softening'] = result['core']['mean']-result['ring']['mean']
    return result


def save(value):
    """@luna-comment Input:partial/final receipt. Output:atomic JSON. Key Steps:owned temporary replacement. Key Params:isolated root. State/Dependencies:filesystem."""
    temporary = OUT/'RESULT.tmp'
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(OUT/'RESULT.json')


def anchors():
    """@luna-comment Input:frozen049-052 manifest. Output:four native grid tiles per image and fixed masks. Key Steps:load unchanged V3 input around selected text ROI centers. Key Params:256/224 original grid, no display rotation. State/Dependencies:read-only PNG/OpenCV/native packer."""
    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    base = DATA/'v3pre_three_mojito280_20261028'
    entries = {e['id']: e for e in json.loads((base/'selection.json').read_text())['samples']}
    ys = {'049': 1780, '050': 1750, '051': 1820, '052': 1900}
    images = []; provenance = []
    for identity, top in ys.items():
        entry = entries[identity]
        source = cv2.cvtColor(cv2.imread(entry['input'], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        # EXIF6: stored center=(display_y,stored_height-display_x).
        cx, cy = top+175, source.shape[0]-1450
        ox, oy = int((cx-128)//224)*224, int((cy-128)//224)*224
        for y in (oy, oy+224):
            for x in (ox, ox+224):
                image = source[y:y+256, x:x+256]
                assert image.shape == (256,256,3)
                images.append(image)
                provenance.append(dict(id=identity,stored_xy=[x,y],source_sha256=entry['input_sha256']))
    packed = rgb_to_y4uv(torch.from_numpy(np.stack(images).transpose(0,3,1,2).copy()).cuda().float()/255)
    mask = regions(SimpleNamespace(lq=packed, valid_mask_rgb=packed.new_ones((len(images),1,256,256))))
    return packed, mask, provenance


def anchor_measure(value, masks, x):
    """@luna-comment Input:anchor update/input and masks. Output:per-image signed force/edge gain delta. Key Steps:fixed input edge projection. Key Params:central differences,15/255 floor. State/Dependencies:detached CUDA tensors."""
    import torch.nn.functional as F
    result = {}
    for n, identity in enumerate(('049','050','051','052')):
        selected = slice(n*4, n*4+4)
        v = value[selected]; ref = x[selected]
        local = {k: m[selected] for k,m in masks.items()}
        stats = measure(v*255, local)
        gx = (ref[:,:,:,2:]-ref[:,:,:,:-2])*.5
        gy = (ref[:,:,2:,:]-ref[:,:,:-2,:])*.5
        dx = (v[:,:,:,2:]-v[:,:,:,:-2])*.5
        dy = (v[:,:,2:,:]-v[:,:,:-2,:])*.5
        gx = F.pad(gx,(1,1,0,0));gy = F.pad(gy,(0,0,1,1))
        dx = F.pad(dx,(1,1,0,0));dy = F.pad(dy,(0,0,1,1))
        energy = gx.square()+gy.square()
        valid = (energy > (15/255)**2).float()
        valid[:,:,:32]=0;valid[:,:,-32:]=0;valid[:,:,:,:32]=0;valid[:,:,:,-32:]=0
        stats['edge_gain_delta'] = float(((gx*dx+gy*dy)*valid).sum()/(energy*valid).sum().clamp_min(1e-12))
        result[identity] = stats
    return result


def target_probe(model, batches):
    """@luna-comment Input:real paired batches/exact SDF criterion. Output:GT-as-output gradient and implied target audit. Key Steps:retain detached GT anchors, differentiate only Y, compare distance-shell profiles. Key Params:configured temperatures/weight unchanged. State/Dependencies:no network updates; isolated artifacts."""
    import inspect
    import numpy as np
    import torch
    from tools.train_v3_edge_pilot import digest
    criterion = model.cri_sdf_boundary
    records = []
    for index, batch in enumerate(batches):
        model.feed_data(batch)
        gt = model.gt_rgb.detach()
        y = (.299*gt[:,:1]+.587*gt[:,1:2]+.114*gt[:,2:3]).requires_grad_(True)
        loss, details = criterion(y, gt, model.text_mask_rgb, model.valid_mask_rgb, return_details=True)
        weighted = loss*model.sdf_boundary_weight
        force = -torch.autograd.grad(weighted, y)[0]
        midpoint = details['midpoint'][:,None,None,None]
        polarity = details['polarity'][:,None,None,None]
        target = midpoint+polarity*criterion.ink_temperature/criterion.sdf_temperature*details['sdf']
        band = details['active_band']
        row = dict(batch=index,weighted_loss_at_exact_GT=float(weighted.detach()),force_at_exact_GT=measure(force*1e6, regions(model)),shells=[])
        for distance in (-3,-2,-1,1,2,3):
            mask = band*(details['sdf']==distance)*(polarity>0)
            count = float(mask.sum())
            row['shells'].append(dict(distance=distance,count=count,gt_y=float((y.detach()*mask).sum()/mask.sum().clamp_min(1))*255,target_y=float((target*mask).sum()/mask.sum().clamp_min(1))*255,desired_shift=float(((target-y.detach())*mask).sum()/mask.sum().clamp_min(1))*255,negative_gradient_x1e6=float((force*mask).sum()/mask.sum().clamp_min(1))*1e6))
        records.append(row)
        if index==0:
            np.savez_compressed(OUT/'SDF_PROFILE.npz',gt=y.detach().cpu().numpy(),target=target.cpu().numpy(),sdf=details['sdf'].cpu().numpy(),band=band.cpu().numpy(),force=force.cpu().numpy())
    source = Path(inspect.getfile(type(criterion)))
    mask = gt.new_zeros((1,1,256,256));mask[:,:,32:224,112:144]=1
    sharp = (.8-.7*mask).requires_grad_(True)
    synthetic_loss, synthetic_details = criterion(sharp,sharp.detach().repeat(1,3,1,1),mask,torch.ones_like(mask),return_details=True)
    synthetic_force = -torch.autograd.grad(synthetic_loss,sharp)[0]
    inside = synthetic_details['active_band']*(synthetic_details['sdf']<0)
    outside = synthetic_details['active_band']*(synthetic_details['sdf']>0)
    force_inside = float((synthetic_force*inside).sum()/inside.sum())
    force_outside = float((synthetic_force*outside).sum()/outside.sum())
    assert force_inside > 0 and force_outside < 0, 'Sharp aligned-stroke sign check failed'
    result = dict(complete=True,records=records,criterion_source=str(source),criterion_sha256=digest(source),temperature=criterion.ink_temperature,sdf_temperature=criterion.sdf_temperature,radius=criterion.radius,weight=model.sdf_boundary_weight,equation='Y_star = midpoint + polarity * ink_temperature / sdf_temperature * sdf',interpretation='Exact isolated BCE target, not the optimum of the full mixed objective; shell distances are native pixels.',synthetic_sharp_aligned_control=dict(ink_y=.1,paper_y=.8,midpoint=.45,negative_gradient_inside=force_inside,negative_gradient_outside=force_outside,passed=True))
    (OUT/'TARGET_PROBE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    """@luna-comment Input:frozen source config/bank/snapshots. Output:exact weighted gradients and reset-Adam interventions. Key Steps:fixed samples, historical stages, reconstruction audits, per-loss exclusion. Key Params:32 patches/4 snapshots. State/Dependencies:isolated CUDA process; no source writes."""
    global OUT
    with_anchors = '--with-anchors' in sys.argv
    sdf_only = '--sdf-only' in sys.argv
    if with_anchors:
        OUT = DATA/'v3_loss_forces_anchors_20261028'
    if sdf_only:
        OUT = DATA/'v3_sdf_target_20261028'
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    sys.path.insert(0, str(DATA/'v3pre_v3m_three_20261027/_build'))
    from tools.train_v3pre_area import bootstrap
    from tools.train_v3_edge_pilot import digest
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.models.v3pre_area_adapters import V3PreAreaDataset
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261028)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'RESULT.json').exists(), 'Refusing to overwrite evidence'
    cfg = yaml.safe_load((SOURCE/'training/configs/full.yml').read_text())
    original = SOURCE/'training/experiments/v3pre_pretrain_full_20261027'
    cfg['path']['pretrain_network_g'] = str(original/'models/net_g_250000.pth')
    cfg['path']['resume_state'] = None
    cfg['datasets']['train']['type'] = 'V3PreAreaDataset'
    config = OUT/'config.yml';config.write_text(yaml.safe_dump(cfg))
    sys.argv = [sys.argv[0], '-opt', str(config)]
    opt, _ = parse_options('/home/guang.yang/workspace/Real-ESRGAN', is_train=True)
    model = build_model(opt)
    anchor_x, anchor_masks, anchor_provenance = anchors() if with_anchors else (None,None,None)
    dataset = V3PreAreaDataset(opt['datasets']['train'])
    probe_path = DATA/'v3pre_three_mojito280_20261028/diagnostics/blur_halo_050051_20261028/TRAINING_PROBE.json'
    probes = json.loads(probe_path.read_text())['records']
    indices = []
    for row in probes:
        si = next(i for i, s in enumerate(dataset.shards) if s['sample_id'] == row['sample_id'])
        eligible = np.flatnonzero((dataset.index[:, 0] == si)&(dataset.index[:, 1] == row['patch_index']))
        indices.append(int(eligible[0]))
    indices += list(map(int, np.random.default_rng(20261028).integers(len(dataset), size=16)))
    batches = [dataset.collate_fn([dataset[i] for i in indices[b:b+4]]) for b in range(0, len(indices), 4)]
    if sdf_only:
        target_probe(model,batches)
        return
    parameters = tuple(p for p in model.net_g.parameters() if p.requires_grad)
    receipt = dict(complete=False, sampling='16 existing stratified purposive text probes plus16 seeded bank draws; descriptive only', indices=indices, line_ids=[b['line_id'] for b in batches], records=[], stages=[], source_sha256=digest(__file__),anchor_provenance=anchor_provenance)
    started = time.time()
    for iteration in (25000, 75000, 150000, 250000):
        checkpoint = original/f'models/net_g_{iteration}.pth'
        statepath = original/f'training_states/{iteration}.state'
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['params']
        adam = torch.load(statepath, map_location='cpu', weights_only=False)['optimizers'][0]
        receipt['stages'].append(dict(iteration=iteration, checkpoint_sha256=digest(checkpoint), optimizer_sha256=digest(statepath), lr=[g['lr'] for g in adam['param_groups']]))
        for batch_index, batch in enumerate(batches):
            model.net_g.load_state_dict(weights, strict=True)
            model.optimizer_g.load_state_dict(copy.deepcopy(adam))
            model.feed_data(batch)
            local = capture(model, iteration)
            terms = {'native': local['selected']}
            for name, raw, scale in [('CP','cp_raw','cp_weight'),('UV','raw_uv','uv_strong_scale'),('bgLF','raw_bg_lf','bg_lf_weight'),('missing','missing','missing_scale'),('false','false','false_scale'),('SDF','raw_sdf','sdf_boundary_scale')]:
                if raw in local and local[scale] > 0:
                    terms[name] = local[raw]*local[scale]
            reconstructed = sum(terms.values())
            assert torch.allclose(reconstructed, local['total'], atol=1e-7, rtol=1e-6), 'Uncaptured active objective'
            masks = regions(model)
            baseline = F.pixel_shuffle(model.output[:, :4].detach(), 2)
            if with_anchors:
                with torch.no_grad():
                    anchor_before = F.pixel_shuffle(model.net_g(anchor_x)[:,:4],2)
            term_grads = {}; directions = {}
            for name, loss in terms.items():
                gradient = torch.autograd.grad(loss, (model.output, *parameters), retain_graph=True, allow_unused=True)
                output_grad = gradient[0]
                if output_grad is None:
                    output_grad = torch.zeros_like(model.output)
                direction = -F.pixel_shuffle(output_grad[:, :4], 2)*1e6
                parameter_grad = [torch.zeros_like(p) if g is None else g.detach().clone() for p, g in zip(parameters, gradient[1:])]
                term_grads[name] = parameter_grad
                directions[name] = dict(weighted_loss=float(loss.detach()), negative_output_gradient_x1e6=measure(direction, masks), parameter_norm=float(sum(g.square().sum() for g in parameter_grad).sqrt()))
            total_grad = torch.autograd.grad(local['total'], parameters, allow_unused=True)
            summed = [sum(gs[i] for gs in term_grads.values()) for i in range(len(parameters))]
            error = float(sum((s-g).square().sum() for s, g in zip(summed, total_grad) if g is not None).sqrt())
            norm = float(sum(g.square().sum() for g in total_grad if g is not None).sqrt())
            assert error/max(norm, 1e-12) < 1e-4, f'Gradient reconstruction failed: error={error}, norm={norm}, terms={directions}'
            record = dict(iteration=iteration, batch=batch_index, terms=directions, reconstruction_relative_error=error/max(norm, 1e-12), interventions={})
            for excluded in ['TOTAL', 'MOMENTUM_ONLY', *terms]:
                model.net_g.load_state_dict(weights, strict=True)
                model.optimizer_g.load_state_dict(copy.deepcopy(adam))
                model.optimizer_g.zero_grad(set_to_none=True)
                for i, p in enumerate(parameters):
                    p.grad = torch.zeros_like(p) if excluded == 'MOMENTUM_ONLY' else summed[i].clone() if excluded == 'TOTAL' else summed[i]-term_grads[excluded][i]
                    assert torch.isfinite(p.grad).all()
                model.optimizer_g.step()
                with torch.no_grad():
                    updated = F.pixel_shuffle(model.net_g(model.lq)[:, :4], 2)
                record['interventions'][excluded] = measure((updated-baseline)*255, masks)
                if with_anchors:
                    with torch.no_grad():
                        anchor_after = F.pixel_shuffle(model.net_g(anchor_x)[:,:4],2)
                    record.setdefault('anchor_interventions',{})[excluded] = anchor_measure(anchor_after-anchor_before, anchor_masks, F.pixel_shuffle(anchor_x[:,:4],2))
            receipt['records'].append(record)
            receipt['elapsed_seconds'] = time.time()-started
            save(receipt)
            print(json.dumps(dict(iteration=iteration,batch=batch_index,elapsed=receipt['elapsed_seconds'],losses=list(terms))), flush=True)
            del local, terms, term_grads, total_grad, summed
    receipt['complete'] = True
    receipt['limitations'] = ['Local one-step causal attribution; not a replay of all historical training.', 'Small descriptive bank sample, not Mojito049-052 gradients or population estimate.', 'TOTAL-minus-excluded includes restored Adam nonlinearity and shared parameters; not additive attribution.', 'Gradient maps describe descent directions, not RGB residual colors.']
    save(receipt)


if __name__ == '__main__':
    main()
