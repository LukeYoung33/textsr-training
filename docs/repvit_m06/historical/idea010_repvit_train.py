"""IDEA-010独立训练、恢复监督及配对里程碑报告。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
BASE = DATA / 'vosr_gt_mask_width_r1ori_500k_20261107/plus1'
BUILD = DATA / 'v3pre_v3m_three_20261027/_build'
RUNTIME = DATA / 'titans_v3m_four_500k_20261110/runtime'
OUT = DATA / 'idea010_repvit_m06_20261111'
DEIT = DATA / 'idea008_deit_tiny_20261111'
B0 = DATA / 'idea009_evit_b0_20261111'
REGISTRY = DATA / 'registry/vosr_gt_width_repaired_20261109/plus1/train'
ARMS = ('l1', 'ori')
SEED = 20260828


def read(path):
    """@luna-comment Input:JSON路径；Output:对象；Key Steps:严格UTF8读取；Key Params:缺失返回空字典；State/Dependencies:只读文件。"""
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save(path, value):
    """@luna-comment Input:自有路径/JSON；Output:原子保存；Key Steps:进程唯一临时文件后替换；Key Params:UTF8；State/Dependencies:仅自有输出。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    """@luna-comment Input:文件；Output:SHA256；Key Steps:流式计算；Key Params:1MiB；State/Dependencies:只读。"""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def bootstrap():
    """@luna-comment Input:已核实运行包；Output:既有paired训练与私有架构注册；Key Steps:绑定旧bootstrap路径后显式注册；Key Params:不扫描新架构或下载；State/Dependencies:仅当前进程registry。"""
    sys.path.insert(0, str(BUILD))
    import torch
    from tools import train_v3_native_regions as native
    native.ROOT = RUNTIME
    from tools.train_filtered_three import bootstrap as original
    original()
    from basicsr.utils.registry import ARCH_REGISTRY
    from idea010_repvit_arch import RepViTM06NativeRestore
    ARCH_REGISTRY.register()(RepViTM06NativeRestore)
    from basicsr.utils.registry import MODEL_REGISTRY
    parent = MODEL_REGISTRY.get('FilteredPairedModel')

    @MODEL_REGISTRY.register()
    class RepViTPairedModel(parent):
        def save(self, epoch, current_iter):
            """@luna-comment Input:epoch/步数；Output:既有原子检查点；Key Steps:父类保存后检查本批停止标记；Key Params:只在检查点退出；State/Dependencies:自有停止文件。"""
            super().save(epoch, current_iter)
            if current_iter > 0 and (OUT / 'STOP_AT_CHECKPOINT').exists():
                raise SystemExit('IDEA-010 budget stop after checkpoint')
    torch.set_num_threads(2)
    return torch


def prepare():
    """@luna-comment Input:平台确认5284版本与旧银行；Output:冻结配置/初始化/48图验证清单；Key Steps:全来源血缘匹配/索引哈希/同初始化配置；Key Params:500K/seed20260828；State/Dependencies:源数据只读，自有目录新建。"""
    import numpy as np
    import yaml
    ready = read(BASE / 'DATA_READY.json')
    assert ready['passed'] and ready['counts']['train']['parents'] == 5284
    assert ready['counts']['val']['parents'] == 583 and ready['no_mojito_in_train_or_val']
    for split in ('train', 'val'):
        assert digest(BASE / 'bank' / (split + '.json')) == ready[split + '_sha256']
    source = BASE / 'source_view/manifests/train.jsonl'
    assert digest(source) == ready['source_manifest_sha256']
    registered = [json.loads(line) for line in (REGISTRY / 'SOURCE_MANIFEST.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    identity = lambda row: (row['dataset'], row['sample_id'])
    registry_map = {identity(row): row for row in registered}
    assert len(rows) == len(registry_map) == 5284
    for row in rows:
        assert row['output_sha256'] == registry_map[identity(row)]['output_sha256']
    torch = bootstrap()
    from idea010_repvit_arch import RepViTM06NativeRestore, profile
    torch.manual_seed(SEED)
    model = RepViTM06NativeRestore()
    OUT.mkdir(parents=True, exist_ok=True)
    initial = OUT / 'initial.pth'
    if initial.exists():
        state = torch.load(initial, map_location='cpu')['params']
        assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    else:
        torch.save({'params': model.state_dict()}, initial)
    save(OUT / 'PROFILE.json', profile(model))
    base = yaml.safe_load((BASE / 'configs/ori.yml').read_text())
    bindings = {}
    for index, arm in enumerate(ARMS):
        cfg = copy.deepcopy(base)
        cfg['name'] = f'idea010_repvit_m06_{arm}_500k_20261111'
        cfg['model_type'] = 'RepViTPairedModel'
        cfg['train']['model_type'] = 'RepViTPairedModel'
        cfg['network_g'] = dict(type='RepViTM06NativeRestore')
        cfg['path'].update(pretrain_network_g=str(initial), experiments_root=str(OUT / 'experiments'), resume_state=None)
        cfg['manual_seed'] = SEED
        cfg['s22_strategy']['audit_root'] = str(OUT / 'audits' / arm)
        cfg['filtered_control']['kind'] = arm
        cfg['train'].update(recall_aware_weight_missing=0. if arm == 'l1' else 1.,
                            recall_aware_weight_false=0. if arm == 'l1' else .25)
        if arm == 'l1':
            cfg['train']['recall_aware_opt'] = None
        cfg['dist_params']['port'] = 37380 + index
        path = OUT / 'configs' / (arm + '.yml')
        path.parent.mkdir(exist_ok=True)
        payload = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists():
            assert path.read_text() == payload, 'Frozen recipe changed'
        else:
            path.write_text(payload)
        smoke = copy.deepcopy(cfg)
        smoke['name'] += '_smoke'
        smoke['path']['experiments_root'] = str(OUT / 'smoke_experiments')
        smoke['s22_strategy']['audit_root'] = str(OUT / 'smoke_audits' / arm)
        smoke_path = OUT / 'configs' / (arm + '_smoke.yml')
        smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
        bindings[arm] = dict(config=str(path), config_sha256=digest(path), smoke=str(smoke_path),
                             name=cfg['name'], initial=str(initial), initial_sha256=digest(initial))
    save(OUT / 'CONFIG_BINDINGS.json', dict(runs=bindings, total_iter=500000,
         shared_initial_sha256=digest(initial), seed=SEED, data_ready=ready,
         platform_record_id='b40e1bc5-7fc9-5abf-b3f4-b1b2bee1fc3c',
         source_sha256=digest(source), registered_source_sha256=digest(REGISTRY / 'SOURCE_MANIFEST.jsonl'),
         source_lineage_matches=5284, code={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}))
    val = read(BASE / 'bank/val.json')
    indices = np.load(BASE / 'bank' / val['index'])
    candidates = {}
    for position, (shard_index, row_index) in enumerate(indices):
        shard = val['shards'][int(shard_index)]
        candidates.setdefault(shard['dataset'], {}).setdefault(int(shard_index), []).append(position)
    chosen = []
    queues = {key: sorted(group, key=lambda i: hashlib.sha256((str(SEED) + str(i)).encode()).hexdigest())
              for key, group in candidates.items()}
    while len(chosen) < 48:
        for dataset in sorted(queues):
            if queues[dataset] and len(chosen) < 48:
                shard_index = queues[dataset].pop(0)
                positions = candidates[dataset][shard_index]
                position = positions[len(positions) // 2]
                chosen.append(dict(index=position, dataset=dataset, shard=shard_index,
                                   row=int(indices[position, 1]), sample_id=val['shards'][shard_index]['sample_id']))
        if not any(queues.values()) and len(chosen) < 48:
            raise ValueError('Insufficient distinct validation parents')
    save(OUT / 'MINI48.json', dict(rows=chosen, selection='seeded source-balanced distinct validation parents',
         validation_target='historical GT, not regenerated +1', val_sha256=digest(BASE / 'bank/val.json')))
    assert read(OUT / 'MINI48.json') == read(DEIT / 'MINI48.json'), 'Fixed validation differs from DeiT'
    print(json.dumps(dict(prepared=True, shared_initial_sha256=digest(initial), mini=len(chosen))))


def worker(config, smoke_output=None):
    """@luna-comment Input:冻结配置/可选smoke回执；Output:实际CUDA训练或预检；Key Steps:注册后复用原trainer；Key Params:FP32/无TF32；State/Dependencies:仅本次任务模型及优化器。"""
    torch = bootstrap()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from tools import train_filtered_three as trainer
    if smoke_output:
        from idea010_repvit_smoke import smoke
        smoke(config, smoke_output)
    else:
        import logging
        logger = logging.getLogger('basicsr')
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        from basicsr.train import train_pipeline
        sys.argv = [sys.argv[0], '-opt', str(config)]
        train_pipeline(str(RUNTIME))


def alive(record):
    """@luna-comment Input:PID身份回执；Output:存活布尔；Key Steps:proc启动时间及非僵尸核查；Key Params:不发信号；State/Dependencies:Linux只读。"""
    try:
        fields = Path(f'/proc/{record["pid"]}/stat').read_text().split()
        return fields[2] != 'Z' and fields[21] == record['start_token']
    except (OSError, KeyError):
        return False


def spawn(name, arguments, gpu):
    """@luna-comment Input:自有子任务名/参数/GPU；Output:启动身份回执；Key Steps:复用仍活进程或独立会话启动；Key Params:两CPU线程；State/Dependencies:自有日志，不停止旧进程。"""
    path = OUT / 'children' / (name + '.json')
    previous = read(path)
    if alive(previous):
        return previous
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', REAL_ESRGAN_DEFER_FINAL_VALIDATION='1')
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    (OUT / 'logs').mkdir(exist_ok=True)
    with (OUT / 'logs' / (name + '.log')).open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *arguments],
            env=env, cwd=OUT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(pid=process.pid, start_token=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                  started=time.time(), gpu=gpu, arguments=arguments)
    save(path, record)
    return record


def checkpoint(binding, step):
    """@luna-comment Input:配置身份/步数；Output:已哈希验证的模型状态回执；Key Steps:核验原子配对文件；Key Params:精确步数；State/Dependencies:本任务只读检查点。"""
    path = OUT / 'experiments' / binding['name'] / 'training_states' / f'{step}.verified.json'
    result = read(path)
    if not result:
        return None
    assert result['iter'] == step
    assert digest(result['model']) == result['model_sha256']
    assert digest(result['state']) == result['state_sha256']
    return result


def deit_checkpoint(arm, step, smoke=False):
    """@luna-comment Input:DeiT组/同一步数/预检标记；Output:已有回执或None；Key Steps:只读精确检查点路径；Key Params:不替换为较晚模型；State/Dependencies:DeiT历史目录不写入。"""
    binding = read(DEIT / 'CONFIG_BINDINGS.json')['runs'][arm]
    directory = 'smoke_experiments' if smoke else 'experiments'
    name = binding['name'] + ('_smoke' if smoke else '')
    result = read(DEIT / directory / name / 'training_states' / f'{step}.verified.json')
    if not result:
        return None
    assert result['iter'] == step
    return result


def b0_checkpoint(arm, step, smoke=False):
    """@luna-comment Input:B0组/精确步数；Output:已验证回执或None；Key Steps:只读同预算模型/优化器SHA；Key Params:smoke独立；State/Dependencies:不修改B0目录。"""
    binding = read(B0 / 'CONFIG_BINDINGS.json')['runs'][arm]
    directory = 'smoke_experiments' if smoke else 'experiments'
    name = binding['name'] + ('_smoke' if smoke else '')
    result = read(B0 / directory / name / 'training_states' / f'{step}.verified.json')
    if result:
        assert result['iter'] == step
        assert digest(result['model']) == result['model_sha256']
        assert digest(result['state']) == result['state_sha256']
    return result


def launch_config(arm, binding):
    """@luna-comment Input:组别和冻结绑定；Output:严格恢复启动配置；Key Steps:降序验证本任务检查点，拒绝无检查点丢进度；Key Params:不回退旧训练；State/Dependencies:只写自有launch配置。"""
    import yaml
    assert digest(binding['config']) == binding['config_sha256']
    assert digest(binding['initial']) == binding['initial_sha256']
    cfg = yaml.safe_load(Path(binding['config']).read_text())
    directory = OUT / 'experiments' / binding['name'] / 'training_states'
    files = sorted(directory.glob('*.verified.json'), key=lambda p: int(p.name.split('.')[0]), reverse=True)
    if files:
        result = checkpoint(binding, int(files[0].name.split('.')[0]))
        cfg['path']['resume_state'] = result['state']
    elif read(OUT / 'audits' / arm / 'RUN_STATUS.json').get('iter', 0) > 3:
        raise RuntimeError('Progress without checkpoint; refusing silent scratch restart: ' + arm)
    path = OUT / 'configs' / (arm + '_launch.yml')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def publish():
    """@luna-comment Input:实时回执/历史；Output:监控JSON与HTML；Key Steps:真实步数/速率/存活分析；Key Params:不捏造ETA；State/Dependencies:只写本任务目录。"""
    import html
    runs = []
    for arm in ARMS:
        status = read(OUT / 'audits' / arm / 'RUN_STATUS.json')
        child = read(OUT / 'children' / ('train_' + arm + '.json'))
        running = alive(child)
        runs.append(dict(arm=arm, label='RepViT-M0.6 ' + ('L1-only' if arm == 'l1' else 'L1-R1_ori'),
                         iteration=status.get('iter', 0), total=500000, gpu=child.get('gpu'),
                         status='running' if running else 'not_running', lr=status.get('lr'), logs=status.get('logs', {})))
    save(OUT / 'monitor.json', dict(updated=time.time(), host='Remote57', runs=runs,
                                  error=read(OUT / 'ERROR.json'), profile=read(OUT / 'PROFILE.json')))
    body = '<meta charset="utf-8"><meta http-equiv="refresh" content="20"><h1>IDEA-010 RepViT-M0.6</h1>'
    body += '<p>Remote57 / +1px修复5284对 / 同初始化 / 500K / 两组</p>'
    for row in runs:
        body += '<pre>' + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre>'
    for path in sorted((OUT / 'reports').glob('*/index.html')):
        body += f'<p><a href="{path.relative_to(OUT).as_posix()}">{path.parent.name}效果报告</a></p>'
    if (OUT / 'ERROR.json').exists():
        body += '<pre>' + html.escape((OUT / 'ERROR.json').read_text()) + '</pre>'
    temp = OUT / 'index.tmp'
    temp.write_text(body, encoding='utf-8')
    temp.replace(OUT / 'index.html')


def evaluate(step, smoke=False):
    """@luna-comment Input:两组相同里程碑；Output:48图像/指标/交互报告；Key Steps:严格加载预测与历史参考/原生RGB指标；Key Params:有效区RGB MAD降序；State/Dependencies:固定mini、只读权重。"""
    import cv2
    import numpy as np
    import yaml
    torch = bootstrap()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from idea010_repvit_arch import RepViTM06NativeRestore
    from idea008_deit_arch import DeiTTinyNativeRestore
    from idea009_evit_arch import EfficientViTB0NativeRestore
    from basicsr.data import build_dataset
    from basicsr.archs import build_network
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    from idea010_repvit_report import write_report
    bindings = read(OUT / 'CONFIG_BINDINGS.json')['runs']
    cfg = yaml.safe_load(Path(bindings['ori']['config']).read_text())
    options = cfg['datasets']['val']
    options['phase'] = 'val'
    dataset = build_dataset(options)
    models, identities = {}, {}
    for arm in ARMS:
        if smoke:
            receipt = read(OUT / 'smoke_experiments' / (bindings[arm]['name'] + '_smoke') / 'training_states' / f'{step}.verified.json')
            assert digest(receipt['model']) == receipt['model_sha256']
        else:
            receipt = checkpoint(bindings[arm], step)
        assert receipt is not None
        model = RepViTM06NativeRestore().cuda().eval()
        model.load_state_dict(torch.load(receipt['model'], map_location='cpu')['params'], strict=True)
        models[arm], identities[arm] = model, receipt
    for arm in ARMS:
        receipt = deit_checkpoint(arm, step, smoke)
        assert receipt is not None
        assert digest(receipt['model']) == receipt['model_sha256']
        assert digest(receipt['state']) == receipt['state_sha256']
        model = DeiTTinyNativeRestore().cuda().eval()
        model.load_state_dict(torch.load(receipt['model'], map_location='cpu')['params'], strict=True)
        models['deit_' + arm], identities['deit_' + arm] = model, receipt
    for arm in ARMS:
        receipt = b0_checkpoint(arm, step, smoke)
        assert receipt is not None
        model = EfficientViTB0NativeRestore().cuda().eval()
        model.load_state_dict(torch.load(receipt['model'], map_location='cpu')['params'], strict=True)
        models['b0_' + arm], identities['b0_' + arm] = model, receipt
    old = read(BASE / 'DONE.json')
    old_config = yaml.safe_load((BASE / 'configs/ori.yml').read_text())
    reference_path = BASE / 'experiments' / old_config['name'] / 'models/net_g_500000.pth'
    old_receipt = read(BASE / 'experiments' / old_config['name'] / 'training_states/500000.verified.json')
    assert digest(reference_path) == old_receipt['model_sha256']
    reference = build_network(old_config['network_g']).cuda().eval()
    reference.load_state_dict(torch.load(reference_path, map_location='cpu')['params'], strict=True)
    models['v3m'] = reference
    identities['v3m'] = dict(checkpoint=str(reference_path), sha256=digest(reference_path), step=500000)
    report = OUT / 'report_preflight' if smoke else OUT / 'reports' / f'{step:06d}'
    images = report / 'images'
    images.mkdir(parents=True, exist_ok=True)
    results = []
    for number, selected in enumerate(read(OUT / 'MINI48.json')['rows']):
        batch = dataset.collate_fn([dataset[selected['index']]])
        native = rgb_to_y4uv(batch['lq_rgb'].cuda())
        valid = batch['valid_mask_rgb'][0, 0].numpy() > .5
        gt = batch['gt_rgb'][0].permute(1, 2, 0).numpy()
        decoded = dict(lq=batch['lq_rgb'][0].permute(1, 2, 0).numpy(), gt=gt)
        with torch.no_grad():
            for name, model in models.items():
                pred = model(native)
                rgb = YUV2RGB(y4uv_to_yuv444(pred), format='RGB', is_train=True, uv_normalized=True)
                decoded[name] = rgb[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        paths, metrics = {}, {}
        for name, array in decoded.items():
            filename = f'{number:03d}_{name}.png'
            pixels = np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)
            assert cv2.imwrite(str(images / filename), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            paths[name] = 'images/' + filename
            if name in models:
                error = decoded[name][valid].astype(np.float64) - gt[valid]
                mse = float(np.mean(error ** 2))
                metrics[name + '_psnr'] = float(-10 * np.log10(max(mse, 1e-15)))
        diff = np.abs(decoded['ori'][valid] - decoded['deit_ori'][valid]) * 255
        metrics.update(mad=float(diff.mean()), max_abs=float(diff.max()), changed_fraction=float((diff.max(axis=1) > 1).mean()))
        metrics['l1_architecture_mad'] = float(np.abs(decoded['l1'][valid] - decoded['deit_l1'][valid]).mean() * 255)
        metrics['repvit_loss_ablation_mad'] = float(np.abs(decoded['ori'][valid] - decoded['l1'][valid]).mean() * 255)
        metrics['b0_architecture_mad'] = float(np.abs(decoded['ori'][valid] - decoded['b0_ori'][valid]).mean() * 255)
        difference = np.clip(np.abs(decoded['ori'] - decoded['deit_ori']) * 8 * 255, 0, 255).astype(np.uint8)
        filename = f'{number:03d}_diff8.png'
        assert cv2.imwrite(str(images / filename), cv2.cvtColor(difference, cv2.COLOR_RGB2BGR))
        paths['diff8'] = 'images/' + filename
        results.append(dict(id=number, source=selected, images=paths, **metrics))
    results.sort(key=lambda row: row['mad'], reverse=True)
    summary = dict(step=step, preflight_only=smoke, rows=results, checkpoints=identities,
        selection=read(OUT / 'MINI48.json'), valid_region='valid_mask_rgb > .5',
        metric='decoded float RGB [0,1], prediction clipped, masked PSNR; MAD in 0..255',
        image_difference_threshold=1, difference_gain=8, primary_comparison='RepViT R1 versus same-step DeiT R1; B0 R1 also shown',
        historical_reference='V3M R1 500K; intermediate EfficientViT checkpoints have smaller training budget',
        validation_target='historical583 GT; not regenerated +1; patch-only not full-frame', user_review='pending')
    save(report / 'metrics.json', summary)
    write_report(report, summary)
    save(report / 'COMPLETE.json', dict(passed=True, step=step, scenes=48, images=480,
         metrics_sha256=digest(report / 'metrics.json'), html_sha256=digest(report / 'index.html')))


def controller():
    """@luna-comment Input:已准备双组批次；Output:最多48h训练/恢复/配对报告；Key Steps:独占锁/哈希smoke门禁/实际空闲卡/检查点监督；Key Params:500K/最多2卡/3次重试；State/Dependencies:仅本任务子进程及输出。"""
    import fcntl
    lock = (OUT / 'controller.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    bindings = read(OUT / 'CONFIG_BINDINGS.json')['runs']
    started = read(OUT / 'CONTROLLER.json').get('batch_started', time.time())
    save(OUT / 'CONTROLLER.json', dict(pid=os.getpid(), batch_started=started, host='Remote57'))
    failures = dict.fromkeys(ARMS, 0)
    while True:
        try:
            used = {read(OUT / 'children' / ('train_' + arm + '.json')).get('gpu') for arm in ARMS
                    if alive(read(OUT / 'children' / ('train_' + arm + '.json')))}
            if time.time() - started > 48 * 3600:
                save(OUT / 'ERROR.json', dict(error='48h budget exceeded; workers instructed to stop at next checkpoint', time=time.time()))
                (OUT / 'STOP_AT_CHECKPOINT').touch()
            for arm in ARMS:
                binding = bindings[arm]
                if checkpoint(binding, 500000):
                    continue
                if alive(read(OUT / 'children' / ('train_' + arm + '.json'))):
                    continue
                if (OUT / 'STOP_AT_CHECKPOINT').exists():
                    continue
                previous = read(OUT / 'children' / ('train_' + arm + '.json'))
                if previous:
                    failures[arm] += 1
                    if failures[arm] > 3:
                        raise RuntimeError('Three retries exhausted: ' + arm)
                inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                cards = [tuple(map(int, line.split(','))) for line in inventory.splitlines()]
                available = [row for row in cards if row[0] in (0, 1, 2, 3) and row[0] not in used and row[1] > 6144 and row[2] < 70]
                if not available:
                    continue
                gpu = min(available, key=lambda row: (row[2], -row[1]))[0]
                smoke_path = OUT / 'smoke' / (arm + '.json')
                if not read(smoke_path).get('passed'):
                    child = spawn('smoke_' + arm, ['worker', '--config', binding['smoke'], '--smoke-output', str(smoke_path)], gpu)
                    while alive(child):
                        publish()
                        time.sleep(10)
                    if not read(smoke_path).get('passed'):
                        raise RuntimeError('CUDA smoke failed; inspect logs: ' + arm)
                launch = launch_config(arm, binding)
                spawn('train_' + arm, ['worker', '--config', str(launch)], gpu)
                used.add(gpu)
            for step in range(50000, 500001, 50000):
                report = OUT / 'reports' / f'{step:06d}'
                if (report / 'COMPLETE.json').exists():
                    continue
                if all(checkpoint(bindings[arm], step) for arm in ARMS) and all(deit_checkpoint(arm, step) for arm in ARMS) and all(b0_checkpoint(arm, step) for arm in ARMS):
                    active = read(OUT / 'children/evaluation.json')
                    if not alive(active):
                        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                            '--format=csv,noheader,nounits'], text=True)
                        own_cards = {read(OUT / 'children' / ('train_' + arm + '.json'))['gpu'] for arm in ARMS}
                        cards = [tuple(map(int, line.split(','))) for line in inventory.splitlines()]
                        gpu = max((row for row in cards if row[0] in own_cards), key=lambda row: row[1])[0]
                        attempt = read(OUT / 'evaluation_attempts.json')
                        attempt[str(step)] = attempt.get(str(step), 0) + 1
                        save(OUT / 'evaluation_attempts.json', attempt)
                        if attempt[str(step)] > 3:
                            raise RuntimeError('Evaluation retries exhausted: ' + str(step))
                        spawn('evaluation', ['evaluate', '--step', str(step)], gpu)
                    break
            publish()
            if all(checkpoint(bindings[arm], 500000) for arm in ARMS) and (OUT / 'reports/500000/COMPLETE.json').exists():
                save(OUT / 'COMPLETE.json', dict(passed=True, arms=2, iterations=500000, report='reports/500000/index.html'))
                return
            if (OUT / 'STOP_AT_CHECKPOINT').exists() and not any(alive(read(OUT / 'children' / ('train_' + arm + '.json'))) for arm in ARMS):
                return
            time.sleep(20)
        except Exception as error:
            save(OUT / 'ERROR.json', dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
            publish()
            return


def main():
    """@luna-comment Input:prepare/controller/worker/evaluate命令；Output:对应隔离操作；Key Steps:显式分派；Key Params:不隐式开训；State/Dependencies:用户指定Remote57批次。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'controller', 'worker', 'evaluate'])
    parser.add_argument('--config')
    parser.add_argument('--smoke-output')
    parser.add_argument('--step', type=int)
    parser.add_argument('--smoke-report', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'worker':
        worker(args.config, args.smoke_output)
    elif args.action == 'evaluate':
        evaluate(args.step, smoke=args.smoke_report)
    else:
        controller()


if __name__ == '__main__':
    main()
