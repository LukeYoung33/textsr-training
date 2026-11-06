"""Copy immutable active banks to local storage and atomically redirect owned links."""
from pathlib import Path
import fcntl
import hashlib
import html
import json
import os
import shutil
import time

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
LOCAL = Path('/data/guang.yang/TextSR/local_banks_20261103')
OUT = DATA / 'training_local_cache_20261103'
OLD = DATA / 'latest_vosr_four_500k_20261102/bank/sources'
NEW = DATA / 'v3m_original_dilate2_l1_500k_20261103/bank/sources'
PLAN = [
    ('lenovo', DATA / 'latest_vosr_four_500k_20261102/lenovo_bank', OLD / 'lenovo'),
    ('base', DATA / 'full_s22_training_matrix_20261028/bank256', OLD / 'base'),
    ('mojito', DATA / 'mojito280_guard_finetune_20261030/bank256', OLD / 'mojito'),
    ('dilate2', DATA / 'v3m_original_dilate2_l1_500k_20261103/train_bank', NEW / 'train'),
]
LIMIT = 120 * 1024 ** 2


def publish(state):
    """@luna-comment Input:copy state. Output:atomic JSON/Chinese HTML progress. Key Steps:serialize and replace owned files. Key Params:refresh15s. State/Dependencies:OUT only; no training mutation."""
    OUT.mkdir(parents=True, exist_ok=True)
    state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    for name, content in [('STATUS.json', json.dumps(state, indent=2)), ('index.html',
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15">'
        '<title>训练数据本地缓存</title><style>body{font:18px sans-serif;padding:30px;background:#edf2f7}pre{white-space:pre-wrap}</style>'
        '<h1>训练数据本地缓存</h1><p>逐文件 SHA256 校验；每份完成后自动切换。速率上限 120 MiB/s。</p>'
        '<pre>' + html.escape(json.dumps(state, ensure_ascii=False, indent=2)) + '</pre>')]:
        tmp = OUT / (name + '.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(OUT / name)


def main():
    """@luna-comment Input:fixed four immutable source banks. Output:verified local copies, per-file receipts and switched symlinks. Key Steps:space/lock gates, throttled streaming hash, independent local hash, per-bank atomic link. Key Params:8MiB blocks,120MiB/s network limit. State/Dependencies:source stat stable; trainers continue with identical bytes and existing mmap eviction."""
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / 'cache.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = []
    for label, source, link in PLAN:
        assert source.is_dir() and link.is_symlink()
        assert link.resolve() in (source.resolve(), (LOCAL / label).resolve())
        for path in sorted(source.rglob('*')):
            assert not path.is_symlink(), str(path)
            if path.is_file():
                files.append((label, source, path, path.stat().st_size))
    total = sum(row[3] for row in files)
    assert shutil.disk_usage(LOCAL).free > total + 20 * 1024 ** 3
    receipt_path = LOCAL / 'verified.jsonl'
    receipts = {}
    if receipt_path.exists():
        for line in receipt_path.read_text().splitlines():
            record = json.loads(line)
            receipts[record['key']] = record
    state = dict(phase='copying', pid=os.getpid(), destination=str(LOCAL), total_gib=round(total/1024**3, 2),
                 completed_gib=0, percent=0, active_bank=None, switched=[], copied_files=0)
    copied = 0
    network_bytes = 0
    start = time.monotonic()
    last_publish = 0
    with receipt_path.open('a', buffering=1) as receipt:
        for label, source, link in PLAN:
            state['active_bank'] = label
            for _, _, path, size in sorted((row for row in files if row[0] == label), key=lambda row: -row[3]):
                relative = path.relative_to(source)
                target = LOCAL / label / relative
                key = label + '/' + relative.as_posix()
                before = path.stat()
                prior = receipts.get(key)
                if prior and target.exists() and target.stat().st_size == size and prior['source_mtime_ns'] == before.st_mtime_ns:
                    pass
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + '.copying')
                    sha = hashlib.sha256()
                    with path.open('rb') as src, tmp.open('wb') as dst:
                        while True:
                            block = src.read(8 * 1024 ** 2)
                            if not block:
                                break
                            dst.write(block)
                            sha.update(block)
                            network_bytes += len(block)
                            delay = network_bytes / LIMIT - (time.monotonic() - start)
                            if delay > 0:
                                time.sleep(min(delay, 1))
                            if time.monotonic() - last_publish > 8:
                                elapsed = time.monotonic() - start
                                rate = network_bytes / max(elapsed, 1)
                                state.update(current_file=key, completed_gib=round(copied/1024**3, 2),
                                             percent=round(100*copied/total, 2), mib_per_second=round(rate/1024**2, 1),
                                             estimated_finish=time.strftime('%m-%d %H:%M', time.localtime(time.time()+(total-copied)/max(rate,1))) if network_bytes >= 10*1024**3 else '吞吐采样中')
                                publish(state)
                                last_publish = time.monotonic()
                        dst.flush()
                        os.fsync(dst.fileno())
                    after = path.stat()
                    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), key
                    check = hashlib.sha256()
                    with tmp.open('rb') as verify:
                        while True:
                            block = verify.read(8 * 1024 ** 2)
                            if not block:
                                break
                            check.update(block)
                    assert check.digest() == sha.digest() and tmp.stat().st_size == size, key
                    tmp.replace(target)
                    receipt.write(json.dumps(dict(key=key, sha256=sha.hexdigest(), bytes=size, source_mtime_ns=before.st_mtime_ns))+'\n')
                copied += size
                state['copied_files'] += 1
            assert link.resolve() in (source.resolve(), (LOCAL / label).resolve())
            temporary = link.with_name(link.name + '.local_cache_tmp')
            assert not temporary.exists() and not temporary.is_symlink()
            temporary.symlink_to(LOCAL / label)
            temporary.replace(link)
            state['switched'].append(dict(bank=label, link=str(link), local=str(LOCAL/label), at=time.strftime('%Y-%m-%d %H:%M:%S')))
            publish(state)
    state.update(phase='complete', percent=100, completed_gib=round(total/1024**3, 2), current_file=None)
    publish(state)
    (OUT/'COMPLETE.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        publish(dict(phase='failed', error=repr(error), pid=os.getpid()))
        raise
