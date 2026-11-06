"""Incremental BasicSR training loss history and standalone chart embedding."""
import json
import math
from pathlib import Path
import re

_CACHE = {}
_STEP = re.compile(r'\biter:\s*([\d,]+),')
_LOSS = re.compile(r'\b(l_[A-Za-z0-9_]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))\b', re.I)


def read_loss_history(path):
    """@luna-comment
    Input: append-only BasicSR log path. Output: ordered step/value dictionaries.
    Key Steps: cache byte position, retain incomplete line, parse finite l_* fields;
    replace duplicate steps and reset on inode change or truncation.
    Key Params: only explicit training iteration lines accepted; missing != zero.
    State/Dependencies: process-local cache keyed by absolute path; read-only logs.
    """
    path = Path(path)
    if not path.exists():
        return []
    key = str(path.resolve())
    with path.open('rb') as stream:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        cached = _CACHE.get(key)
        if not cached or cached['identity'] != identity or stat.st_size < cached['offset']:
            cached = dict(identity=identity, offset=0, pending=b'', points={})
            _CACHE[key] = cached
        stream.seek(cached['offset'])
        chunk = cached['pending'] + stream.read()
        cached['offset'] = stream.tell()
    lines = chunk.split(b'\n')
    cached['pending'] = lines.pop()
    for raw in lines:
        line = raw.decode('utf-8', errors='replace')
        step = _STEP.search(line)
        if step is None:
            continue
        values = {}
        for metric, token in _LOSS.findall(line):
            value = float(token)
            if math.isfinite(value):
                values[metric] = value
        if values:
            cached['points'][int(step.group(1).replace(',', ''))] = values
    return [dict(step=step, values=values) for step, values in sorted(cached['points'].items())]


def render_loss_charts(history):
    """@luna-comment
    Input: arm-to-loss-history mapping. Output: self-contained chart section/script.
    Key Steps: serialize numeric history safely and embed local canvas implementation.
    Key Params: raw metrics are logged weighted losses, optional EMA over log records.
    State/Dependencies: sibling offline_loss_charts.html asset; no CDN/server/fetch.
    """
    asset = Path(__file__).with_name('offline_loss_charts.html').read_text(encoding='utf-8')
    payload = json.dumps(history, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return asset.replace('@@LOSS_HISTORY@@', payload)
