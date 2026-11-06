"""Local validation browser. Run with Python, then open the printed localhost URL."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import threading
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from PIL import Image, ImageOps

try:
    from tools.lenovo_annotations import (SCHEMA, DEFAULT_CATEGORIES, annotation_path,
        scan_directory, write_manifest, update_annotation, manual_catalog, resolve_manifest_path)
except ModuleNotFoundError:
    from lenovo_annotations import (SCHEMA, DEFAULT_CATEGORIES, annotation_path,
        scan_directory, write_manifest, update_annotation, manual_catalog, resolve_manifest_path)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / 'datasets/generated_images/lenovo_validation_aspect_2to5_stronger_c5_priority_128_20260911/manifest.json'
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


class ResultArmChoice(ValueError):
    """A parent evaluation directory needs an explicit arm selection."""

    def __init__(self, side, root, choices):
        super().__init__(f'{side} 包含多个结果 arm，请选择其中一个')
        self.side, self.root, self.choices = side, root, choices


def path_key(value):
    """Use GT basename plus variant directory, never an output enumeration index."""
    path = PurePosixPath(str(value).replace('\\', '/'))
    return path.parent.name + '/' + path.name


def resolve_directory(value):
    """Fetch explicitly requested Remote58 directories into distinct local caches."""
    value = value.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if not value.startswith('/AI_DNS/') and not value.startswith('/home/'):
        raise ValueError('目录不存在：' + value)
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    cache = REPO / 'datasets/viewer_cache' / digest
    destination = cache / PurePosixPath(value.rstrip('/')).name
    # Re-fetch on each explicit load; a previous interrupted transfer is not a valid cache.
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(REPO / 'tools/remote58_fetch.ps1'), '-RemotePath', value.rstrip('/'),
                    '-LocalDirectory', str(cache)], check=True, capture_output=True, text=True, timeout=900)
    if not destination.is_dir():
        raise ValueError('同步后未找到目录：' + str(destination))
    return destination


def select_result_root(root, side, selected=''):
    """Resolve a single evaluator visual root without merging experiments."""
    if root is None:
        return None
    if (root / 'visuals').is_dir():
        return root / 'visuals'
    choices = sorted((child.name, child / 'visuals') for child in root.iterdir()
                     if child.is_dir() and (child / 'visuals').is_dir())
    if not choices:
        return root
    selected = selected.strip()
    for name, visual_root in choices:
        if selected == name:
            return visual_root
    if len(choices) == 1:
        return choices[0][1]
    raise ResultArmChoice(side, root, [name for name, _ in choices])


def index_results(root):
    """Index grouped evaluation outputs by their recorded source identity."""
    matched = {}
    if root is None:
        return matched
    if (root / 'visuals').is_dir():
        root = root / 'visuals'
    for meta in sorted(root.rglob('metadata.json')):
        data = json.loads(meta.read_text(encoding='utf-8-sig'))
        order = data.get('variant_order', [])
        if not order:
            continue
        for row, item in enumerate(order):
            key = (str(data['gt_path']).replace('\\', '/'), str(item['lq_path']).replace('\\', '/'))
            if key in matched:
                raise ValueError('重复的结果身份：' + str(key))
            matched[key] = {name: (meta.parent / file, row if name != 'gt' else None, len(order))
                            for name, file in [('gt', 'gt.png'), ('lq', 'lq_vertical.png'), ('out', 'output_vertical.png')]}
    if not matched:
        # Flat outputs must retain variant_N/basename; ambiguous basenames are not guessed.
        for file in root.rglob('*'):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                key = path_key(file)
                if key in matched:
                    raise ValueError('结果文件名称不唯一：' + key)
                matched[key] = {'out': (file, None, 1)}
    return matched


def build_catalog(manifest_path, root_a=None, root_b=None):
    """Build actual-stage labels and immutable image references, with explicit missing matches."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version') == SCHEMA:
        return manual_catalog(manifest_path, manifest, (root_a, root_b))
    indexes = [index_results(root_a), index_results(root_b)]
    assets, rows = {}, []

    def register(ref):
        if ref is None or not ref[0].is_file():
            return None
        token = str(len(assets))
        assets[token] = ref
        return '/image/' + token

    def local_source(value, subtree):
        original = Path(value)
        if original.is_file():
            return (original, None, 1)
        parts = PurePosixPath(value.replace('\\', '/')).parts
        if subtree in parts:
            relative = parts[parts.index(subtree):]
            return (manifest_path.parent.joinpath(*relative), None, 1)
        return None

    for record in manifest['records']:
        gt_name = PurePosixPath(record['gt']).name
        for variant in record['variants']:
            key = (str(record['gt']).replace('\\', '/'), str(variant['lq']).replace('\\', '/'))
            results = [index.get(key) or index.get(path_key(variant['lq'])) for index in indexes]
            stages = variant.get('stages', {})
            stages = stages.get('stages', []) if isinstance(stages, dict) else stages
            categories = list(dict.fromkeys(s['branch'].upper() for s in stages if s.get('branch', 'always') != 'always'))
            if not categories:
                categories = variant.get('problem_categories', [])
            fallback = next((r for r in results if r and 'gt' in r), {})
            gt = register(local_source(record['gt'], 'gt')) or register(fallback.get('gt'))
            lq = register(local_source(variant['lq'], 'lq')) or register(fallback.get('lq'))
            rows.append({'id': gt_name.removesuffix('.png') + '/v' + str(variant['variant']),
                         'source_gt': record['gt'], 'source_lq': variant['lq'],
                         'variant': variant['variant'],
                         'output_references': {
                             label: {'file': str(result['out'][0]), 'stack_row': result['out'][1],
                                     'stack_count': result['out'][2]} if result else None
                             for label, result in zip(('A', 'B'), results)},
                         'categories': categories, 'seed': str(variant.get('seed', '')),
                         'geometry': record.get('geometry'), 'stages': stages,
                         'images': {'GT': gt, 'LQ': lq,
                                    'A': register(results[0].get('out')) if results[0] else None,
                                    'B': register(results[1].get('out')) if results[1] else None}})
    return rows, assets


class ViewerServer(ThreadingHTTPServer):
    def __init__(self, address, manifest):
        super().__init__(address, Handler)
        self.manifest = manifest
        self.rows, self.assets = build_catalog(manifest)
        self.lock = threading.Lock()
        self.status = {'busy': False, 'error': '', 'revision': 0}
        self.annotation = None
        self.annotation_manifest = None
        self.annotation_lock = threading.Lock()

    def annotate(self, request):
        """Serialize session edits and publish a revision to reject stale browser writes."""
        with self.annotation_lock:
            action = request.get('action')
            if action == 'open':
                if request.get('existing'):
                    destination = resolve_manifest_path(request['existing'].strip())
                    data = json.loads(destination.read_text(encoding='utf-8-sig'))
                    if data.get('schema_version') != SCHEMA:
                        raise ValueError('这不是人工问题标注 manifest')
                else:
                    root = resolve_directory(request.get('directory', ''))
                    if root is None:
                        raise ValueError('请输入图像目录')
                    destination = (resolve_manifest_path(request['destination'].strip()) if request.get('destination', '').strip()
                                   else annotation_path(root, REPO / 'datasets/issue_annotations'))
                    data = scan_directory(root, destination, request.get('directory'))
                self.annotation, self.annotation_manifest = data, destination
                self.annotation_revision = getattr(self, 'annotation_revision', 0) + 1
            elif self.annotation is None:
                raise ValueError('请先打开标注目录或 manifest')
            elif action != 'state':
                if request.get('revision') != self.annotation_revision:
                    raise ValueError('标注已在其他页面更新，请重新打开 manifest 后再编辑')
                data = copy.deepcopy(self.annotation)
                update_annotation(data, request)
                write_manifest(self.annotation_manifest, data)
                self.annotation = data
                self.annotation_revision += 1
            return {'data': self.annotation, 'manifest': str(self.annotation_manifest),
                    'revision': self.annotation_revision}

    def load(self, request):
        try:
            manifest = Path(request.get('manifest') or self.manifest)
            a = select_result_root(resolve_directory(request.get('a', '')), '结果 A', request.get('a_arm', ''))
            b = select_result_root(resolve_directory(request.get('b', '')), '结果 B', request.get('b_arm', ''))
            rows, assets = build_catalog(manifest, a, b)
            with self.lock:
                self.rows, self.assets, self.manifest = rows, assets, manifest
                self.status['revision'] += 1
                self.status.pop('choices', None)
        except ResultArmChoice as error:
            self.status['error'] = str(error)
            self.status['choices'] = {'side': error.side, 'root': str(error.root), 'arms': error.choices}
        except Exception as error:
            self.status['error'] = str(error)
            self.status.pop('choices', None)
        finally:
            self.status['busy'] = False


class Handler(BaseHTTPRequestHandler):
    def reply(self, payload, content_type='application/json; charset=utf-8', status=200):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/':
                return self.reply(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/annotate':
                return self.reply(Path(__file__).with_name('lenovo_annotation.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/status':
                return self.reply(self.server.status)
            if path == '/api/catalog':
                with self.server.lock:
                    data = json.loads(self.server.manifest.read_text(encoding='utf-8-sig'))
                    definitions = dict(DEFAULT_CATEGORIES)
                    definitions.update(data.get('category_definitions', {}))
                    for record in self.server.rows:
                        for category in record['categories']:
                            definitions.setdefault(category, category)
                    return self.reply({'rows': self.server.rows, 'manifest': str(self.server.manifest),
                                       'category_definitions': definitions})
            if path.startswith('/annotation-image/'):
                with self.server.annotation_lock:
                    record = self.server.annotation['records'][int(path.split('/')[-1])]
                    root = Path(self.server.annotation['image_root'])
                    if not root.is_absolute():
                        root = self.server.annotation_manifest.parent / root
                    file = root / record['image']
                    if not file.resolve().is_relative_to(root.resolve()):
                        raise ValueError('图像路径越界')
                with Image.open(file) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                return self.reply(buffer.getvalue(), 'image/png')
            if path.startswith('/image/'):
                with self.server.lock:
                    file, row, count = self.server.assets[path.split('/')[-1]]
                with Image.open(file) as source:
                    source = ImageOps.exif_transpose(source).convert('RGB')
                    if row is not None:
                        if source.height % count:
                            raise ValueError('纵向拼图高度不能被版本数整除')
                        height = source.height // count
                        source = source.crop((0, row * height, source.width, (row + 1) * height))
                    buffer = io.BytesIO()
                    source.save(buffer, format='PNG')
                return self.reply(buffer.getvalue(), 'image/png')
            self.reply({'error': 'Not found'}, status=404)
        except Exception as error:
            self.reply({'error': str(error)}, status=400)

    def do_POST(self):
        if self.path not in ('/api/load', '/api/annotation'):
            return self.reply({'error': 'Not found'}, status=404)
        # Local browser only: reject cross-origin state-changing requests.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.reply({'error': 'Origin rejected'}, status=403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length < 32768:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(length))
            if self.path == '/api/annotation':
                return self.reply(self.server.annotate(request))
            with self.server.lock:
                if self.server.status['busy']:
                    return self.reply({'error': '正在加载，请稍候'}, status=409)
                self.server.status.update(busy=True, error='')
            threading.Thread(target=self.server.load, args=(request,), daemon=True).start()
            self.reply({'started': True})
        except Exception as error:
            self.reply({'error': str(error)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ViewerServer(('127.0.0.1', args.port), args.manifest)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
