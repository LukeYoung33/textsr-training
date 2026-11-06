"""Persistent, image-only issue annotation manifests shared with the comparison viewer."""
import hashlib
import json
import os
import uuid
from pathlib import Path

from PIL import Image, ImageOps

SCHEMA = 'lenovo_manual_issues_v1'
DEFAULT_CATEGORIES = {'C1': '模糊', 'C2': '细笔画/端点损失', 'C3': '彩色色斑',
                      'C4': '色度扩散', 'C5': '黑白边/振铃', 'C6': '毛刺/边缘变形',
                      'C7': '重影', 'C8': '二次屏摄'}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def resolve_manifest_path(value):
    """Accept an existing directory, a new directory name, or an explicit JSON filename."""
    path = Path(value).expanduser().resolve()
    if path.is_dir() or (not path.exists() and not path.suffix):
        path = path / 'manifest.json'
    return path


def write_manifest(path, data):
    """Replace a complete UTF-8 document atomically, preserving the previous file on failure."""
    path = resolve_manifest_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scan_directory(root, destination, source_directory=None):
    """Recursively inspect images and merge saved labels using their relative paths."""
    root, destination = Path(root).resolve(), resolve_manifest_path(destination)
    old = json.loads(destination.read_text(encoding='utf-8-sig')) if destination.is_file() else {}
    if old and (old.get('schema_version') != SCHEMA or old.get('image_root') != str(root)):
        raise ValueError('已有 manifest 属于另一目录或格式，请选择新的保存路径')
    previous = {r['id']: r for r in old.get('records', [])}
    records, errors = [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                width, height = image.size
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            fingerprint = [stat.st_size, stat.st_mtime_ns]
            saved = previous.get(relative, {})
            unchanged = saved.get('fingerprint') == fingerprint
            records.append({'id': relative, 'image': relative, 'width': width, 'height': height,
                            'fingerprint': fingerprint,
                            'problem_categories': saved.get('problem_categories', []),
                            'reviewed': bool(saved.get('reviewed', False) and unchanged),
                            'note': saved.get('note', '')})
        except Exception as error:
            errors.append({'image': str(path), 'error': str(error)})
    if not records:
        raise ValueError('目录下没有可解码的图像' + (f'；失败 {len(errors)} 张' if errors else ''))
    data = {'schema_version': SCHEMA, 'image_root': str(root),
            'source_directory': source_directory or str(root),
            'category_definitions': old.get('category_definitions', dict(DEFAULT_CATEGORIES)),
            'records': records, 'scan_errors': errors}
    write_manifest(destination, data)
    return data


def annotation_path(root, base):
    return Path(base) / hashlib.sha256(str(root).encode()).hexdigest()[:16] / 'manifest.json'


def update_annotation(data, request):
    """Validate edits before saving; empty reviewed labels explicitly mean no listed issue."""
    action = request['action']
    if action == 'category':
        name = str(request.get('name', '')).strip()
        if not name or len(name) > 100:
            raise ValueError('问题名称需为 1–100 个字符')
        definitions = data['category_definitions']
        category_id = request.get('category_id')
        if category_id and category_id not in definitions:
            raise ValueError('类别不存在')
        if name in definitions.values() and definitions.get(category_id) != name:
            raise ValueError('问题名称已存在')
        if not category_id:
            category_id = 'CUSTOM_' + uuid.uuid4().hex[:12]
        definitions[category_id] = name
    elif action == 'delete_category':
        category_id = str(request.get('category_id', ''))
        if not category_id.startswith('CUSTOM_') or category_id not in data['category_definitions']:
            raise ValueError('只能删除已有的自定义问题')
        if any(category_id in record.get('problem_categories', []) for record in data['records']):
            raise ValueError('该问题已被图像使用，不能删除')
        del data['category_definitions'][category_id]
    elif action == 'record':
        record = next((r for r in data['records'] if r['id'] == request.get('id')), None)
        if record is None:
            raise ValueError('图像不存在')
        categories = request.get('categories', [])
        if not isinstance(categories, list) or any(c not in data['category_definitions'] for c in categories):
            raise ValueError('未知问题类别')
        note = str(request.get('note', ''))
        if len(note) > 10000:
            raise ValueError('备注过长')
        record.update(problem_categories=list(dict.fromkeys(categories)),
                      reviewed=bool(request.get('reviewed', True)), note=note)
    else:
        raise ValueError('不支持的编辑操作')


def manual_catalog(manifest_path, manifest, roots):
    """Pair manual images and outputs by full relative filename, preserving duplicate basenames."""
    rows, assets = [], {}
    image_root = Path(manifest['image_root'])
    if not image_root.is_absolute():
        image_root = manifest_path.parent / image_root

    def register(path):
        if path is None or not path.is_file():
            return None
        token = str(len(assets))
        assets[token] = (path, None, 1)
        return '/image/' + token

    for record in manifest['records']:
        relative = Path(record['image'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('图像路径必须位于 image_root 内')
        outputs = [root / relative if root else None for root in roots]
        rows.append({'id': record['id'], 'source_lq': str(image_root / relative), 'source_gt': None,
                     'categories': record['problem_categories'], 'reviewed': record['reviewed'],
                     'note': record.get('note', ''), 'seed': '', 'geometry': None, 'stages': [],
                     'output_references': {label: {'file': str(path), 'stack_row': None, 'stack_count': 1}
                                           if path and path.is_file() else None
                                           for label, path in zip(('A', 'B'), outputs)},
                     'images': {'GT': None, 'LQ': register(image_root / relative),
                                'A': register(outputs[0]), 'B': register(outputs[1])}})
    return rows, assets
