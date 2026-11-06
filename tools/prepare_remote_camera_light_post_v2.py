"""Stage a byte-bound, worker-only Remote58 bundle in the approved Y output directory."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('Y:/guang.yang/TextSR/data/camera_post_v2_20261023/bundle')
REMOTE = '/AI_DNS/guang.yang/TextSR/data/camera_post_v2_20261023'
FILES = ('tools/light_text_postprocess.py', 'tools/light_text_postprocess_v2.py', 'tools/doc_warp_composition.py')


def digest(path):
    """@luna-comment Input: source or staged file. Output: SHA256 hex. Key Steps: read bounded source bytes. Key Params: files are small Python/JSON assets. State/Dependencies: no private paths."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage():
    """@luna-comment Input: frozen repository source/280 manifest. Output: isolated remote worker bundle and exact source hash manifest. Key Steps: copy core unchanged, derive Y-to-AI_DNS worker adapter, assert no viewer import, write two output-root variants. Key Params: pilot and full directories under approved Y root. State/Dependencies: no remote execution/download; Y writes only."""
    (DEST/'tools').mkdir(parents=True, exist_ok=True)
    (DEST/'tools/__init__.py').write_text('', encoding='utf-8')
    for relative in FILES:
        source = ROOT/relative
        target = DEST/relative
        if target.exists() and digest(target) != digest(source):
            raise AssertionError('Existing frozen bundle mismatch '+relative)
        shutil.copyfile(source, target)
    manifest = ROOT/'output/camera_2x2_comparison_20261022/manifest.json'
    target_manifest = DEST/'output/camera_2x2_comparison_20261022/manifest.json'
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, target_manifest)
    source_runner = (ROOT/'tools/run_camera_light_post_v2.py').read_text(encoding='utf-8')
    source_runner = source_runner.replace('from tools.build_camera_2x2_viewer import render_html', 'render_html = None  # worker-only: no viewer generation')
    source_runner = source_runner.replace("Path('Y:/'+value)", "Path('/AI_DNS/'+value)")
    source_runner = source_runner.replace("Path('Y:/guang.yang/TextSR/data/vosr_final_evaluation_20261022')", "Path('/AI_DNS/guang.yang/TextSR/data/vosr_final_evaluation_20261022')")
    source_runner = source_runner.replace("Path('Y:/guang.yang/TextSR/data/camera_robust_pre_native_v22_017_fix_20261022')", "Path('/AI_DNS/guang.yang/TextSR/data/camera_robust_pre_native_v22_017_fix_20261022')")
    source_runner = source_runner.replace("Path('Y:/guang.yang/TextSR/data/camera_post_v2_20261023/full')", "Path('"+REMOTE+"/full')")
    if 'from tools.build_camera_2x2_viewer import render_html' in source_runner or "Path('Y:/'+value)" in source_runner:
        raise AssertionError('Remote adaptation incomplete')
    (DEST/'tools/remote_runner_full.py').write_text(source_runner, encoding='utf-8')
    pilot_runner = source_runner.replace("Path('"+REMOTE+"/full')", "Path('"+REMOTE+"/remote_pilot')")
    (DEST/'tools/remote_runner_pilot.py').write_text(pilot_runner, encoding='utf-8')
    (DEST/'probe_runtime.py').write_text("import cv2, numpy as np, sys\nprint(sys.version.split()[0], cv2.__version__, np.__version__)\n", encoding='utf-8')
    items = list(FILES)+['tools/run_camera_light_post_v2.py', 'output/camera_2x2_comparison_20261022/manifest.json']
    report = {'source_hashes': {v: digest(ROOT/v) for v in items},
              'bundle_hashes': {str(v.relative_to(DEST)): digest(v) for v in DEST.rglob('*.py')},
              'manifest_sha256': digest(target_manifest), 'remote_root': REMOTE}
    (DEST/'BUNDLE.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(stage(), indent=2))
