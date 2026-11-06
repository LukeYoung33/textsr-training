"""Exact fixed32 subset and stored-RGB interpolation toward two thinning targets."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mojito_s22_thinning import digest, save

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
SUBSET = DATA/'mojito280_subset32_20261104'
THIN = DATA/'mojito280_s22_thin_20261104'
OUT = DATA/'mojito32_s22_thin_blend_20261104'


def write_image(path, image):
    """@luna-comment Input:destination,uint8nativeBGR. Output:losslessPNG. Key Steps:atomicPNGwrite anddecodedexactcheck. Key Params:compression3. State/Dependencies:ownedoutput only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem+'.tmp.png')
    assert cv2.imwrite(str(tmp), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert np.array_equal(cv2.imread(str(tmp)), image)
    tmp.replace(path)


def process(pair):
    """@luna-comment Input:selected sample,verified thinning row. Output:10images andreceipt. Key Steps:exactID/hash/nativeorientation checks;copy4endpoints;6roundedRGBblends;bounds/monotonicity. Key Params:alpha.25/.5/.75,twoGTarms. State/Dependencies:CPU/sourceimmutable,noOCR/warp/resampling."""
    cv2.setNumThreads(1)
    selected, row = pair
    ident = selected['id']
    assert ident == row['id'] and selected['sample_id'] == row['name']
    assert digest(selected['input']) == selected['input_sha256']
    assert digest(row['source']['lq']) == row['source_sha256']['lq']
    subset_input = cv2.imread(selected['input'])
    reference_input = cv2.imread(row['source']['lq'])
    if selected['orientation'] == 6:
        subset_input = cv2.rotate(subset_input, cv2.ROTATE_90_CLOCKWISE)
    else:
        assert selected['orientation'] == 1
    assert np.array_equal(subset_input, reference_input), (ident, 'input_identity')
    assert subset_input.shape == (row['height'], row['width'], 3)
    endpoints = {'v3_input': Path(row['source']['lq']), 'original_gt': Path(row['source']['gt'])}
    expected = {'v3_input': row['source_sha256']['lq'], 'original_gt': row['source_sha256']['gt']}
    for arm in ('minus1', 'minus2'):
        endpoints[arm] = THIN/row['outputs'][arm]['path']
        expected[arm] = row['outputs'][arm]['sha256']
    outputs = {}
    for label, source in endpoints.items():
        assert digest(source) == expected[label]
        destination = OUT/label/f'{ident}.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert digest(destination) == expected[label]
        outputs[label] = dict(path=str(destination.relative_to(OUT)), sha256=expected[label])
    source_float = reference_input.astype(np.float32)
    checks = {}
    for arm in ('minus1', 'minus2'):
        gt = cv2.imread(str(endpoints[arm]))
        assert gt.shape == reference_input.shape
        target = gt.astype(np.float32)
        previous = reference_input
        residual = target-source_float
        arm_checks = []
        for percent in (25, 50, 75):
            alpha = percent/100
            ideal = source_float+alpha*residual
            value = np.rint(ideal).astype(np.uint8)
            assert np.all(value >= np.minimum(reference_input, gt)) and np.all(value <= np.maximum(reference_input, gt))
            assert np.all(np.abs(value.astype(np.int16)-gt) <= np.abs(previous.astype(np.int16)-gt))
            error = float(np.abs(value.astype(np.float32)-ideal).max())
            assert error <= .5
            label = arm+f'_a{percent}'
            destination = OUT/label/f'{ident}.png'
            write_image(destination, value)
            outputs[label] = dict(path=str(destination.relative_to(OUT)), sha256=digest(destination), gt_fraction=alpha)
            arm_checks.append(dict(gt_fraction=alpha, max_rounding_error=error, mae_to_gt=float(np.abs(value.astype(np.float32)-target).mean())))
            previous = value
        checks[arm] = arm_checks
    receipt = dict(id=ident, name=selected['name'], width=row['width'], height=row['height'], outputs=outputs, checks=checks,
        input_orientation=selected['orientation'], input_rotation='90CW' if selected['orientation']==6 else 'none',
        subset_input=selected['input'], subset_sha256=selected['input_sha256'], canonical_input_sha256=row['source_sha256']['lq'], decoded_input_exact=True)
    save(OUT/'receipts'/f'{ident}.json', receipt)
    return receipt


def main():
    """@luna-comment Input:fixed32selection andprior280receipts. Output:320verifiedPNG files,manifest/completion. Key Steps:exclusive lock,freeze selection,8CPUworkers,progress/finalreceipt. Key Params:32samples,twoarms,3blendfractions. State/Dependencies:ownedOUT,sourceimmutable."""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = json.loads((SUBSET/'selection.json').read_text())['samples']
    ids = (SUBSET/'ids.txt').read_text().split()
    assert len(selected) == len(set(ids)) == 32 and [x['id'] for x in selected] == ids
    source = {r['id']:r for r in json.loads((THIN/'RESULTS.json').read_text())['rows']}
    save(OUT/'BINDING.json', dict(selection_sha256=digest(SUBSET/'selection.json'), thinning_results_sha256=digest(THIN/'RESULTS.json'),
        code_sha256=digest(__file__), method='stored uint8 RGB float32 convex interpolation;np.rint once', fractions=[.25,.5,.75]))
    shutil.copyfile(SUBSET/'ids.txt', OUT/'ids.txt')
    shutil.copyfile(SUBSET/'selection.json', OUT/'selection.json')
    rows, start = [], time.time()
    with ProcessPoolExecutor(8) as pool:
        futures = [pool.submit(process, (x, source[x['id']])) for x in selected]
        for future in as_completed(futures):
            rows.append(future.result())
            state = dict(completed=len(rows), total=32, seconds=time.time()-start)
            save(OUT/'STATUS.json', state)
            print(json.dumps(state), flush=True)
    rows.sort(key=lambda r:ids.index(r['id']))
    assert sum(len(r['outputs']) for r in rows) == 320
    save(OUT/'RESULTS.json', dict(passed=True, rows=rows))
    save(OUT/'COMPLETE.json', dict(passed=True, images=32, files=320, intermediate_files=192, endpoint_files=128, seconds=time.time()-start))


if __name__ == '__main__':
    main()
