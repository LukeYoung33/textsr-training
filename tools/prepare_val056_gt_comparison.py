"""Source-near training substitution for the requested validation056 three-GT comparison."""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'near_val056_train558_gt_comparison_20261028'


def main():
    """@luna-comment Input:historical056 reference and user-authorized same-family train candidates. Output:three genuine raw GTs and independent V3 variants. Key Steps:rank same acquisition/near crop origin;verify3 target lineages/frozen preprocessing;exact cached2u replay;native shape/shared-delta checks. Key Params:strength1/chunk128,no resize for selected558. State/Dependencies:CPU immutable source, isolated output; no teacher regeneration or training changes."""
    import cv2
    import numpy as np
    from tools.train_v3_edge_pilot import save,digest
    from tools.prepare_v3pre_training import REFERENCES
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    assert {key:digest(ROOT/key) for key in REFERENCES}==REFERENCES
    selection=DATA/'pretrained_four_val583_20261027/selection.json'
    entry=next(e for e in json.loads(selection.read_text())['samples'] if e['id']=='056');reference=entry['row']
    assert reference['sample_id']=='batch_3_043-128_000566' and reference['dataset']=='lenovo0926'
    train_manifest=DATA/'vosr_2u_block256_v1_20261023/source_view/manifests/train.jsonl'
    rows=[json.loads(line) for line in train_manifest.read_text().splitlines() if line.strip()]
    candidates=[r for r in rows if r['dataset']==reference['dataset'] and r['sample_id'].startswith('batch_3_043-128_') and abs(r['crop_xywh'][0]-reference['crop_xywh'][0])<32 and abs(r['crop_xywh'][1]-reference['crop_xywh'][1])<32]
    candidates.sort(key=lambda r:(abs(int(r['sample_id'].rsplit('_',1)[1])-566),r['sample_id']))
    row=candidates[0];assert row['sample_id']=='batch_3_043-128_000558'
    current=next(json.loads(line) for line in (DATA/'v3pre_v3m_three_20261027/pretrain/source_view/manifests/train.jsonl').read_text().splitlines() if line.strip() and json.loads(line)['sample_id']==row['sample_id'])
    assert row['target_refresh']['interpolation']=='identity'
    sources={'original':(row['original_gt_lineage']['path'],row['original_gt_lineage']['sha256']),
             'gt_vosr':(row['previous_gt']['path'],row['previous_gt']['sha256']),
             'gt_vosr2u':(row['files']['gt'],row['output_sha256']['gt'])}
    records={};shape=None
    for key,(filename,expected) in sources.items():
        assert digest(filename)==expected
        image=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
        assert image is not None and image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3
        if shape is None:shape=image.shape
        assert image.shape==shape
        shutil.copyfile(filename,OUT/(key+'.png'))
        estimate=revision2.estimate(image)
        q,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
        low,info=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],q,tail,estimate['nodes'])
        packed=dict(nodes=estimate['nodes'],tables=neighbor.tables(low,estimate['fields']['high']),native_shape=list(image.shape[:2]))
        result,telemetry=pre640.render(image,packed,strength=1.,chunk=128)
        assert result.shape==shape and result.dtype==np.uint8
        delta=result.astype(np.int16)-image.astype(np.int16)
        assert np.array_equal(delta[...,0],delta[...,1]) and np.array_equal(delta[...,1],delta[...,2])
        target=OUT/(key+'_v3.png');assert cv2.imwrite(str(target),result,[cv2.IMWRITE_PNG_COMPRESSION,3])
        record=dict(source=filename,source_sha256=expected,raw_copy_sha256=digest(OUT/(key+'.png')),v3_sha256=digest(target),shape_hw=list(shape[:2]),shared_channel_delta=True)
        if key=='gt_vosr2u':
            cache=current['files']['gt'];assert digest(cache)==current['output_sha256']['gt']
            assert np.array_equal(result,cv2.imread(cache,cv2.IMREAD_UNCHANGED))
            record.update(cached_v3=cache,cached_v3_sha256=current['output_sha256']['gt'],cached_pixel_equal=True)
        records[key]=record;print(json.dumps(dict(completed=key,shape_hw=list(shape[:2]))),flush=True)
    save(OUT/'PROVENANCE.json',dict(id='558',split='train',reference_id='056',reference_sample_id=reference['sample_id'],sample_id=row['sample_id'],dataset=row['dataset'],source_identity=row['source_identity'],reference_source_identity=reference['source_identity'],crop_xywh=row['crop_xywh'],reference_crop_xywh=reference['crop_xywh'],selection_rule='same dataset/acquisition family, crop origin within32px, nearest numeric ID',shape_hw=list(shape[:2]),selection_sha256=digest(selection),train_manifest_sha256=digest(train_manifest),records=records,original_gt_lineage=row['original_gt_lineage'],photometry_version=row.get('photometry_version'),missing=[],preprocessing_references=REFERENCES,complete_six=True))


if __name__=='__main__':main()
