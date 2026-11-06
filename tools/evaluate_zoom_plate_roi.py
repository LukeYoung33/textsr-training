"""Native plate-ROI enhancement with zero-ended feathered residual pasteback."""
from pathlib import Path
import sys,json,copy,shutil,inspect,argparse,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import evaluate_zoom_plate_two_models as core
from tools.run_uniform20_steep_scurve import save,digest
PREVIOUS=core.OUT;core.OUT=core.DATA/'zoom_bmp_plate_roi_two_models_20261105';core.__file__=__file__
# Visually selected plate bounds on 384x512 previews; native images are3072x4096.
BOXES=[(123,219,217,257),(131,235,228,269),(148,208,258,250),(147,225,261,270),
       (129,184,242,225),(145,181,257,220),(135,218,225,254),(133,222,227,261),
       (132,224,219,261),(130,227,216,264),(146,219,243,258),(151,221,250,263)]


def alpha_mask(height,width,feather=32):
    """@luna-comment Input:ROI height/width,pixel feather. Output:float32 2D alpha with exactzero border. Key Steps:nearestedge distance,smoothstep,productaxes. Key Params:32nativepx,platecore outside ramp. State/Dependencies:NumPy purefunction."""
    import numpy as np
    x=np.minimum(np.arange(width),width-1-np.arange(width))/feather;y=np.minimum(np.arange(height),height-1-np.arange(height))/feather
    x=np.clip(x,0,1);y=np.clip(y,0,1);x=x*x*(3-2*x);y=y*y*(3-2*y)
    return (y[:,None]*x[None,:]).astype(np.float32)


def paste(original,prediction,box):
    """@luna-comment Input:originalBGR8,processedROI andxyxy. Output:full image andoutside equality. Key Steps:residual alpha blend,saturatinground,pasteownedrectangle. Key Params:32px featherzeroedge. State/Dependencies:source immutable."""
    import numpy as np
    x0,y0,x1,y1=box;crop=original[y0:y1,x0:x1];assert prediction.shape==crop.shape
    alpha=alpha_mask(y1-y0,x1-x0);value=np.clip(np.rint(crop.astype(np.float32)+alpha[...,None]*(prediction.astype(np.float32)-crop)),0,255).astype(np.uint8)
    final=original.copy();final[y0:y1,x0:x1]=value
    assert np.array_equal(value[alpha==0],crop[alpha==0])
    assert np.array_equal(final[:y0],original[:y0]) and np.array_equal(final[y1:],original[y1:]) and np.array_equal(final[y0:y1,:x0],original[y0:y1,:x0]) and np.array_equal(final[y0:y1,x1:],original[y0:y1,x1:])
    return final


def prepare():
    """@luna-comment Input:explicit12BMPs,visually chosen plates andprior500Kmodels. Output:immutableROIselection/crops/bindings. Key Steps:sourcehash,same-native crop48pxcontext,overlayproof,no warp/resize. Key Params:32pxfeather,corefullyopaque. State/Dependencies:owned newoutput only."""
    import cv2,numpy as np
    out=core.OUT;old=[e for e in json.loads((PREVIOUS/'selection.json').read_text())['samples'] if e['group']=='zoom_bmp'];assert len(old)==12
    selection=[];full=[];proof=[]
    for i,(e,b) in enumerate(zip(old,BOXES),1):
        assert digest(e['source'])==e['source_sha256'];image=cv2.imread(e['source']);assert image.shape==(4096,3072,3)
        plate=[int(n*8) for n in b];box=[max(0,plate[0]-48),max(0,plate[1]-48),min(3072,plate[2]+48),min(4096,plate[3]+48)];x0,y0,x1,y1=box
        source=out/'roi_source'/(e['id']+'.png');sha=core.write(source,image[y0:y1,x0:x1]);entry=copy.deepcopy(e)
        entry.update(source=str(source),source_sha256=sha,format='PNG',width=x1-x0,height=y1-y0,output=str(out/'output'/(e['id']+'.png')),input=str(out/'preprocess'/(e['id']+'.png')));selection.append(entry)
        full.append(dict(id=e['id'],source=e['source'],source_sha256=e['source_sha256'],plate_xyxy=plate,roi_xyxy=box,full_shape=[4096,3072,3],feather=32,selection_method='visually selected original BMP plate bounding rectangle;48px context',native_no_resize=True))
        overlay=cv2.resize(image,(384,512));cv2.rectangle(overlay,(box[0]//8,box[1]//8),(box[2]//8,box[3]//8),(0,255,0),2);cv2.rectangle(overlay,(b[0],b[1]),(b[2],b[3]),(255,128,0),1);cv2.putText(overlay,f'{i:02d}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);proof.append(overlay)
        alpha=alpha_mask(y1-y0,x1-x0);assert np.all(alpha[plate[1]-y0:plate[3]-y0,plate[0]-x0:plate[2]-x0]==1)
    core.write(out/'ROI_SELECTION.png',np.concatenate([np.concatenate(proof[i:i+4],1) for i in range(0,12,4)],0));save(out/'FULL_SOURCE_BINDINGS.json',dict(samples=full))
    bindings=json.loads((PREVIOUS/'MODEL_BINDINGS.json').read_text());(out/'models').mkdir(exist_ok=True)
    for arm,b in bindings.items():
        for key,suffix in [('config','.yml'),('checkpoint','.pth')]:
            source=Path(b[key]);assert digest(source)==b[key+'_sha256'];target=out/'models'/(arm+suffix);shutil.copy2(source,target);assert digest(target)==b[key+'_sha256'];b[key]=str(target)
    save(out/'MODEL_BINDINGS.json',bindings);save(out/'selection.json',dict(samples=selection));run=json.loads((PREVIOUS/'RUN_BINDING.json').read_text());run.update(selection_sha256=digest(out/'selection.json'),outputs=72,roi_only=True,full_source_bindings_sha256=digest(out/'FULL_SOURCE_BINDINGS.json'));save(out/'RUN_BINDING.json',run)


def audit():
    """@luna-comment Input:completed12ROI six-stage results. Output:72fullframe composites,hash/zeroedge/outside invariant audit. Key Steps:verifyrawROIstage/modelbindings;pasteeachstage once intooriginal,neverwholeframepost;checkoriginalunchanged. Key Params:same alpha forpre/model/post. State/Dependencies:owned newfiles."""
    import cv2
    out=core.OUT;rows=json.loads((out/'selection.json').read_text())['samples'];full=json.loads((out/'FULL_SOURCE_BINDINGS.json').read_text())['samples'];bindings=json.loads((out/'MODEL_BINDINGS.json').read_text());run=json.loads((out/'RUN_BINDING.json').read_text());assert len(rows)==len(full)==12
    assert digest(out/'selection.json')==run['selection_sha256'] and digest(out/'FULL_SOURCE_BINDINGS.json')==run['full_source_bindings_sha256']
    for p,sha in run['code_sha256'].items():assert digest(ROOT/p)==sha
    results=[]
    for e,f in zip(rows,full):
        assert e['id']==f['id'] and digest(f['source'])==f['source_sha256'];original=cv2.imread(f['source']);pre=json.loads((out/'pre_receipts'/(e['id']+'.json')).read_text())
        sources={'preprocess':(e['input'],pre['input_sha256'])}
        for a,b in bindings.items():
            assert digest(b['checkpoint'])==b['checkpoint_sha256'] and digest(b['config'])==b['config_sha256'];r=json.loads((out/'receipts'/a/(e['id']+'.json')).read_text());assert r['binding']==b and r['input_sha256']==pre['input_sha256']
            assert r['restoration']['stitch_policy']=='crop8_smooth16'
            sources['model/'+a]=(str(out/'model'/a/(e['id']+'.png')),r['model_sha256']);sources['post/'+a]=(str(out/'post'/a/(e['id']+'.png')),r['post_sha256'])
        outputs={};path=out/'full/output'/(e['id']+'.png');outputs['output']=dict(path=str(path),sha256=core.write(path,original))
        for stage,(source,sha) in sources.items():
            assert digest(source)==sha;value=cv2.imread(source);final=paste(original,value,f['roi_xyxy']);path=out/'full'/stage/(e['id']+'.png');outputs[stage]=dict(path=str(path),sha256=core.write(path,final),outside_roi_unchanged=True,zero_alpha_border_unchanged=True)
        results.append(dict(id=e['id'],source=f['source'],plate_xyxy=f['plate_xyxy'],roi_xyxy=f['roi_xyxy'],feather=32,stages=outputs))
    save(out/'RESULTS.json',dict(passed=True,samples=results));save(out/'COMPLETE.json',dict(passed=True,samples=12,models=2,full_stage_outputs=72,roi_stage_outputs=72,feather_pixels=32,all_outside_roi_unchanged=True,all_zero_alpha_edges_unchanged=True))


def bind():
    """@luna-comment Input:acceptedparallelpipeline. Output:isolatedROI routing/counters/audit. Key Steps:reuseactualpre/model/postworkers;replaceonlyprepare/progress/audit. Key Params:12samples,72stageunits. State/Dependencies:process-local module."""
    source=inspect.getsource(core.progress).replace('168','72').replace('/28','/12').replace('Zoom/车牌','车牌ROI羽化贴回')
    namespace=dict(core.__dict__);exec(compile(source,'<plate_roi_progress>','exec'),namespace);core.progress=namespace['progress'];core.prepare=prepare;core.audit=audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','preprocess','worker']);p.add_argument('--arm',choices=list(core.LABELS));p.add_argument('--rank',type=int,default=0);args=p.parse_args();bind()
    if args.mode=='worker':core.worker(args.arm,args.rank)
    elif args.mode=='preprocess':core.preprocess()
    else:core.run()
