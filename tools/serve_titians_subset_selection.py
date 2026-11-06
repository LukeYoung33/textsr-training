"""Dedicated75-scene Titians selection UI using the existing loopback service."""
from pathlib import Path
import argparse
import copy
import json
import os
import secrets
import shutil
import sys
import time
from urllib.parse import unquote, urlparse
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import serve_titians_selection as service
from tools.prepare_titians_selection import save,sha

OUT=ROOT/'output/titians_subset75_selection_20261105'
SOURCE=Path('Y:/guang.yang/TextSR/data/titians_two_500k_20261105')
RANGES=[(26,30),(36,40),(61,65),(96,100),(111,120),(161,165),(186,190),(226,235),(241,255),(281,290)]
IDS=[f'{n:03d}' for lo,hi in RANGES for n in range(lo,hi+1)]


def local_path(value):
    """@luna-comment Input:frozen manifest path/URL. Output:local mapped path. Key Steps:decode file scheme and remote NAS prefix. Key Params:no arbitrary request paths. State/Dependencies:trusted source metadata."""
    if value.startswith('file:///'):
        value=unquote(value[len('file:///'):])
    return value.replace('/AI_DNS/','Y:/',1) if os.name=='nt' else value.replace('Y:/','/AI_DNS/',1)


def prepare():
    """@luna-comment Input:completed300-image comparison and explicit75 IDs. Output:isolated subset manifest and frozen provenance. Key Steps:completion/audit metadata,exact subset,normalize paths,keep original IDs. Key Params:no inference or old-image revalidation. State/Dependencies:read-only NAS results,owned OUT."""
    assert len(IDS)==len(set(IDS))==75
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    assert json.loads((SOURCE/'AUDIT.json').read_text())['outputs']==1200
    source=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    rows={r['id']:r for r in source['rows']}
    assert set(IDS)<=set(rows)
    subset=copy.deepcopy(source)
    subset['rows']=[copy.deepcopy(rows[k]) for k in IDS]
    for row in subset['rows']:
        row['lq']=local_path(row['lq'])
        row['images']={k:local_path(v) for k,v in row['images'].items()}
        row['images']['lq']=row['lq']
    subset.update(title='Titians · 75张独立选图',subtitle='每张单选 · 自动保存 · 选完导出到独立目录',
        defaults='v3_input,v3m_dilate2_r1_width,v3m_mf_uv_mix50,lq',
        notes='仅包含指定75张，保留原始编号。默认两个模型的固化后处理结果；下拉可切换模型原始输出。勾选的是当前窗口对应的具体版本。每张仅保存一个选择，不会自动代选。完成75张后点击导出，文件复制至独立selected目录并附带选择清单。')
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'manifest.json',subset)
    save(OUT/'SUBSET_AUDIT.json',dict(passed=True,ids=IDS,count=75,source_manifest_sha256=sha(SOURCE/'manifest.json'),old_images_rechecked=False))
    target=ROOT/'evaluation/titians_subset75_selection_20261105'
    target.mkdir(exist_ok=True)
    (target/'index.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="0;url=http://127.0.0.1:8877/">',encoding='utf-8')


def manifest():
    """@luna-comment Input:owned prepared manifest. Output:validated75 row inventory. Key Steps:exact ID/order and model allowlist. Key Params:no mutable arbitrary source paths. State/Dependencies:local manifest only."""
    data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert [r['id'] for r in data['rows']]==IDS
    keys={m['key'] for m in data['models']}
    assert len(keys)==6
    assert all(set(r['images'])==keys for r in data['rows'])
    return data


def validate(picks,rows,complete=False):
    """@luna-comment Input:untrusted single-choice mapping. Output:validated allowlist mapping. Key Steps:ID/key/type validation,optional full coverage. Key Params:75 scenes,one value per ID. State/Dependencies:frozen rows/model keys."""
    if not isinstance(picks,dict) or any(k not in rows or not isinstance(v,str) or v not in service.KEYS for k,v in picks.items()):
        raise ValueError('选择内容无效')
    if complete and set(picks)!=set(rows):
        raise ValueError(f'请为全部{len(rows)}张图各选一个结果后再导出')
    return picks


def export(picks,rows):
    """@luna-comment Input:complete frozen75 choices. Output:copied selected PNGs/hash provenance in new directory. Key Steps:unique destination,copy actual selected version,verify hashes,COMPLETE last. Key Params:no hard links or overwrites. State/Dependencies:allowlisted read-only images,owned exports root."""
    dest=OUT/'exports'/('selected_'+time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3))
    try:
        validate(picks,rows,complete=True)
        (dest/'selected').mkdir(parents=True)
        records=[]
        bindings=json.loads((SOURCE/'MODEL_BINDINGS.json').read_text())
        for n,(ident,row) in enumerate(rows.items(),1):
            key=picks[ident];src=Path(row['images'][key]);target=dest/'selected'/(ident+'_'+Path(row['name']).stem+'.png')
            expected=sha(src)
            shutil.copy2(src,target)
            assert sha(target)==expected
            arm=key[:-4] if key.endswith('_raw') else key
            records.append(dict(id=ident,filename=row['name'],selected_key=key,source_result=str(src),selected_file=str(target),sha256=expected,model_binding=bindings.get(arm),postprocessed=key in bindings))
            with service.LOCK:
                service.EXPORT=dict(phase='running',completed=n,total=len(rows),folder=str(dest),selected_folder=str(dest/'selected'))
            save(dest/'progress.json',service.EXPORT)
        save(dest/'selection_manifest.json',dict(picks=picks,records=records,source_manifest_sha256=sha(OUT/'manifest.json')))
        save(dest/'COMPLETE.json',dict(passed=True,images=len(records)))
        with service.LOCK:
            service.EXPORT=dict(phase='complete',completed=len(rows),total=len(rows),folder=str(dest),selected_folder=str(dest/'selected'))
            save(OUT/'LAST_EXPORT.json',service.EXPORT)
    except Exception as error:
        with service.LOCK:
            service.EXPORT=dict(phase='failed',error=str(error),folder=str(dest))
            save(OUT/'LAST_EXPORT.json',service.EXPORT)


class Handler(service.Handler):
    def respond(self,value,status=200):
        """@luna-comment Input:accepted service JSON response. Output:subset-sized export progress. Key Steps:normalize initial legacy total in response copy. Key Params:75 rows. State/Dependencies:existing160 service remains unchanged."""
        value=copy.deepcopy(value)
        state=value.get('export',value)
        if isinstance(state,dict) and state.get('phase')=='running':
            state['total']=len(IDS)
        return super().respond(value,status)

    def do_GET(self):
        """@luna-comment Input:loopback GET. Output:custom selectable gallery or accepted allowlisted service response. Key Steps:rewrite only trusted model paths,inject dedicated JS/token. Key Params:four synchronized panels,six choices. State/Dependencies:existing service handles images/state."""
        if urlparse(self.path).path!='/':
            return super().do_GET()
        try:
            data=manifest()
            for row in data['rows']:
                row['lq']='/image/'+row['id']+'/lq'
                row['images']={k:'/image/'+row['id']+'/'+k for k in service.KEYS}
            text=service.render_html(data)
            js=(ROOT/'tools/titians_subset_selection_ui.js').read_text(encoding='utf-8').replace('__TOKEN__',service.TOKEN)
            body=text.replace('</html>','<script>'+js+'</script></html>').encode('utf-8')
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
            pass
        except Exception as error:
            self.respond(dict(error=str(error)),400)


def configure():
    """@luna-comment Input:prepared75 manifest. Output:existing service scoped to isolated dataset. Key Steps:bind own output/keys/validation/export,restore last export status. Key Params:existing160 service unaffected in separate process. State/Dependencies:process-local module state."""
    data=manifest()
    service.OUT=OUT
    service.KEYS=tuple(m['key'] for m in data['models'])
    service.manifest=manifest
    service.validate=validate
    service.export=export
    if (OUT/'LAST_EXPORT.json').exists():
        service.EXPORT=json.loads((OUT/'LAST_EXPORT.json').read_text(encoding='utf-8'))
    saved=service.choices()
    validate(saved,{r['id']:r for r in data['rows']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--port',type=int,default=8877);args=parser.parse_args()
    prepare()
    if not args.prepare_only:
        configure()
        print(json.dumps(dict(url=f'http://127.0.0.1:{args.port}/',images=len(IDS),output=str(OUT))),flush=True)
        ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
