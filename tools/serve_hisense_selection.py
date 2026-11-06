"""Independent Hisense68 four-candidate selection and verified file export."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse
import argparse
import copy
import json
import secrets
import shutil
import sys
import time

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import serve_titians_selection as service
from tools.prepare_titians_selection import save,sha

SOURCE=Path('Y:/guang.yang/TextSR/data/hisense_l772_two_models_20261105')
OUT=ROOT/'output/hisense_l772_selection_20261105'
KEYS=('v3_input','v3m_dilate2_r1_width','v3m_mf_uv_mix50','lq')


def local_path(path):
    """@luna-comment Input:trustedNASpath. Output:Windowsmappedpath. Key Steps:singleprefixreplacement. Key Params:AI_DNS->Y. State/Dependencies:no request-controlledpaths."""
    return path.replace('/AI_DNS/','Y:/',1)


def check_header(item):
    """@luna-comment Input:path/expectednativeWH. Output:validatedimageheader. Key Steps:PILheadercompare. Key Params:no decoding/reencoding. State/Dependencies:read-onlysources."""
    path,size=item
    with Image.open(path) as image:
        assert image.size==size,(path,image.size,size)
    return True


def prepare():
    """@luna-comment Input:complete68source/results metadata. Output:isolatedfourcandidate frozenmanifest. Key Steps:IDmatching,exactpoststages,272headers,hashbindings. Key Params:numericUIIDsretain sourceID/name. State/Dependencies:no model/OCR/post rerun,choicesuntouched."""
    done=json.loads((SOURCE/'COMPLETE.json').read_text());assert done['passed'] and done['samples']==68
    result=json.loads((SOURCE/'RESULTS.json').read_text());assert result['passed']
    selected=json.loads((SOURCE/'selection.json').read_text())['samples']
    results={r['id']:r for r in result['samples']}
    assert len(selected)==len(results)==68 and {r['id'] for r in selected}==set(results)
    rows=[];headers=[]
    stages={'v3_input':'preprocess','v3m_dilate2_r1_width':'r1_width_post','v3m_mf_uv_mix50':'mf_uv_mix50_post'}
    for index,s in enumerate(selected,1):
        r=results[s['id']];assert r['source']==s['source']
        paths={k:local_path(r['stages'][v]['path']) for k,v in stages.items()}
        hashes={k:r['stages'][v]['sha256'] for k,v in stages.items()}
        paths['lq']=local_path(s['source']);hashes['lq']=s['source_sha256']
        rows.append(dict(id=f'{index:03d}',source_id=s['id'],name=Path(s['source']).name,width=s['width'],height=s['height'],
                         lq=paths['lq'],images=paths,expected_sha256=hashes))
        headers.extend((p,(s['width'],s['height'])) for p in paths.values())
    with ThreadPoolExecutor(8) as pool:assert all(pool.map(check_header,headers))
    models=[dict(key=k,label=label,note=note) for k,label,note in [
        ('v3_input','V3预处理输入','已完成的预处理图'),
        ('v3m_dilate2_r1_width','V3M · 笔画加粗GT＋粗细R1｜固化后处理','现有r1_width_post结果；无再次增强'),
        ('v3m_mf_uv_mix50','V3M · MF/UV mix50｜固化后处理','现有mf_uv_mix50_post结果；无再次增强'),
        ('lq','原始 LQ','原始JPG；导出保留原始文件')]]
    data=dict(title='Hisense L772 · 68张选图',subtitle='每张单选 · 自动保存 · 选完导出新文件夹',
        notes='默认：V3输入 / 两个模型固化后处理 / 原始LQ。每张勾选一个版本，可随时改选；选择自动保存。选完68张后导出，按实际选择复制原文件，保留PNG/JPG格式并附带清单。',
        defaults=','.join(KEYS),models=models,rows=rows)
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'manifest.json',data)
    save(OUT/'SOURCE_AUDIT.json',dict(passed=True,images=68,headers=272,results_sha256=sha(SOURCE/'RESULTS.json'),selection_sha256=sha(SOURCE/'selection.json')))


def manifest():
    """@luna-comment Input:ownedmanifest. Output:validatedfourchoiceinventory. Key Steps:uniqueIDs/allowlistcheck. Key Params:68productionrows. State/Dependencies:localdisk."""
    data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert len(data['rows'])==len({r['id'] for r in data['rows']})==68
    assert all(set(r['images'])==set(KEYS) for r in data['rows'])
    return data


def validate(picks,rows,complete=False):
    """@luna-comment Input:untrustedchoiceobject/frozenrows. Output:allowlistedmapping. Key Steps:type/ID/key/fullcoverage. Key Params:onekeyperID,no defaultpick. State/Dependencies:no writes."""
    if not isinstance(picks,dict) or any(k not in rows or not isinstance(v,str) or v not in KEYS for k,v in picks.items()):
        raise ValueError('选择内容无效')
    if complete and set(picks)!=set(rows):
        raise ValueError(f'请为全部{len(rows)}张图各选一个版本后再导出')
    return picks


def export(picks,rows):
    """@luna-comment Input:completechoice snapshot/frozenrows. Output:newselectedfolder andhashmanifest. Key Steps:validate,comparefrozenSHA,copyactualextension,verifycopy,COMPLETElast. Key Params:never overwrite,PNG/JPG unchanged. State/Dependencies:ownexports/status;sourceimmutable."""
    dest=OUT/'exports'/('selected_'+time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3))
    try:
        validate(picks,rows,True)
        (dest/'selected').mkdir(parents=True)
        records=[]
        for n,(ident,row) in enumerate(rows.items(),1):
            key=picks[ident];source=Path(row['images'][key]);expected=row['expected_sha256'][key]
            assert sha(source)==expected,(ident,'source changed')
            target=dest/'selected'/(ident+'_'+Path(row['name']).stem+source.suffix.lower())
            shutil.copy2(source,target);assert sha(target)==expected
            records.append(dict(id=ident,source_id=row['source_id'],selected_key=key,source=str(source),file=target.name,sha256=expected))
            with service.LOCK:service.EXPORT=dict(phase='running',completed=n,total=len(rows),folder=str(dest),selected_folder=str(dest/'selected'))
            save(dest/'progress.json',service.EXPORT)
        save(dest/'selection_manifest.json',dict(picks=picks,records=records))
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
        """@luna-comment Input:baseJSONresponse. Output:correctHisenseprogresscount. Key Steps:copyresponse,normalizelegacytotal. Key Params:actualmanifestcount. State/Dependencies:baseHTTPwriter."""
        value=copy.deepcopy(value);state=value.get('export',value)
        if isinstance(state,dict) and state.get('phase')=='running':state['total']=68
        super().respond(value,status)

    def do_GET(self):
        """@luna-comment Input:loopbackGET. Output:fourpanel selectableUI orbaseimage/state. Key Steps:trustedURLrewrite,injectexistingselectionJS/token withHisenselabels. Key Params:no arbitraryimagepath. State/Dependencies:existingsharedrenderer."""
        if urlparse(self.path).path!='/':return super().do_GET()
        try:
            data=manifest()
            for row in data['rows']:
                row['lq']='/image/'+row['id']+'/lq'
                row['images']={k:'/image/'+row['id']+'/'+k for k in KEYS}
            text=service.render_html(data)
            js=(ROOT/'tools/titians_subset_selection_ui.js').read_text(encoding='utf-8').replace('__TOKEN__',service.TOKEN)
            js=js.replace('titians_subset75','hisense_l772_68')
            body=text.replace('</html>','<script>'+js+'</script></html>').encode('utf-8')
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        except Exception as error:self.respond(dict(error=str(error)),400)


def configure():
    """@luna-comment Input:preparedHisensemanifest. Output:process-localbasebindings. Key Steps:scopeOUT/allowlist/validation/export,restorepicks/status. Key Params:separateprocessfrom8877. State/Dependencies:ownedlocalstate."""
    service.OUT=OUT;service.KEYS=KEYS;service.manifest=manifest;service.validate=validate;service.export=export
    if (OUT/'LAST_EXPORT.json').exists():service.EXPORT=json.loads((OUT/'LAST_EXPORT.json').read_text())
    validate(service.choices(),{r['id']:r for r in manifest()['rows']})


def main():
    """@luna-comment Input:port/prepare CLI. Output:independentresidentloopbackselectionservice. Key Steps:prepareifmissing,configure,serve. Key Params:8878default. State/Dependencies:sourceNASavailable;existingservicesuntouched."""
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8878);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if not (OUT/'manifest.json').exists() or args.prepare_only:prepare()
    if args.prepare_only:return
    configure();print(f'http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
