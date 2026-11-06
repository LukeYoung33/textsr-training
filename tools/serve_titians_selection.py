"""Loopback selection UI with durable choices and allowlisted paired export."""
from pathlib import Path
import sys,json,threading,time,secrets,shutil,os,mimetypes
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_titians_selection import OUT,KEYS,sha,save
from tools.build_camera_2x2_viewer import render_html
LOCK=threading.Lock();TOKEN=secrets.token_hex(24);EXPORT={'phase':'idle'}


def manifest():
    """@luna-comment Input:complete local manifest. Output:validated row mapping. Key Steps:load frozen inventory. Key Params:160 unique IDs. State/Dependencies:read-only."""
    source=Path(os.environ.get('TITIANS_SELECTION_MANIFEST',str(OUT/'manifest.json')))
    m=json.loads(source.read_text(encoding='utf-8'));assert len(m['rows'])==len({r['id'] for r in m['rows']})==160
    if os.name=='nt':
        for r in m['rows']:
            for key in ['lq','original']:r[key]=r[key].replace('/AI_DNS/','Y:/',1)
            r['images']={k:v.replace('/AI_DNS/','Y:/',1) for k,v in r['images'].items()}
    return m


def choices():
    """@luna-comment Input:optional saved choices. Output:selection dictionary. Key Steps:read durable JSON. Key Params:no default selection. State/Dependencies:caller lock."""
    p=OUT/'choices.json';return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def validate(picks,rows,complete=False):
    """@luna-comment Input:untrusted choices. Output:validated mapping or error. Key Steps:restrict IDs/keys;optional complete coverage. Key Params:one value per scene. State/Dependencies:no user-provided file paths."""
    if not isinstance(picks,dict) or any(k not in rows or v not in KEYS for k,v in picks.items()):raise ValueError('选择内容无效')
    if complete and set(picks)!=set(rows):raise ValueError('请为全部160组各选一张后再导出')
    return picks


def export(picks,rows):
    """@luna-comment Input:validated frozen160 selections. Output:paired folder and hashed provenance. Key Steps:background copy/hardlink,unique names,verify each pair;complete marker last. Key Params:exact original JPG,selected native PNG;never overwrite. State/Dependencies:owned export root."""
    global EXPORT
    dest=OUT/'exports'/time.strftime('selected_%Y%m%d_%H%M%S');dest=dest.with_name(dest.name+'_'+secrets.token_hex(3))
    try:
        (dest/'selected').mkdir(parents=True);(dest/'original').mkdir();records=[]
        for n,(ident,row) in enumerate(rows.items(),1):
            key=picks[ident];selected=Path(row['images'][key]);original=Path(row['original']);stem=Path(row['name']).stem
            rec=dict(id=ident,filename=row['name'],selected_key=key,source_result=str(selected),original_source=str(original))
            for kind,src,target in [('selected',selected,dest/'selected'/(stem+'.png')),('original',original,dest/'original'/row['name'])]:
                expected=sha(src)
                try:os.link(src,target)
                except OSError:shutil.copy2(src,target)
                assert sha(target)==expected;rec[kind+'_sha256']=expected
            records.append(rec)
            with LOCK:EXPORT=dict(phase='running',completed=n,total=len(rows),folder=str(dest))
            save(dest/'progress.json',EXPORT)
        save(dest/'selection_manifest.json',dict(picks=picks,records=records,postprocess='frozen current Y gain.75 deadband2 softcap24'))
        save(dest/'COMPLETE.json',dict(passed=True,pairs=len(records)))
        with LOCK:EXPORT=dict(phase='complete',completed=len(rows),total=len(rows),folder=str(dest));save(OUT/'LAST_EXPORT.json',EXPORT)
    except Exception as e:
        with LOCK:EXPORT=dict(phase='failed',error=str(e),folder=str(dest));save(OUT/'LAST_EXPORT.json',EXPORT)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        """@luna-comment Input:request log args. Output:none. Key Steps:suppress noisy image logs. Key Params:errors returned to UI. State/Dependencies:none."""
        pass

    def respond(self,value,status=200):
        """@luna-comment Input:JSON response. Output:UTF8 HTTP. Key Steps:no-store response. Key Params:explicit length. State/Dependencies:socket."""
        data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

    def do_GET(self):
        """@luna-comment Input:loopback GET. Output:gallery,state or allowlisted image. Key Steps:ID lookup,never arbitrary path;stream image. Key Params:no directory browsing. State/Dependencies:complete manifest."""
        try:
            path=urlparse(self.path).path
            if path=='/state':
                with LOCK:self.respond(dict(picks=choices(),export=EXPORT));return
            if path=='/':
                source=Path(os.environ.get('TITIANS_SELECTION_MANIFEST',str(OUT/'manifest.json')))
                if not source.exists():
                    progress=source.parent/'STATUS.json';state=json.loads(progress.read_text()) if progress.exists() else dict(phase='等待服务器完成后处理')
                    import html
                    data=('<meta charset="utf-8"><meta http-equiv="refresh" content="15"><h1>正在准备选图页面</h1><p>服务器后处理完成后自动进入，15秒刷新。</p><pre>'+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>').encode('utf-8');self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
                m=manifest()
                for r in m['rows']:
                    r['lq']='/image/'+r['id']+'/lq';r['images']={k:'/image/'+r['id']+'/'+k for k in KEYS}
                body=render_html(m);extra=(ROOT/'tools/titians_selection_ui.js').read_text(encoding='utf-8').replace('__TOKEN__',TOKEN)
                body=body.replace('</html>','<script>'+extra+'</script></html>');data=body.encode('utf-8');self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            if path.startswith('/image/'):
                _,_,ident,key=path.split('/');rows={r['id']:r for r in manifest()['rows']};r=rows[ident];p=Path(r['lq'] if key=='lq' else r['images'][key]);size=p.stat().st_size
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(p.name)[0] or 'image/png');self.send_header('Content-Length',str(size));self.send_header('Cache-Control','private,max-age=3600');self.end_headers()
                with p.open('rb') as f:shutil.copyfileobj(f,self.wfile)
                return
            self.respond(dict(error='Not found'),404)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        except Exception as e:self.respond(dict(error=str(e)),400)

    def do_POST(self):
        """@luna-comment Input:token-authorized choice/export request. Output:durable selection or async export ID. Key Steps:validate payload and origin;lock mutation;export snapshot only after all selected. Key Params:64KiB maximum,no arbitrary destination. State/Dependencies:loopback token."""
        global EXPORT
        try:
            if self.headers.get('X-Selection-Token')!=TOKEN:raise ValueError('页面已过期，请刷新')
            length=int(self.headers.get('Content-Length','0'));assert 0<length<65536
            body=json.loads(self.rfile.read(length));rows={r['id']:r for r in manifest()['rows']}
            with LOCK:
                if self.path=='/choose':
                    pick=validate({body['id']:body['key']},rows);saved=choices();saved.update(pick);save(OUT/'choices.json',saved);self.respond(dict(saved=True,count=len(saved)));return
                if self.path=='/export':
                    if EXPORT['phase']=='running':raise ValueError('正在导出，请等待')
                    picks=validate(choices(),rows,complete=True);EXPORT=dict(phase='running',completed=0,total=160)
                    threading.Thread(target=export,args=(dict(picks),rows),daemon=True).start();self.respond(EXPORT);return
            self.respond(dict(error='Not found'),404)
        except Exception as e:self.respond(dict(error=str(e)),400)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);ThreadingHTTPServer(('127.0.0.1',8876),Handler).serve_forever()
