"""Read-only local mirror of the bounded experiment dashboard."""
from datetime import datetime,timezone
import json
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.build_camera_2x2_viewer import render_html


def main():
    """@luna-comment Input:explicit local mirror invocation. Output:refreshing local HTML/receipt. Key Steps:bounded read-only SSH pull and atomic replace. Key Params:15s poll,20s timeout. State/Dependencies:Remote58; last good snapshot retained on disconnect."""
    run_name='v3_edge_cont10k_20261028' if '--continue-2k' in sys.argv else 'v3_edge_pilot_20261028'
    remote_root='/AI_DNS/guang.yang/TextSR/data/'+run_name
    output=Path(__file__).resolve().parents[1]/'evaluation'/run_name
    output.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','Remote58','cat '+remote_root+'/dashboard.html'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=20)
            page=result.stdout.decode('utf-8');assert '<html lang="zh-CN">' in page
            if '--continue-2k' in sys.argv:page=page.replace('同一 Full250K 权重与 Adam 状态','共同 Full250K 起点；恢复各组自己的 2K 权重与 Adam 状态')
            temp=output/'index.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(output/'index.html')
            # Download only new immutable checkpoint previews; dashboard does not depend on Y mapping.
            comparison=output/'comparison';comparison.mkdir(exist_ok=True)
            manifest=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','Remote58','cat '+remote_root+'/comparison/manifest.json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
            if manifest.returncode==0:
                inventory=json.loads(manifest.stdout)
                filenames={name for row in inventory['rows'] for name in [row['lq'],*row['images'].values()]}
                missing=[name for name in sorted(filenames) if not (comparison/name).exists()]
                if missing:
                    staging=comparison/'_download';staging.mkdir(exist_ok=True)
                    command='tar -cf - -C '+remote_root+'/comparison -- '+' '.join(shlex.quote(name) for name in missing)
                    bundle=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','Remote58',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=50)
                    with tarfile.open(fileobj=io.BytesIO(bundle.stdout),mode='r:') as archive:
                        for member in archive.getmembers():
                            assert member.isfile() and member.name in missing and Path(member.name).name==member.name
                            target=staging/member.name;target.write_bytes(archive.extractfile(member).read());target.replace(comparison/member.name)
                for filename,content in [('manifest.json',manifest.stdout),('viewer.html',render_html(inventory).replace('← Previous','← 上一张').replace('Next →','下一张 →').replace("comparisonState='?????????'","comparisonState='加载中'").encode('utf-8'))]:
                    temporary=comparison/(filename+'.tmp');temporary.write_bytes(content);temporary.replace(comparison/filename)
            for filename in ['METRICS.json']:
                remote=remote_root+'/comparison/'+filename
                fetched=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','Remote58','cat '+remote],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
                if fetched.returncode==0:
                    temporary=comparison/(filename+'.tmp');temporary.write_bytes(fetched.stdout);temporary.replace(comparison/filename)
            if (comparison/'viewer.html').exists():
                page=page.replace('<h1>','<p><a href="comparison/viewer.html">打开 049–052 早期检查点效果对比</a></p><h1>',1)
                temp=output/'index.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(output/'index.html')
            state=dict(passed=True,bytes=len(result.stdout),checked_at=datetime.now(timezone.utc).isoformat())
        except Exception as error:state=dict(passed=False,error=repr(error),checked_at=datetime.now(timezone.utc).isoformat())
        (output/'MIRROR_STATUS.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        time.sleep(15)


if __name__=='__main__':main()
