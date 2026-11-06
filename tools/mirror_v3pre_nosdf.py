"""Read-only local dashboard mirror for the independent no-SDF pretrain runs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    """@luna-comment Input:explicit mirror invocation. Output:local self-contained dashboard/freshness receipt. Key Steps:bounded SSH read;UTF8 atomic replace;retain last page on disconnect. Key Params:30s polling. State/Dependencies:Remote58, no training control."""
    output=Path(__file__).resolve().parents[1]/'evaluation/v3pre_v3m_nosdf_250k_20261028';output.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','Remote58','cat /AI_DNS/guang.yang/TextSR/data/v3pre_v3m_nosdf_250k_20261028/dashboard.html'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=20)
            page=result.stdout.decode('utf-8');assert '<html lang="zh-CN">' in page
            # The shared legacy renderer guesses names unless an ASCII separator is present.
            start=page.find('const lossNames = ');end=page.find('\n',start)
            if start>=0:page=page[:start]+'const lossNames = lossArms;'+page[end:]
            page=page.replace(",['l_sdf_boundary','SDF boundary']",'').replace('Training loss history','训练 loss 曲线')
            temp=output/'index.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(output/'index.html')
            state=dict(passed=True,bytes=len(result.stdout),checked_at=datetime.now(timezone.utc).isoformat())
        except Exception as error:state=dict(passed=False,error=repr(error),checked_at=datetime.now(timezone.utc).isoformat())
        (output/'MIRROR_STATUS.json').write_text(json.dumps(state,indent=2),encoding='utf-8');time.sleep(30)


if __name__=='__main__':main()
