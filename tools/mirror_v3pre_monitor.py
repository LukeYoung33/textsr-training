"""Mirror the server-resident standalone training dashboard without mapped-drive writes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from v3pre_dashboard import render_dashboard


def main():
    """@luna-comment Input:local owned output directory. Output:refreshing dashboard and mirror receipt.
    Key Steps:bounded read-only SSH pull, render readable stage/arm cards, atomic
    HTML replace, record actual source
    freshness/failure and retry. Key Params:30s polling; no training control.
    State/Dependencies:SSH Remote58, local Python; last good page retained on disconnect.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'Remote58',
                'cat /AI_DNS/guang.yang/TextSR/data/v3pre_v3m_three_20261027/dashboard.html'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25, check=True)
            source = result.stdout.decode('utf-8')
            assert '<h1>V3 preprocessing / three V3M training chains</h1>' in source
            page = render_dashboard(source)
            temporary = output/'index.html.tmp'
            temporary.write_text(page, encoding='utf-8')
            temporary.replace(output/'index.html')
            status = dict(passed=True, bytes=len(result.stdout))
        except Exception as error:
            status = dict(passed=False, error=repr(error), policy='last good page retained; source timestamp visible')
        status['checked_at'] = datetime.now(timezone.utc).isoformat()
        (output/'MIRROR_STATUS.json').write_text(json.dumps(status, indent=2))
        time.sleep(30)


if __name__ == '__main__':
    main()
