from .gradient_check import check
from pathlib import Path
import json,csv
if __name__=='__main__':
    rows=check();p=Path('results/gradient_check');p.mkdir(parents=True,exist_ok=True)
    (p/'gradient_check.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (p/'gradient_check.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print('parameter | max abs | relative | count | pass')
    for r in rows: print(r)
