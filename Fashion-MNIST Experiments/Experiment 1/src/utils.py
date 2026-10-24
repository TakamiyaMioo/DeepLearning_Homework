from pathlib import Path
from datetime import datetime
import csv,json,yaml
def load_config(path):
    with Path(path).open(encoding='utf-8') as f:return yaml.safe_load(f)
def save_json(data,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def save_csv(rows,path):
    if not rows:return
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run_id():return datetime.now().strftime('%Y%m%d_%H%M%S_%f')
