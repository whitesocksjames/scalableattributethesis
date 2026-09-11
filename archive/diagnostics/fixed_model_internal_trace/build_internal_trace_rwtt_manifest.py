#!/usr/bin/env python3
"""Build deterministic 20 low/median/high Train + fixed Val28 trace manifest."""
import argparse
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--statistics',required=True)
p.add_argument('--val28-manifest',required=True)
p.add_argument('--n30-h5-root',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
frame=pd.read_csv(a.statistics).sort_values(['r2_E_D111','source'],kind='stable')
n=len(frame)
groups={'train_low':frame.iloc[:20],
        'train_typical':frame.iloc[n//2-10:n//2+10],
        'train_high':frame.iloc[-20:]}
rows=[]
for group,part in groups.items():
    for _,row in part.iterrows():
        rel=Path(row.source).relative_to('/REDACTED/FAU_PROJECT_ROOT/datasets/RWTT/processed/train_h5/h5/100000')
        rows.append(('RWTT_Train',group,str(Path(a.n30_h5_root)/rel),'h5'))
for line in Path(a.val28_manifest).read_text().splitlines():
    if line.strip(): rows.append(('RWTT_Val','val28',str(Path(a.n30_h5_root)/line.strip()),'h5'))
if len(rows)!=88 or len({r[2] for r in rows})!=88: raise RuntimeError('expected 88 unique H5')
Path(a.output).write_text('\n'.join('\t'.join(r) for r in rows)+'\n')
print('wrote',len(rows),a.output)
