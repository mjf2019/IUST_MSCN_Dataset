"""Summarize saved Burst evaluations without training or rerunning inference."""
import argparse
import csv
import json
import re
from pathlib import Path


def collect(root, seed, lr, reward_mode, replay_size, continual=False, session_id=None):
    selected = {}
    for path in sorted(root.glob('*_metrics.json'), key=lambda p:p.stat().st_mtime):
        match = re.search(r'AdvBurstTrainedBurst_size_(\d+)_Step', path.name)
        if not match:
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        if (d.get('seed') != seed or d.get('online_lr') != lr
                or d.get('routing_reward_mode','absolute') != reward_mode
                or d.get('context_replay_size',0) != replay_size
                or d.get('stream_order') != 'shuffle'
                or d.get('eval_epsilon') != 0.2
                or d.get('budget_fraction') != 0.2
                or d.get('selection_cost') != 0.0001):
            continue
        if bool(d.get('continual',False)) != continual or (session_id is not None and d.get('continual_session') != session_id):
            continue
        mode = d.get('selector_mode')
        if mode in ('learned','random'):
            selected[int(match[1]),mode] = d
    return selected


def display(rows, keys):
    def value(v):
        return '-' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)
    widths={k:max(len(k),*(len(value(r.get(k))) for r in rows)) for k in keys}
    print(' '.join(k.rjust(widths[k]) for k in keys))
    for r in rows:
        print(' '.join(value(r.get(k)).rjust(widths[k]) for k in keys))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=0.0001)
    p.add_argument('--routing-reward-mode',choices=['absolute','relative'],default='absolute')
    p.add_argument('--context-replay-size',type=int,default=0)
    p.add_argument('--levels',type=int,nargs='+',default=[1,5,7,10,15,17,20])
    p.add_argument('--continual',action='store_true')
    p.add_argument('--session-id')
    a=p.parse_args();root=a.output/'Results'
    selected=collect(root,a.seed,a.lr,a.routing_reward_mode,a.context_replay_size,a.continual,a.session_id)
    if not selected:p.error('No matching metrics found in '+str(root))
    rows=[]
    for level in a.levels:
        row={'PER':level}
        for mode,prefix in [('learned','RL'),('random','Random')]:
            d=selected.get((level,mode))
            if d is None:continue
            f=d.get('forgetting',{});ref=f.get('source_validation',{});same=f.get('same_level',{})
            row.update({prefix+'_Acc':100*d['accuracy'],prefix+'_F1':100*d['macro_f1'],prefix+'_Labels':d['unique_labels_queried'],prefix+'_Oracle':100*d['routing_oracle']['accuracy'] if 'routing_oracle' in d else None,prefix+'_RefInitial':100*ref['initial_accuracy'] if ref else None,prefix+'_RefFinal':100*ref['final_accuracy'] if ref else None,prefix+'_RefDrop':ref.get('drop_pp'),prefix+'_CurrentFinal':100*same['final_accuracy'] if same else None})
            row['Frozen']=100*d['base_accuracy']
        if 'RL_Acc' in row and 'Random_Acc' in row:row['Gain_pp']=row['RL_Acc']-row['Random_Acc']
        rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r));mean={'PER':'Mean'}
    for k in fields:
        values=[r[k] for r in rows if k!='PER' and isinstance(r.get(k),(int,float))]
        if values:mean[k]=sum(values)/len(values)
    rows.append(mean)
    print('\nCombined results (%; Gap/Drop in percentage points; positive Drop = net forgetting):')
    columns = {
        'PER':'PER', 'Frozen':'Frozen', 'RL':'RL_Acc', 'Random':'Random_Acc',
        'Gap':'Gain_pp', 'OracleRL':'RL_Oracle', 'OracleRnd':'Random_Oracle',
        'LabRL':'RL_Labels', 'LabRnd':'Random_Labels',
        'Ref0':'RL_RefInitial', 'RefRL':'RL_RefFinal', 'RefRnd':'Random_RefFinal',
        'DropRL':'RL_RefDrop', 'DropRnd':'Random_RefDrop',
        'FinalRL':'RL_CurrentFinal', 'FinalRnd':'Random_CurrentFinal'}
    display([{short:r.get(full) for short,full in columns.items()} for r in rows],list(columns))
    print('Ref: clean reference; Final: current-level Context-only final accuracy; RL/Random: online full-system accuracy.')
    missing=[f'{n}/{m}' for n in a.levels for m in ('learned','random') if (n,m) not in selected]
    if missing:print('Missing evaluations (shown as -): '+', '.join(missing))
    suffix=('_continual_'+(a.session_id or 'latest')) if a.continual else ''
    target=root/f'combined_seed{a.seed}_lr{a.lr:g}_replay{a.context_replay_size}_{a.routing_reward_mode}{suffix}.csv'
    with target.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('\nCombined summary saved:',target.resolve())
    print('Oracle is an observer-only route ceiling along each policy update trajectory.')

if __name__=='__main__':main()
