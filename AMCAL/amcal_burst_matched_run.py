"""Separate train/test commands for the final bank and all five matched baselines.

Requires already prepared official data; never downloads or overwrites a run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]


def commands(a):
    burst=a.dataset/f'AdvBurst_FTSC-{a.representation}'
    levels=['5','7','10','12','15','17','20']
    methods=['cnn','ral','af','dfe','arf']
    baseline_common=['--burst-dir',str(burst),'--output',str(a.baseline_run),
                     '--methods',*methods,'--seed',str(a.seed),'--device',a.device]
    bank_common=['--burst-dir',str(burst),'--output',str(a.amcal_run),
                 '--seed',str(a.seed),'--context-lr','0.0005','--budget-fraction','0.2',
                 '--query-schedule','policy','--update-policy','learned','--device',a.device]
    if a.command=='evaluate':
        return [
            ['AMCAL/amcal_burst_structured_bank_protocol.py','evaluate',*bank_common,
             '--levels',*levels,'--eval-epsilon','0.05','--modes','learned','random'],
            ['AMCAL/amcal_burst_baselines.py','evaluate',*baseline_common,
             '--levels',*levels,'--budget-fraction','0.2','--ral-budget-mode','prefix']]
    result=[]
    if not a.source_run.exists():
        result.append(['AMCAL/amcal_burst_two_stage_protocol.py','train',
            '--burst-dir',str(burst),'--burst-mapping-dir',str(burst),
            '--official-split-dir',str(a.dataset),'--output',str(a.source_run),
            '--seed',str(a.seed),'--multishift-levels','0','1','3',
            '--stage1-rl-epochs','12','--cnn-multishift-epochs','30',
            '--cnn-multishift-patience','10','--stage2-pretrain-epochs','20',
            '--base-epochs','100','--base-patience','15','--pretrain-epochs','20',
            '--context-epochs','30','--context-patience','10',
            '--train-online-lr','0.0001','--budget-fraction','0.2',
            '--selection-cost','0.0001','--routing-reward-mode','absolute','--device',a.device])
    result.extend([
        ['AMCAL/amcal_burst_structured_bank_protocol.py','train',*bank_common,
         '--source-run',str(a.source_run),'--levels','0','1','3',
         '--epochs','12','--patience','5','--router-warmup-episodes','3','--train-epsilon','0.2'],
        ['AMCAL/amcal_burst_baselines.py','train',*baseline_common,
         '--source-run',str(a.amcal_run),'--train-levels','0','1','3',
         '--budget-fraction','0.2','--af-epochs','30','--dfe-epochs','200',
         '--patience','10','--arf-trees','10','--arf-train-passes','1']])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--representation',choices=['IAT','PS'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    for name in ('dataset','source-run','amcal-run','baseline-run'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    root=REPOSITORY/'AMCAL'/f'ANT_Burst_{a.representation}_TEST'
    runs=root/'protocol_runs'
    a.dataset=(a.dataset or root/'Dataset_official_v1').resolve()
    a.source_run=(a.source_run or runs/f'three_action_twostage_levels1_3_contextpre20_seed{a.seed}').resolve()
    a.amcal_run=(a.amcal_run or runs/f'structured_bank3_policy_lr0p0005_seed{a.seed}').resolve()
    a.baseline_run=(a.baseline_run or runs/f'baselines_levels1_3_rf10_seed{a.seed}').resolve()
    if len({a.source_run,a.amcal_run,a.baseline_run})!=3:p.error('Use distinct source, AMCAL and baseline outputs.')
    required=[a.dataset/'official_partitions.json']
    burst=a.dataset/f'AdvBurst_FTSC-{a.representation}'
    for level in (0,1,3,5,7,10,12,15,17,20):
        stem=f'AdvBurstTrainedBurst_size_{level}_Step_2000'
        required.extend([burst/f'{stem}.csv',burst/f'{stem}_rows.csv'])
    missing=[str(path) for path in required if not path.is_file()]
    if missing:p.error('Prepared official files missing; no download attempted: '+missing[0])
    if a.command=='train':
        for path in (a.amcal_run,a.baseline_run):
            if path.exists():p.error('Training needs a fresh output: '+str(path))
        if a.source_run.exists():
            manifest=a.source_run/'two_stage_training.json'
            models=a.source_run/'Models'
            if not manifest.is_file() or not all((models/name).is_file() for name in
                ('initial_cnn_model.pth','adversarial_best_context.pth','adversarial_best_dqn.pth','protocol_preprocessing.joblib')):
                p.error('Source run is incomplete; choose a fresh --source-run.')
            if set(json.loads(manifest.read_text())['cnn_refinement_domains'])!={0,1,3}:
                p.error('Source run must use training domains 0/1/3.')
    else:
        required=[a.amcal_run/'Models/structured_bank_ddqn.pth']+[
            a.baseline_run/'Models'/f'{method}.joblib' for method in ('cnn','ral','af','dfe','arf')]
        if any(not path.is_file() for path in required):p.error('Train all methods first, or provide their matching run paths.')
    work=commands(a)
    for index,arguments in enumerate(work,1):
        print(f'\n{a.representation} | {a.command} | stage {index}/{len(work)} | {arguments[0]}',flush=True)
        result=subprocess.run([sys.executable,*arguments],cwd=REPOSITORY)
        if result.returncode:raise SystemExit(result.returncode)
    print(f'\nCompleted {a.command}. AMCAL: {a.amcal_run}\nBaselines: {a.baseline_run}',flush=True)


if __name__=='__main__':main()
