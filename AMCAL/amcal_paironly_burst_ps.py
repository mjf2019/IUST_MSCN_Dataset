"""Run the fixed PairOnly recipe on PS-only source models and source-row splits."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import amcal_final_enhancement as runner

TRAIN_LEVELS = [0, 1, 3]
TEST_LEVELS = [5, 7, 10, 12, 15, 17, 20]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(a):
    base = a.output_root/'accuracy'
    utility = a.output_root/'counterfactual'
    final = a.output_root/'paironly'
    common = ['--burst-dir',str(a.burst_dir),'--profile','accuracy','--seed',str(a.seed),
        '--device',a.device,'--context-lr','0.0005','--context-alpha','0.8','--gamma','0.95',
        '--budget-fraction','0.2','--eval-epsilon','0.05','--update-policy','learned']
    stages = [
        dict(name='accuracy',source=str(a.source_run),output=str(base),args=[
            'train','--source-run',str(a.source_run),'--output',str(base),*common,
            '--warm-start-selector','--query-schedule','policy','--levels','0','1','3',
            '--epochs',str(a.base_epochs),'--patience',str(a.patience),
            '--warmup-episodes',str(a.base_warmup_episodes)]),
        dict(name='counterfactual',source=str(base),output=str(utility),args=[
            'train','--source-run',str(base),'--output',str(utility),*common,
            '--warm-start-enhancement','--aligned-sequence','--counterfactual-utility',
            '--query-schedule','uniform','--levels','0','1','3',
            '--epochs',str(a.utility_epochs),'--patience',str(a.utility_epochs)]),
        dict(name='paironly',source=str(utility),output=str(final),args=[
            'train','--source-run',str(utility),'--output',str(final),*common,
            '--warm-start-enhancement','--aligned-sequence','--one-context-rl-training',
            '--query-value-only-training','--paired-query-only-training','--context-count','1',
            '--query-schedule','policy','--levels','0','1','3',
            '--epochs',str(a.pair_epochs),'--patience',str(a.patience)])]
    evaluation = ['evaluate','--output',str(final),'--burst-dir',str(a.burst_dir),
        '--context-count','1','--query-schedule','policy','--update-policy','learned',
        '--modes','learned','--levels',*map(str,TEST_LEVELS),'--seed',str(a.seed),
        '--device',a.device,'--eval-epsilon','0.05','--budget-fraction','0.2',
        '--context-lr','0.0005','--routing-oracle-audit','--skip-existing-completed']
    if a.evaluation_tag: evaluation += ['--evaluation-tag',a.evaluation_tag]
    return stages,evaluation


def preflight(a):
    models=a.source_run/'Models'
    required=[*runner.SOURCE_FILES,'structured_bank_ddqn.pth']
    missing=[str(models/name) for name in required if not (models/name).is_file()]
    if missing: raise ValueError('PS source Models missing: '+', '.join(missing))
    runner.legacy.core.device=torch.device('cpu')  # Validation only; each stage selects its requested device.
    artifact=runner.load_artifact(models,a.burst_dir)  # Reject IAT preprocessing/PS data mismatch.
    rows={}
    for level in TRAIN_LEVELS+TEST_LEVELS:
        splits=('train_ids','val_ids') if level in TRAIN_LEVELS else ('test_ids',)
        rows[str(level)]={}
        for split in splits:
            _,labels,_=runner.legacy.read_domain(a.burst_dir,artifact,level,split)
            rows[str(level)][split]=len(labels)
    source_hashes={name:digest(models/name) for name in required}
    domains={str(level):digest(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
             for level in TRAIN_LEVELS+TEST_LEVELS}
    return dict(source_model_hashes=source_hashes,domain_sha256=domains,partition_rows=rows,
        source_partition_sha256=digest(models/'protocol_preprocessing.joblib'))


def stage_inputs(stage):
    models=Path(stage['source'])/'Models'
    names=[*runner.SOURCE_FILES,'structured_bank_ddqn.pth' if stage['name']=='accuracy' else 'final_enhancement.pth']
    return dict(args=stage['args'],input_hashes={name:digest(models/name) for name in names})


def execute_stage(stage):
    output=Path(stage['output']); marker=output/'pipeline_stage_complete.json'
    expected=stage_inputs(stage)
    checkpoint=output/'Models'/'final_enhancement.pth'
    log=output/'enhancement_training.json'
    if output.exists() and any(output.iterdir()):
        if marker.is_file() and checkpoint.is_file() and log.is_file():
            recorded=json.loads(marker.read_text())
            if (recorded.get('inputs')==expected and recorded.get('checkpoint_sha256')==digest(checkpoint)
                    and recorded.get('training_log_sha256')==digest(log)):
                print('Skipping completed PS training stage:',stage['name'],flush=True)
                return
        raise ValueError(f'Incomplete/mismatched training stage preserved: {output}. Use a fresh --output-root.')
    print('PS training stage:',stage['name'],flush=True)
    runner.main(stage['args'])
    if not checkpoint.is_file() or not log.is_file():raise RuntimeError('Training did not write checkpoint and log.')
    runner.write_json(marker,dict(inputs=expected,checkpoint_sha256=digest(checkpoint),training_log_sha256=digest(log)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['All','Train','Evaluate','Audit'],default='All')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    p.add_argument('--source-run',type=Path,default=Path('AMCAL/ANT_Burst_PS_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42'))
    p.add_argument('--burst-dir',type=Path,default=Path('AMCAL/ANT_Burst_PS_TEST/Dataset_official_v1/AdvBurst_FTSC-PS'))
    p.add_argument('--output-root',type=Path)
    p.add_argument('--base-epochs',type=int,default=12)
    p.add_argument('--base-warmup-episodes',type=int,default=3)
    p.add_argument('--utility-epochs',type=int,default=3)
    p.add_argument('--pair-epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--evaluation-tag',default='')
    a=p.parse_args(argv)
    if a.output_root is None:a.output_root=Path(f'AMCAL/ANT_Burst_PS_TEST/protocol_runs/paironly_pipeline_seed{a.seed}')
    if min(a.base_epochs,a.utility_epochs,a.pair_epochs,a.patience)<1 or a.base_warmup_episodes<0:p.error('Invalid training epochs/patience.')
    provenance=preflight(a)
    stages,evaluation=make_plan(a)
    # Action and repeat tag do not alter the fixed training recipe.
    plan=dict(version=1,representation='Burst_PS',seed=a.seed,train_levels=TRAIN_LEVELS,
        test_levels=TEST_LEVELS,source_run=str(a.source_run),burst_dir=str(a.burst_dir),
        stages=stages,provenance=provenance,
        method='PairOnly one Context; full online RL; no protected routing replay',
        evaluation='independent and persistent sequential; observer-only routing oracle')
    plan_path=a.output_root/'pipeline_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Saved PS pipeline recipe/provenance differs; use a fresh output root.')
    else:
        if a.output_root.exists() and any(a.output_root.iterdir()):raise ValueError('Nonempty output root without pipeline provenance.')
        if a.action not in ('All','Train'):raise ValueError('PS training has not run; use --action All or Train first.')
        runner.write_json(plan_path,plan)
    if a.action in ('All','Train'):
        for stage in stages:execute_stage(stage)
    else:
        # Evaluation-only still validates the complete chain without starting training.
        for stage in stages:
            out=Path(stage['output']);marker=out/'pipeline_stage_complete.json'
            if not marker.is_file():raise ValueError('PS training incomplete: '+str(out))
            recorded=json.loads(marker.read_text())
            if (recorded.get('inputs')!=stage_inputs(stage)
                    or recorded.get('checkpoint_sha256')!=digest(out/'Models'/'final_enhancement.pth')
                    or recorded.get('training_log_sha256')!=digest(out/'enhancement_training.json')):
                raise ValueError('PS stage provenance changed: '+str(out))
    if a.action in ('All','Evaluate'):runner.main(evaluation)
    if a.action in ('All','Audit'):runner.main([*evaluation,'--continual'])
    print('PS PairOnly outputs:',a.output_root/'paironly',flush=True)


if __name__=='__main__':main()
