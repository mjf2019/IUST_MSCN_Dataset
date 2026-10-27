"""No-fit TRAIN/VAL audit of the existing UtilityEarly Actor and its target.

VAL labels are full-information *observer* labels for isolated offline probes.
This is not a deployable VAL policy or a budgeted VAL performance benchmark.
No TEST split is loaded and no checkpoint, normalizer, scale or threshold is fit.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import amcal_budget_actor as budget
from enhancements.actor_diagnostics import ranking_report,print_ranking_report,print_macro_summary
from enhancements.budget_actor_math import scaled_competitive_score

pilot=budget.pilot


def same_weights(before,after):
    return set(before)==set(after) and all(torch.equal(value.cpu(),after[key].cpu()) for key,value in before.items())


class Recorder:
    def __init__(self,scales,level,split):self.scales,self.level,self.split,self.rows=scales,level,split,[]
    def observe(self,state,row):
        if row.get('score') is None:return
        record=dict(row,split=self.split,level=self.level)
        record['ranking_target']=scaled_competitive_score(record,self.scales)
        record['net_utility']=record['weighted_gain']-record['weighted_harm']-record['request_cost']
        self.rows.append(record)


def filter_probe(domain,reference,reserved):
    keys=set(pilot.feature_keys(reference[0]))
    mask=np.asarray([int(source_id) not in reserved and key not in keys
        for source_id,key in zip(domain[2],pilot.feature_keys(domain[0]))])
    return pilot.subset(domain,mask)


def collect_split(a,healthy,artifact,anchors,reference,reserved,saved,split,root):
    split_ids={'TRAIN':'train_ids','VAL':'val_ids'}[split]
    directory=root/split;directory.mkdir(exist_ok=True)
    reports=[];csv_rows=[]
    for level in (0,1,3):
        raw=pilot.base.legacy.read_domain(a.burst_dir,artifact,level,split_ids)
        domain=budget.pilot_domain(a,filter_probe(raw,reference,reserved),702 if split=='TRAIN' else 703)
        recorder=Recorder(saved['signal_scales'],level,split)
        pilot.base.legacy.core.set_seed(a.seed)
        live=budget.runtime(a,healthy,artifact,saved['query_state_dict'])
        before_query=deepcopy(live[3].state_dict());before_router=deepcopy(live[2].policy.state_dict())
        print(f'\nOFFLINE {split} level {level}: {len(domain[1])} rows | temporary uniform-early updates | no Actor fitting',flush=True)
        # Training=True enables counterfactual observer probes, not fitting:
        # collect_query_targets_only skips Actor/value optimization, and
        # budget_actor_mode prevents router updates. All live states are copies.
        updated,result,_=pilot.stream(a,live,domain,anchors,reference,domain,a.seed+level*101,
            training=True,diagnostic_sink=recorder)
        if not same_weights(before_query,updated[3].state_dict()) or not same_weights(before_router,updated[2].policy.state_dict()):
            raise RuntimeError('A diagnostic changed the Actor or routing/future heads.')
        if result['labels']!=result['cap']:raise RuntimeError('Uniform trajectory did not consume its cap.')
        report=ranking_report(recorder.rows,min(result['cap'],len(recorder.rows)),a.seed+level)
        report.update(split=split,level=level,trajectory_samples=result['samples'],trajectory_labels=result['labels'],
            model_weights_unchanged=True,observer_labels_not_counted_as_deployment_budget=True)
        reports.append(report);print_ranking_report(split,level,report)
        pilot.write_csv(directory/f'level_{level}_targets.csv',recorder.rows)
        pilot.write_json(directory/f'level_{level}_report.json',report)
        for selector,values in report['selectors'].items():
            csv_rows.append(dict(split=split,level=level,selector=selector,candidates=report['candidates'],
                top_k=report['top_k'],**values,
                actor_target_spearman=report['correlations']['actor_target']['spearman'],
                target_utility_spearman=report['correlations']['target_utility']['spearman'],
                actor_utility_spearman=report['correlations']['actor_utility']['spearman']))
    pilot.write_csv(directory/'summary.csv',csv_rows)
    # Macro-average the per-level comparisons, rather than mixing domains and
    # selecting all top-k candidates from the easiest level.
    mean_selectors={name:{key:float(np.mean([r['selectors'][name][key] for r in reports]))
        for key in reports[0]['selectors'][name]} for name in reports[0]['selectors']}
    summary=dict(levels=[0,1,3],selectors=mean_selectors,
        mean_correlations={key:{kind:None if any(r['correlations'][key][kind] is None for r in reports)
            else float(np.mean([r['correlations'][key][kind] for r in reports])) for kind in ('pearson','spearman')}
            for key in reports[0]['correlations']},reports=reports)
    print_macro_summary(split,summary)
    pilot.write_json(directory/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--budget-percent',type=int,choices=[5,10],default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pilot-rows',type=int,default=300)
    p.add_argument('--epochs',type=int,default=2,help='Existing Actor run identifier; no epochs are run.')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--run',type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parent/f'ANT_Burst_{args.dataset}_TEST'
    run=args.run or root/'protocol_runs'/f'budget_actor_q{args.budget_percent:02d}_seed{args.seed}_pilot{args.pilot_rows}_ep{args.epochs}_scaled_early20'
    checkpoint=run/'Models'/'budget_actor.pth';actor_hash=pilot.digest(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    complete=json.loads((run/'training_complete.json').read_text(encoding='utf-8'))
    spec=saved['protocol']
    if complete['checkpoint_sha256']!=actor_hash or complete['protocol']!=spec:
        raise ValueError('Actor checkpoint/completion mismatch.')
    if (spec['dataset']!=args.dataset or spec['seed']!=args.seed or
        spec.get('query_window_fraction')!=.2 or not saved.get('signal_scales')):
        raise ValueError('Use a matching UtilityEarly checkpoint with frozen TRAIN scales.')
    if spec['pilot_rows']!=args.pilot_rows or spec['budget_fraction']!=args.budget_percent/100 or spec['epochs']!=args.epochs:
        raise ValueError('Run identifiers differ from the saved checkpoint.')
    a=SimpleNamespace(**{key:spec[key] for key in
        ('dataset','seed','budget_fraction','pilot_rows','epochs','steps_per_epoch','batch_size','actor_lr')})
    a.device=args.device;a.output=run;a.utility_early=True;a.query_window_fraction=.2;a.progress_every=250
    a.teacher=root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    healthy,artifact,anchors,train_reference,teacher_hash=budget.setup(a)
    if spec!=budget.protocol(a,teacher_hash):raise ValueError('Saved Actor protocol or teacher changed.')
    a.actor_experiment=dict(version=1,control='natural',normalizer=saved['normalizer'])
    a.collect_query_targets_only=True
    print(f"Loaded {a.dataset} Actor: selected epoch {saved.get('selected_epoch')} of {a.epochs}; "
        f"{a.pilot_rows} rows/domain; existing {100*a.budget_fraction:.0f}% budget; no fitting",flush=True)
    print('Probe utility is future loss reduction minus old-domain loss harm and cost, not final accuracy.',flush=True)
    output=run/'actor_diagnostics'
    execution=dict(version=1,actor_checkpoint_sha256=actor_hash,teacher_checkpoint_sha256=teacher_hash,
        protocol=spec,scales=saved['signal_scales'],splits=['TRAIN','VAL'],levels=[0,1,3],
        trajectory='uniform early feedback, same temporary trajectory for every ranking selector',
        interpretation='offline full-information ranking audit; oracles are not deployable; no policy fitting or TEST access',
        utility='weighted_gain - weighted_harm - request_cost; one update, future block without further adaptation')
    if (output/'completed.json').exists():
        completion=json.loads((output/'completed.json').read_text(encoding='utf-8'))
        if json.loads((output/'execution.json').read_text(encoding='utf-8'))!=execution:
            raise ValueError('Diagnostic settings differ from the completed audit.')
        for name,expected in completion['sha256'].items():
            if pilot.digest(output/name)!=expected:raise ValueError('Completed diagnostic file changed: '+name)
        print('Using completed diagnostic. See TRAIN/summary.json and VAL/summary.json.',flush=True)
        display=json.loads((output/'comparison.json').read_text(encoding='utf-8'))
        for split,summary in display.items():print_macro_summary(split,summary)
        return
    output.mkdir(exist_ok=True)
    if (output/'execution.json').exists() and json.loads((output/'execution.json').read_text(encoding='utf-8'))!=execution:
        raise ValueError('Partial diagnostic settings differ; output preserved.')
    pilot.write_json(output/'execution.json',execution)
    train=collect_split(a,healthy,artifact,anchors,train_reference,
        set(healthy['provenance']['reference_source_ids']),saved,'TRAIN',output)
    val_source=pilot.base.legacy.read_domain(a.burst_dir,artifact,0,'val_ids')
    _,val_reference,val_reserved=pilot.reference_partition(val_source,a.seed)
    val=collect_split(a,healthy,artifact,anchors,val_reference,val_reserved,saved,'VAL',output)
    comparison={split:dict(mean_correlations=report['mean_correlations'],selectors=report['selectors'])
        for split,report in [('TRAIN',train),('VAL',val)]}
    pilot.write_json(output/'comparison.json',comparison)
    print('\nREADOUT: Actor~target measures target learning; target~utility audits the target itself.',flush=True)
    print('Compare TRAIN versus VAL. Correlation alone does not prove useful online adaptation.',flush=True)
    print('OracleUtility is an observer-only upper bound for this one-update probe, not final stream accuracy.',flush=True)
    if pilot.digest(checkpoint)!=actor_hash or pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=teacher_hash:
        raise RuntimeError('A saved checkpoint changed during diagnostics.')
    files=[path for path in output.rglob('*') if path.is_file() and path.suffix in ('.json','.csv')
        and path.name!='completed.json']
    pilot.write_json(output/'completed.json',dict(sha256={str(path.relative_to(output)).replace('\\','/'):pilot.digest(path) for path in files}))
    print(f'Diagnostic results: {output}',flush=True)


if __name__=='__main__':main()
