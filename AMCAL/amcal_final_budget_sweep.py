"""Full-TEST inference-budget sensitivity of FINAL AMCAL; no offline retraining.

Five existing training seeds, fixed evaluation seed42, epsilon=.05. Context,
Adam and feedback memory reset for every level and budget. No model snapshots
are created. Verified existing epsilon=.05 budget20 cases are reused.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import amcal_final_multiseed as run
import amcal_final_multiseed_worker as worker

EPSILON = .05
DEFAULT_BUDGETS = [10,15,20,25,30,35,40,45,50]
BUDGETS = DEFAULT_BUDGETS
SCORES = ('AC','PR','RC','F1')
FIELDS = ['AC','PR','RC','F1','labels_percent','budget_used_percent',
          'milliseconds_per_row'] + [
    p+k for p in ('original_initial_','original_final_','original_forgetting_',
                  'target_seen_initial_','target_seen_final_') for k in SCORES]


def validate_budgets(values):
    values = [float(v) for v in values]
    if (not values or len(set(values))!=len(values) or
            any(not math.isfinite(v) or not 0<v<=100 for v in values)):
        raise ValueError('Use distinct finite labeling caps in (0,100].')
    return sorted(values)


def tag(value):
    return format(value,'.8g').replace('.','p')


def make_row(dataset,seed,level,budget,beta,result):
    n = result['samples']; cap = int((budget/100)*n)
    if (n!=(968 if dataset=='IAT' else 801) or result['cap']!=cap or
            not 0<=result['labels']<=cap or result['td_updates']!=0):
        raise ValueError('Full TEST/budget/frozen-Q contract violated.')
    row = dict(dataset=dataset,training_seed=seed,evaluation_seed=42,level=level,
        budget_percent=budget,epsilon=EPSILON,beta=beta,**result['online'],
        samples=n,labels=result['labels'],cap=cap,
        labels_percent=100*result['labels']/n,
        budget_used_percent=100*result['labels']/cap if cap else 0.,
        milliseconds_per_row=result['milliseconds_per_row'])
    for metric in SCORES:
        row['original_initial_'+metric] = result['source_before'][metric]
        row['original_final_'+metric] = result['source_after'][metric]
        row['original_forgetting_'+metric] = result['source_before'][metric]-result['source_after'][metric]
        row['target_seen_initial_'+metric] = result['target_before'][metric]
        row['target_seen_final_'+metric] = result['target_after'][metric]
    return row


def reuse_twenty(paths,plan,checkpoint_sha,level,seed):
    root = paths['evaluation'].parent/'evaluation_eps0p05_v1'
    ef = root/'execution.json'; location = root/f'level_{level}'
    mf,tf,cf = [location/f for f in ('metrics.json','trace.csv','completed.json')]
    if not all(f.is_file() for f in (ef,mf,tf,cf)):
        return None
    ex = run.read(ef); done = run.read(cf)
    if (ex.get('evaluation_epsilon')!=EPSILON or ex.get('budget_fraction')!=.2 or
            ex.get('evaluation_seed')!=42 or ex.get('training_seed')!=seed or
            ex.get('dataset')!=plan['dataset'] or not ex.get('reset_per_level') or
            ex.get('TEST_data')!=plan['fixed_TEST_data'] or
            ex.get('learned_checkpoint_sha256')!=checkpoint_sha):
        raise ValueError('Existing budget20 is present but does not match the final recipe.')
    if (done['execution_sha256']!=run.digest(ef) or
            any(run.digest(f)!=done['sha256'][f.name] for f in (mf,tf))):
        raise ValueError('Existing budget20 files changed.')
    return mf,tf,run.read(mf),dict(source_execution=str(ef),
                                  source_execution_sha256=run.digest(ef))


def run_case(base,exp,learner,domain,reference):
    choose = learner.choose
    def explore(state,can_query,ignored_epsilon,rng):
        return choose(state,can_query,EPSILON,rng)
    learner.choose = explore
    try:
        result,trace = base.run_episode(exp,learner,domain,reference,42,training=False,mode='DDQN')
    finally:
        learner.choose = choose
    for row in trace:
        row['epsilon'] = EPSILON
        row['evaluation_policy'] = 'DDQN_EPSILON_GREEDY'
    result['evaluation_epsilon'] = EPSILON
    return result,trace


def evaluate(output, plan, seed, device, paths, base, meta, torch):
    ds = plan['dataset']; root = output/f'seed{seed}'/'final_budget_sweep_eps0p05_v1'
    model = paths['meta']/'Models'/'learned_beta.pth'
    done = run.read(paths['meta']/'training_complete.json')
    saved = torch.load(model, map_location='cpu', weights_only=False)
    if (done['protocol'] != saved['protocol'] or done['sha256'] != run.digest(model)
            or saved['protocol']['seed'] != seed):
        raise ValueError('Independent learned-beta checkpoint is incomplete or changed.')
    burst = run.ROOT/f'ANT_Burst_{ds}_TEST'/'Dataset_official_v1'/f'AdvBurst_FTSC-{ds}'
    a = SimpleNamespace(dataset=ds, seed=seed, device=device,
        pilot_rows=plan['train_rows_per_level'], beta=float(saved['beta']), budget_fraction=.2,
        burst_dir=burst, teacher=paths['context'], actor_run=paths['selector'], output=root)
    exp = base.PaperGainExperiment(a)
    sp = saved['protocol']['source_protocol']['protocol']
    if (exp.teacher_hash != sp['healthy_teacher_sha256'] or exp.actor_sha != sp['actor_checkpoint_sha256']
            or exp.healthy['model_hashes'] != sp['source_models']):
        raise ValueError('Per-seed Context/selector lineage differs.')
    # A separate, data-only setup fixes TEST IDs/observer/replay/order at seed42.
    # Its trained scores/policy never select or initialize this seed's prediction.
    runs = Path(plan['reference']).parent
    fixed = SimpleNamespace(dataset=ds, seed=42, device=device,
        pilot_rows=plan['train_rows_per_level'], beta=0., budget_fraction=.2, burst_dir=burst,
        teacher=runs/'minimax_query_batch5_seed42_v2_h16_c0.05_actorstd_natural',
        actor_run=runs/'competitive_selector_seed42_p300_r2_v1_paper_reward',
        output=root)
    canonical = base.PaperGainExperiment(fixed)
    canonical.a.pilot_rows = 0
    domains, reference = canonical.split('test_ids', base.TEST)
    if canonical.data_manifest('test_ids', base.TEST) != plan['fixed_TEST_data']:
        raise ValueError('Fixed full TEST stream/reference differs from selected final run.')
    exp.anchors = canonical.anchors
    del canonical
    a.pilot_rows = 0; a.seed = plan['evaluation_seed']
    base.pilot.base.legacy.core.set_seed(a.seed)
    source = torch.load(paths['warmstart']/'Models'/'two_action_ddqn.pth',
                        map_location='cpu', weights_only=False)
    learner = meta.learner_from_source(exp, source)
    learner.policy.load_state_dict(saved['policy_state'])
    with torch.no_grad():
        learner.log_beta.copy_(learner.log_beta.new_tensor(math.log(float(saved['beta']))))
    learner.policy.eval().requires_grad_(False); learner.log_beta.requires_grad_(False)
    before = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
              base.weight_hash(learner.policy.state_dict()), learner.beta_value(), learner.steps, learner.updates)

    for budget in BUDGETS:
        a.budget_fraction = budget/100
        location = root/f'budget{tag(budget)}'
        execution = dict(version=1,dataset=ds,training_seed=seed,evaluation_seed=42,
            plan_sha256=run.digest(output/'plan.json'),checkpoint_sha256=done['sha256'],
            budget_percent=budget,budget_fraction=budget/100,epsilon=EPSILON,
            beta=float(saved['beta']),TEST_data=plan['fixed_TEST_data'],
            levels=run.LEVELS,test_rows=0,reset_per_budget_and_level=True,
            prediction='Context -> frozen CNN; pre-feedback online scores',
            learning='no offline retraining or online DDQN/beta updates; Context alone adapts',
            original_forgetting='fixed original-domain TEST observer before minus after; percentage points',
            target_endpoints='seen adaptation-stream diagnostics, not independent Gain or sequential BWT',
            source_replay='same canonical seed42 TRAIN pool as final budget20',
            evaluator_sha256=run.digest(Path(__file__)))
        ef = location/'execution.json'
        if ef.exists() and run.read(ef)!=execution:
            raise ValueError('Budget-sweep recipe changed; preserve existing output.')
        run.write(ef,execution)
        rows = []
        for level in run.LEVELS:
            case = location/f'level_{level}'
            mf,tf,cf = [case/f for f in ('metrics.json','trace.csv','completed.json')]
            if cf.exists():
                record = run.read(cf)
                if (record['execution_sha256']!=run.digest(ef) or
                        any(run.digest(f)!=record['sha256'][f.name] for f in (mf,tf))):
                    raise ValueError('Completed budget case changed.')
                result = run.read(mf)
                reused = bool(record.get('reused_budget20'))
                status = 'RESUME'
            else:
                case.mkdir(parents=True,exist_ok=True)
                existing = reuse_twenty(paths,plan,done['sha256'],level,seed) if budget==20 else None
                if existing:
                    old_metric,old_trace,result,provenance = existing
                    shutil.copy2(old_metric,mf); shutil.copy2(old_trace,tf)
                    reused = True; status = 'REUSE_BUDGET20'
                else:
                    print(f'START DATASET={ds} TRAIN_SEED={seed} BUDGET={budget:g}% '
                          f'LEVEL={level} EPSILON=.05 FULL_TEST={len(domains[level][1])}',flush=True)
                    base.pilot.base.legacy.core.set_seed(42)
                    result,trace = run_case(base,exp,learner,domains[level],reference)
                    base.pilot.write_json(mf,result); base.pilot.write_csv(tf,trace)
                    provenance = None; reused = False; status = 'EVALUATED'
                # Validate results before marking the case complete.
                make_row(ds,seed,level,budget,float(saved['beta']),result)
                run.write(cf,dict(execution_sha256=run.digest(ef),
                    reused_budget20=reused,reuse_provenance=provenance,
                    sha256={f.name:run.digest(f) for f in (mf,tf)}))
            row = make_row(ds,seed,level,budget,float(saved['beta']),result)
            rows.append(row)
            print(f'{status} DATASET={ds} TRAIN_SEED={seed} BUDGET={budget:g}% LEVEL={level} '
                  f'AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}% '
                  f'LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
                  f'CAP_USED={row["budget_used_percent"]:.2f}% '
                  f'ORIGINAL_FORGET_AC={row["original_forgetting_AC"]:+.2f} pp',flush=True)
        after = (base.weight_hash(exp.cnn.state_dict()),base.weight_hash(exp.template.state_dict()),
                 base.weight_hash(learner.policy.state_dict()),learner.beta_value(),learner.steps,learner.updates)
        if before!=after or run.digest(model)!=done['sha256']:
            raise RuntimeError('Budget evaluation modified an initial/frozen checkpoint.')
        run.write(location/'summary_by_level.json',rows)
        run.csv_write(location/'summary_by_level.csv',rows)
        mean = {k:sum(r[k] for r in rows)/len(rows) for k in FIELDS}
        run.write(location/'completed.json',dict(training_seed=seed,budget_percent=budget,
            plan_sha256=run.digest(output/'plan.json'),execution_sha256=run.digest(ef),
            summary_sha256=run.digest(location/'summary_by_level.json'),mean=mean))
        print(f'BUDGET COMPLETE DATASET={ds} TRAIN_SEED={seed} BUDGET={budget:g}% '
              f'AC={mean["AC"]:.2f}% F1={mean["F1"]:.2f}% '
              f'LABELS={mean["labels_percent"]:.2f}% '
              f'ORIGINAL_FORGET_AC={mean["original_forgetting_AC"]:+.2f} pp',flush=True)


def summarize_rows(rows,seeds,budgets,dataset):
    expected = {(s,b,l) for s in seeds for b in budgets for l in run.LEVELS}
    actual = [(r['training_seed'],r['budget_percent'],r['level']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Missing or duplicate seed/budget/level cases.')
    per_seed,by_level,overall = [],[],[]
    for budget in budgets:
        for seed in seeds:
            selected = [r for r in rows if r['budget_percent']==budget and r['training_seed']==seed]
            row = dict(dataset=dataset,training_seed=seed,evaluation_seed=42,budget_percent=budget,
                       epsilon=EPSILON,levels=len(selected))
            row.update({k:sum(r[k] for r in selected)/len(selected) for k in FIELDS})
            per_seed.append(row)
        selected = [r for r in per_seed if r['budget_percent']==budget]
        row = dict(dataset=dataset,budget_percent=budget,epsilon=EPSILON,seeds=len(seeds),levels=7)
        for k in FIELDS:
            values = [r[k] for r in selected]
            row[k+'_mean'] = sum(values)/len(values)
            row[k+'_sd'] = run.moments(values)[1] if len(values)>1 else None
        overall.append(row)
        for level in run.LEVELS:
            selected = [r for r in rows if r['budget_percent']==budget and r['level']==level]
            row = dict(dataset=dataset,budget_percent=budget,level=level,seeds=len(seeds),epsilon=EPSILON)
            for k in FIELDS:
                values = [r[k] for r in selected]
                row[k+'_mean'] = sum(values)/len(values)
                row[k+'_sd'] = run.moments(values)[1] if len(values)>1 else None
            by_level.append(row)
    return per_seed,by_level,overall


def collect(output,plan,seeds,budgets):
    rows=[]
    for seed in seeds:
        for budget in budgets:
            root=output/f'seed{seed}'/'final_budget_sweep_eps0p05_v1'/f'budget{tag(budget)}'
            done=run.read(root/'completed.json'); execution=run.read(root/'execution.json')
            if (done['plan_sha256']!=run.digest(output/'plan.json') or
                    done['execution_sha256']!=run.digest(root/'execution.json') or
                    done['summary_sha256']!=run.digest(root/'summary_by_level.json') or
                    execution['evaluator_sha256']!=run.digest(Path(__file__)) or
                    execution['training_seed']!=seed or execution['budget_percent']!=budget):
                raise ValueError('Incomplete or changed budget-sweep results.')
            rows.extend(run.read(root/'summary_by_level.json'))
    per_seed,by_level,overall = summarize_rows(rows,seeds,budgets,plan['dataset'])
    target=output/'Results_final_budget_sweep_eps0p05_v1'/(
        'seeds_'+'_'.join(map(str,seeds)))/('budgets_'+'_'.join(tag(b) for b in budgets))
    run.csv_write(target/'all_seed_budget_levels.csv',rows)
    run.csv_write(target/'summary_per_seed_budget.csv',per_seed)
    run.csv_write(target/'summary_by_budget_level.csv',by_level)
    run.csv_write(target/'summary_by_budget.csv',overall)
    run.write(target/'completed.json',dict(dataset=plan['dataset'],seeds=seeds,budgets=budgets,
        plan_sha256=run.digest(output/'plan.json'),evaluator_sha256=run.digest(Path(__file__)),
        cases=len(rows),definition='SD across per-training-seed seven-level means; sample SD',
        target_endpoints='seen-stream diagnostics; original forgetting is per independently reset level',
        summary_sha256=run.digest(target/'summary_by_budget.csv')))
    for row in overall:
        print(f'MEAN DATASET={plan["dataset"]} BUDGET={row["budget_percent"]:g}% '
              f'AC={row["AC_mean"]:.2f} +/- {row["AC_sd"] if row["AC_sd"] is not None else 0.:.2f}% '
              f'F1={row["F1_mean"]:.2f} +/- {row["F1_sd"] if row["F1_sd"] is not None else 0.:.2f}% '
              f'LABELS={row["labels_percent_mean"]:.2f}% '
              f'ORIGINAL_FORGET_AC={row["original_forgetting_AC_mean"]:+.2f} pp',flush=True)
    print('OUTPUT: '+str(target),flush=True)


def main():
    global BUDGETS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--budgets',type=float,nargs='+',default=DEFAULT_BUDGETS)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,48,54,62,72])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--action',choices=['Evaluate','Collect'],default='Evaluate')
    args=p.parse_args()
    try: BUDGETS=validate_budgets(args.budgets)
    except ValueError as error: p.error(str(error))
    output=run.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'protocol_runs'/'final_multiseed_fixedcnn_v1'
    plan=run.read(output/'plan.json')
    if (plan['dataset']!=args.dataset or plan['evaluation_seed']!=42 or plan['test_rows']!=0 or
            not args.seeds or len(set(args.seeds))!=len(args.seeds) or
            any(s not in plan['training_seeds'] for s in args.seeds)):
        p.error('Use existing final training seeds and the full fixed TEST plan.')
    original,argv=worker.evaluate,sys.argv
    worker.evaluate=evaluate
    try:
        if args.action=='Evaluate':
            for seed in args.seeds:
                print(f'=== FINAL BUDGET SWEEP DATASET={args.dataset} TRAIN_SEED={seed} '
                      f'BUDGETS={BUDGETS} EPSILON=.05 FULL TEST NO RETRAINING ===',flush=True)
                sys.argv=['amcal_final_multiseed_worker.py','--output-root',str(output),
                          '--training-seed',str(seed),'--stage','evaluate','--device',args.device]
                worker.main()
    finally:
        worker.evaluate,sys.argv=original,argv
    collect(output,plan,args.seeds,BUDGETS)


if __name__=='__main__':
    main()
