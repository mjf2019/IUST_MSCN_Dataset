"""Independent target Gain for existing final AMCAL, epsilon=.05, five seeds.

A joint ~20% TEST observer split is removed before adaptation at every level.
Observer IDs/features are also purged from source replay. Budget is 20% of the
remaining adaptation stream (~16% of total eligible target rows). This separate
evaluation must not replace original full-stream online scores.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import amcal_final_multiseed as run
import amcal_final_multiseed_worker as worker
from burst_table_completion_utils import table_row, aggregate

EPSILON = .05


def predict(torch, exp, context, data):
    prediction = []
    with torch.no_grad():
        for start in range(0,len(data[1]),64):
            prediction.extend(exp.cnn(context(data[0][start:start+64])).argmax(1).cpu().tolist())
    return prediction


def paid_episode(base, exp, learner, domain, reference):
    choose = learner.choose

    def explore(state, can_query, ignored_epsilon, rng):
        return choose(state, can_query, EPSILON, rng)

    learner.choose = explore
    try:
        result, trace = base.run_episode(exp,learner,domain,reference,42,training=False,mode='DDQN')
    finally:
        learner.choose = choose
    for row in trace:
        row['epsilon'] = EPSILON
    result['evaluation_epsilon'] = EPSILON
    return result, trace


def evaluate(output, plan, seed, device, paths, base, meta, torch):
    ds = plan['dataset']; root = output/f'seed{seed}'/'independent_gain_eps0p05_v1'
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

    from enhancements.independent_target_holdout import split_targets, purge_replay, ContextObserver
    streams, audits, split = split_targets(domains,42)
    exp.anchors, purge = purge_replay(exp.anchors,audits)
    execution = dict(version=1,dataset=ds,training_seed=seed,evaluation_seed=42,
        plan_sha256=run.digest(output/'plan.json'),checkpoint_sha256=done['sha256'],
        epsilon=EPSILON,budget_fraction_of_stream=.2,TEST_data=plan['fixed_TEST_data'],
        split=split,replay_purge=purge,model_reset_per_level=True,
        Q_beta='frozen; only temporary Context adapts',
        scope='independent target observers excluded across all levels by flow ID and exact features',
        online='separate reduced-stream evaluation; do not replace original full-stream online scores',
        historical_pretraining='existing weights reused; no claim of removing historical pretraining duplicates',
        evaluator_sha256=run.digest(Path(__file__)),
        helper_sha256=run.digest(run.ROOT/'burst_table_completion_utils.py'),
        split_implementation_sha256=run.digest(run.ROOT/'enhancements'/'independent_target_holdout.py'))
    ef = root/'execution.json'
    if ef.exists() and run.read(ef)!=execution:
        raise ValueError('Independent Gain recipe changed; preserve results.')
    run.write(ef,execution)
    run.write(root/'target_split_manifest.json',split)
    rows = []
    for level in run.LEVELS:
        location = root/f'level_{level}'
        mf, tf, pf, cf = [location/name for name in
            ('metrics.json','online_trace.csv','observer_predictions.csv','completed.json')]
        if cf.exists():
            record = run.read(cf)
            if (record['execution_sha256']!=run.digest(ef) or
                    any(run.digest(f)!=record['sha256'][f.name] for f in (mf,tf,pf))):
                raise ValueError('Independent Gain case changed.')
            result = run.read(mf)
        else:
            stream, audit = streams[level],audits[level]
            if len(audit)<3 or len(reference)<3:
                raise ValueError('Endpoint observers require original flow IDs.')
            base.pilot.base.legacy.core.set_seed(42)
            target_before = predict(torch,exp,exp.template,audit)
            source_before = predict(torch,exp,exp.template,reference)
            observer = ContextObserver(exp)
            result,trace = paid_episode(base,observer,learner,stream,reference)
            if observer.live_context is None:
                raise RuntimeError('Live adapted Context was not captured.')
            target_after = predict(torch,exp,observer.live_context,audit)
            source_after = predict(torch,exp,observer.live_context,reference)
            held_ids = set(map(int,audit[2]))
            if held_ids & {int(r['source_row']) for r in trace}:
                raise RuntimeError('Independent observer entered the adaptation stream.')
            result['adaptation_stream_seen_before'] = result.pop('target_before')
            result['adaptation_stream_seen_after'] = result.pop('target_after')
            result['target_before'] = base.pilot.metrics(audit[1],target_before,len(a.classes))
            result['target_after'] = base.pilot.metrics(audit[1],target_after,len(a.classes))
            result['source_before'] = base.pilot.metrics(reference[1],source_before,len(a.classes))
            result['source_after'] = base.pilot.metrics(reference[1],source_after,len(a.classes))
            result['original_domain_forgetting'] = {
                k:result['source_before'][k]-result['source_after'][k] for k in ('AC','PR','RC','F1')}
            result['target_gain'] = {k:result['target_after'][k]-result['target_before'][k]
                                     for k in ('AC','PR','RC','F1')}
            result.update(target_observer_rows=len(audit[1]),
                original_observer_rows=len(reference[1]),total_target_rows=len(domains[level][1]),
                endpoint_definition='independent fixed rows; before/after on identical IDs',
                labels_percent_of_all_target_rows=100*result['labels']/len(domains[level][1]))
            predictions = []
            for name, data, first, last in [
                    ('target_holdout',audit,target_before,target_after),
                    ('original0_TEST',reference,source_before,source_after)]:
                predictions.extend(dict(domain=name,source_row=int(i),truth=int(y),
                    initial_prediction=int(b),final_prediction=int(f))
                    for i,y,b,f in zip(data[2],data[1],first,last))
            base.pilot.write_json(mf,result)
            base.pilot.write_csv(tf,trace)
            base.pilot.write_csv(pf,predictions)
            run.write(cf,dict(execution_sha256=run.digest(ef),
                sha256={f.name:run.digest(f) for f in (mf,tf,pf)}))
        if (result['samples']!=len(streams[level][1]) or
                result['cap']!=int(.2*len(streams[level][1])) or
                not 0<=result['labels']<=result['cap'] or result['td_updates']!=0):
            raise RuntimeError('Independent stream/budget/frozen-Q contract failed.')
        row = table_row(ds,'AMCAL-independent-eps0.05',seed,level,result['online'],
            result['target_before'],result['target_after'],result['source_before'],
            result['source_after'],result['samples'],result['labels'])
        row.update(target_observer_rows=result['target_observer_rows'],
                   original_observer_rows=result['original_observer_rows'],
                   total_target_rows=result['total_target_rows'],
                   labels_percent_of_all_target_rows=result['labels_percent_of_all_target_rows'])
        rows.append(row)
        print(f'INDEPENDENT GAIN DATASET={ds} TRAIN_SEED={seed} LEVEL={level} EPSILON=.05 '
              f'STREAM_AC={row["AC"]:.2f}% F1={row["F1"]:.2f}% '
              f'TARGET_AC={row["target_initial_AC"]:.2f}->{row["target_final_AC"]:.2f}% '
              f'GAIN_AC={row["target_gain_AC"]:+.2f} GAIN_PR={row["target_gain_PR"]:+.2f} '
              f'GAIN_RC={row["target_gain_RC"]:+.2f} GAIN_F1={row["target_gain_F1"]:+.2f} '
              f'FORGET_AC={row["original_forgetting_AC"]:+.2f} '
              f'LABELS={result["labels"]}/{result["cap"]}',flush=True)
    after = (base.weight_hash(exp.cnn.state_dict()),base.weight_hash(exp.template.state_dict()),
             base.weight_hash(learner.policy.state_dict()),learner.beta_value(),learner.steps,learner.updates)
    if before!=after or run.digest(model)!=done['sha256']:
        raise RuntimeError('Independent evaluation changed initial weights/checkpoint.')
    run.write(root/'summary_by_level.json',rows)
    run.csv_write(root/'summary_by_level.csv',rows)
    run.write(root/'completed.json',dict(seed=seed,plan_sha256=run.digest(output/'plan.json'),
        execution_sha256=run.digest(ef),summary_sha256=run.digest(root/'summary_by_level.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--action',choices=['Evaluate','Collect'],default='Evaluate')
    args = p.parse_args()
    output = run.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'protocol_runs'/'final_multiseed_fixedcnn_v1'
    plan = run.read(output/'plan.json')
    if (plan['dataset']!=args.dataset or plan['evaluation_seed']!=42 or
            sorted(plan['training_seeds'])!=[42,48,54,62,72]):
        p.error('Expected existing final five-seed plan.')
    original,argv = worker.evaluate,sys.argv
    worker.evaluate = evaluate
    try:
        if args.action=='Evaluate':
            for seed in plan['training_seeds']:
                sys.argv = ['amcal_final_multiseed_worker.py','--output-root',str(output),
                    '--training-seed',str(seed),'--stage','evaluate','--device',args.device]
                worker.main()
    finally:
        worker.evaluate,sys.argv = original,argv
    rows = []
    for seed in plan['training_seeds']:
        root = output/f'seed{seed}'/'independent_gain_eps0p05_v1'
        done = run.read(root/'completed.json')
        if (done['plan_sha256']!=run.digest(output/'plan.json') or
                done['execution_sha256']!=run.digest(root/'execution.json') or
                done['summary_sha256']!=run.digest(root/'summary_by_level.json')):
            raise ValueError('Incomplete independent Gain results.')
        rows.extend(run.read(root/'summary_by_level.json'))
    target = output/'Results_independent_gain_eps0p05_v1'
    overall = aggregate(target,rows,plan['training_seeds'],'AMCAL-independent-eps0.05')
    print('ALL SEEDS COMPLETE '+json.dumps(overall,sort_keys=True),flush=True)
    print('OUTPUT: '+str(target),flush=True)


if __name__=='__main__':
    main()
