"""Sequential retention of FINAL two-action AMCAL, without offline retraining.

Context, Adam, protected replay, acquired-feedback history and policy RNG persist
across target levels. Only the per-level annotation cap/progress reset. DDQN,
beta and CNN remain fixed. Joint group-disjoint target observers never adapt.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import amcal_final_multiseed as run
import amcal_final_multiseed_worker as worker
from final_sequential_metrics import LEVELS, METRICS, summarize, support_scale

EPSILON = .05


def observe(base, torch, exp, context, targets, source, stage, adapted_level):
    rows, predictions = [], []
    with torch.no_grad():
        for level,data in [(0,source),*sorted(targets.items())]:
            predicted = []
            for start in range(0,len(data[1]),64):
                predicted.extend(exp.cnn(context(data[0][start:start+64])).argmax(1).cpu().tolist())
            rows.append(dict(stage=stage,adapted_level=adapted_level,domain=level,
                samples=len(data[1]),**base.pilot.metrics(data[1],predicted,len(exp.a.classes))))
            predictions.extend(dict(stage=stage,domain=level,source_row=int(i),
                truth=int(y),prediction=int(p)) for i,y,p in zip(data[2],data[1],predicted))
    return rows,predictions


def sequential_stage(base, torch, exp, learner, session, domain, level, blocked_ids, blocked_keys):
    np = base.np
    a = exp.a
    order = np.random.default_rng(42).permutation(len(domain[1]))
    x = domain[0][torch.as_tensor(order,device=domain[0].device)]
    y,ids = np.asarray(domain[1])[order],np.asarray(domain[2])[order]
    keys = base.pilot.feature_keys(x)
    if blocked_ids.intersection(map(int,ids)) or blocked_keys.intersection(keys):
        raise RuntimeError('An observer ID/feature entered adaptation.')
    n = len(y); cap = int(.2*n); used = 0
    if n<1: raise ValueError('Empty sequential adaptation domain.')
    context,optimizer,memory = session.context,session.optimizer,session.memory
    context.eval()
    offset = session.clock
    predictions,trace = [],[]
    if x.is_cuda: torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    frozen,embeddings = base.pilot.base.legacy.base_cache(exp.cnn,x)
    executor = base.BankExecutor(exp.cnn,[context])
    for position in range(n):
        inputs = x[position:position+1]
        with torch.no_grad():
            outputs = executor.outputs(inputs,frozen[position:position+1])
            state = base.query_features(outputs,embeddings[position:position+1],
                session.paid_embeddings,session.paid_labels,used,cap,position,n,
                session.last_update-offset).detach().cpu().reshape(-1)
            # History is cumulative but annotation budget is per-domain.
            # Normalize the two class-support entries by cumulative paid labels.
            state[-5:-3] *= support_scale(used,len(session.paid_labels))
            action,q = learner.choose(state,used<cap,EPSILON,session.query_rng)
            prediction = int(outputs[0][0,1].argmax())
            frozen_prediction = int(outputs[0][0,0].argmax())
            entropy = float(base.pilot.normalized_entropy(outputs[0][0,1].cpu().numpy()))
        if action not in (0,1) or (action==1 and used>=cap):
            raise RuntimeError('Illegal sequential feedback action.')
        truth = int(y[position])
        if action==1:
            label = torch.tensor([truth],device=x.device)
            paid,replay = memory.draw(ids[position],inputs,reference_keys=blocked_keys,
                                     current_key=keys[position])
            base.pilot.context_min_step(context,optimizer,exp.cnn,(inputs,label),
                                       paid,replay,a.replay_weight)
            memory.add(ids[position],inputs,truth)
            session.paid_embeddings.append(embeddings[position:position+1].detach().cpu().clone())
            session.paid_labels.append(truth)
            used += 1
            session.last_update = offset+position+1
        predictions.append(prediction)
        trace.append(dict(level=level,step=position+1,global_step=offset+position+1,
            source_row=int(ids[position]),truth=truth,prediction=prediction,
            frozen_prediction=frozen_prediction,context_prediction=prediction,route=1,
            action=int(action),queried=bool(action),labels_used=used,cap=cap,
            cumulative_labels=len(session.paid_labels),epsilon=EPSILON,
            q_skip=float(q[0]),q_query=float(q[1]),context_entropy=entropy,
            prediction_before_feedback=True))
    session.clock += n
    if x.is_cuda: torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter()-started
    if used>cap or used!=sum(r['queried'] for r in trace):
        raise RuntimeError('Sequential annotation accounting failed.')
    return dict(level=level,samples=n,labels=used,cap=cap,labels_percent=100*used/n,
        cumulative_labels=len(session.paid_labels),**base.pilot.metrics(y,predictions,len(a.classes)),
        milliseconds_per_row=1000*elapsed/n,memory_counts=memory.storage.counts(),
        Q_beta='fixed; no TEST reward/TD updates'),trace


def snapshot(torch,session,execution_sha):
    return dict(version=1,execution_sha256=execution_sha,
        context={k:v.detach().cpu().clone() for k,v in session.context.state_dict().items()},
        optimizer=session.optimizer.state_dict(),memory=session.memory,
        paid_embeddings=session.paid_embeddings,paid_labels=session.paid_labels,
        clock=session.clock,last_update=session.last_update,
        query_rng=session.query_rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore(torch,session,state,execution_sha):
    if state['version']!=1 or state['execution_sha256']!=execution_sha:
        raise ValueError('Sequential resume state belongs to another run.')
    session.context.load_state_dict(state['context'])
    session.context.eval()
    session.optimizer.load_state_dict(state['optimizer'])
    for name in ('memory','paid_embeddings','paid_labels','clock','last_update'):
        setattr(session,name,state[name])
    session.query_rng.bit_generator.state = state['query_rng']
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def evaluate(output, plan, seed, device, paths, base, meta, torch):
    ds = plan['dataset']; root = output/f'seed{seed}'/'final_sequential_eps0p05_v1'
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

    from enhancements.independent_target_holdout import split_targets,purge_replay
    streams,audits,split = split_targets(domains,42)
    observers = {0:reference,**audits}
    exp.anchors,purge = purge_replay(exp.anchors,observers)
    blocked_ids = {int(i) for data in observers.values() for i in data[2]}
    blocked_keys = set().union(*(set(base.pilot.feature_keys(data[0])) for data in observers.values()))
    execution = dict(version=1,dataset=ds,training_seed=seed,evaluation_seed=42,
        plan_sha256=run.digest(output/'plan.json'),checkpoint_sha256=done['sha256'],
        levels=LEVELS,epsilon=EPSILON,budget_fraction_per_adaptation_domain=.2,
        TEST_data=plan['fixed_TEST_data'],target_split=split,replay_purge=purge,
        retained='Context weights, Adam state, protected replay, paid history and policy RNG across all levels',
        reset='per-domain annotation counter and relative domain progress only',
        history_support='class support normalized by cumulative acquired labels; budget state remains per-domain',
        prediction='Context -> frozen CNN; prediction fixed before current label',
        frozen='CNN, DDQN and beta; no TEST TD learning or reward probes',
        observers='same original-domain and seven joint group-disjoint target sets at stages 0..7; never updated/replayed',
        scope='separate sequential evaluation; target observers take approximately 20 percent of eligible TEST',
        budget_denominator='20 percent of remaining adaptation stream, approximately 16 percent of total eligible target',
        historical_pretraining='existing models reused; no retrospective pretraining duplicate removal claimed',
        evaluator_sha256=run.digest(Path(__file__)),
        metrics_sha256=run.digest(run.ROOT/'final_sequential_metrics.py'),
        split_sha256=run.digest(run.ROOT/'enhancements'/'independent_target_holdout.py'))
    ef = root/'execution.json'
    if ef.exists():
        if run.read(ef)!=execution:
            raise ValueError('Sequential recipe changed; preserve results and use a separate run.')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Unmatched sequential output directory.')
    else:
        run.write(ef,execution)
    execution_sha = run.digest(ef)
    context = exp.context().eval()
    session = SimpleNamespace(context=context,
        optimizer=torch.optim.Adam(context.parameters(),lr=a.context_lr,weight_decay=1e-4),
        memory=exp.memory(42),paid_embeddings=[],paid_labels=[],
        query_rng=base.np.random.default_rng(42+71),clock=0,last_update=0)
    initial_matrix,initial_predictions,initial_done = [root/name for name in
        ('initial_matrix.json','initial_observer_predictions.csv','initial_complete.json')]
    if initial_done.exists():
        record = run.read(initial_done)
        if (record['execution_sha256']!=execution_sha or
                any(run.digest(f)!=record['sha256'][f.name] for f in (initial_matrix,initial_predictions))):
            raise ValueError('Initial sequential observers changed.')
        matrix = run.read(initial_matrix)
    else:
        matrix,predictions = observe(base,torch,exp,session.context,audits,reference,0,0)
        run.write(initial_matrix,matrix)
        base.pilot.write_csv(initial_predictions,predictions)
        run.write(initial_done,dict(execution_sha256=execution_sha,
            sha256={f.name:run.digest(f) for f in (initial_matrix,initial_predictions)}))
    online = []
    for stage,level in enumerate(LEVELS,1):
        location = root/f'stage{stage}_level{level}'
        metric,trace_file,matrix_file,pred_file,state_file,completed = [location/name for name in
            ('online_metrics.json','online_trace.csv','observer_matrix.json',
             'observer_predictions.csv','adaptive_state.pth','completed.json')]
        files = (metric,trace_file,matrix_file,pred_file,state_file)
        if completed.exists():
            record = run.read(completed)
            if (record['execution_sha256']!=execution_sha or
                    any(run.digest(f)!=record['sha256'][f.name] for f in files)):
                raise ValueError('Completed sequential stage changed.')
            state = torch.load(state_file,map_location='cpu',weights_only=False)
            restore(torch,session,state,execution_sha)
            row,observed = run.read(metric),run.read(matrix_file)
            print(f'RESUMED DATASET={ds} TRAIN_SEED={seed} STAGE={stage} LEVEL={level} '
                  'Context/Adam/replay/history restored',flush=True)
        else:
            print(f'START SEQUENTIAL DATASET={ds} TRAIN_SEED={seed} STAGE={stage}/7 '
                  f'LEVEL={level} EPSILON=.05 CAP={int(.2*len(streams[level][1]))} '
                  f'PREVIOUS_PAID={len(session.paid_labels)} | NO RESET OF ADAPTIVE MODEL',flush=True)
            row,trace = sequential_stage(base,torch,exp,learner,session,streams[level],
                                         level,blocked_ids,blocked_keys)
            observed,predictions = observe(base,torch,exp,session.context,audits,reference,stage,level)
            run.write(metric,row)
            base.pilot.write_csv(trace_file,trace)
            run.write(matrix_file,observed)
            base.pilot.write_csv(pred_file,predictions)
            temporary = state_file.with_suffix('.pth.tmp')
            torch.save(snapshot(torch,session,execution_sha),temporary)
            temporary.replace(state_file)
            run.write(completed,dict(execution_sha256=execution_sha,
                sha256={f.name:run.digest(f) for f in files}))
        online.append(row); matrix.extend(observed)
        if (session.clock!=sum(r['samples'] for r in online) or
                len(session.paid_labels)!=sum(r['labels'] for r in online)):
            raise RuntimeError('Sequential history/counter continuity failed.')
        print(f'STAGE COMPLETE DATASET={ds} TRAIN_SEED={seed} LEVEL={level} '
              f'ONLINE_AC={row["AC"]:.2f}% F1={row["F1"]:.2f}% '
              f'LABELS={row["labels"]}/{row["cap"]} CUMULATIVE_PAID={len(session.paid_labels)}',flush=True)
        for observed_row in observed:
            if observed_row['domain']==0 or observed_row['domain'] in LEVELS[:stage]:
                print(f'  RETEST AFTER_LEVEL={level} OBSERVER_DOMAIN={observed_row["domain"]} '
                      f'AC={observed_row["AC"]:.2f}% F1={observed_row["F1"]:.2f}%',flush=True)
        frozen = (base.weight_hash(exp.cnn.state_dict()),base.weight_hash(exp.template.state_dict()),
                  base.weight_hash(learner.policy.state_dict()),learner.beta_value(),learner.steps,learner.updates)
        if frozen!=before:
            raise RuntimeError('Sequential TEST modified a frozen source model/DDQN/beta.')
    summary = summarize(matrix,online)
    summary.update(dataset=ds,training_seed=seed,evaluation_seed=42,epsilon=EPSILON,
                   protocol='sequential 5,7,10,12,15,17,20 with fixed independent observers')
    run.write(root/'summary.json',summary)
    base.pilot.write_csv(root/'observer_matrix.csv',matrix)
    base.pilot.write_csv(root/'online_by_level.csv',online)
    run.write(root/'completed.json',dict(training_seed=seed,execution_sha256=execution_sha,
        summary_sha256=run.digest(root/'summary.json'),
        matrix_sha256=run.digest(root/'observer_matrix.csv')))
    if run.digest(model)!=done['sha256']:
        raise RuntimeError('Original final checkpoint file changed.')
    print(f'SEED SUMMARY DATASET={ds} TRAIN_SEED={seed} '+json.dumps(summary['AC'],sort_keys=True),flush=True)


def collect(output,plan,seeds):
    rows = []
    for seed in seeds:
        root = output/f'seed{seed}'/'final_sequential_eps0p05_v1'
        done = run.read(root/'completed.json')
        ex = run.read(root/'execution.json')
        if (done['execution_sha256']!=run.digest(root/'execution.json') or
                done['summary_sha256']!=run.digest(root/'summary.json') or
                done['matrix_sha256']!=run.digest(root/'observer_matrix.csv') or
                ex['plan_sha256']!=run.digest(output/'plan.json') or
                ex['evaluator_sha256']!=run.digest(Path(__file__)) or ex['training_seed']!=seed):
            raise ValueError('Incomplete or changed sequential results.')
        summary = run.read(root/'summary.json')
        row = dict(dataset=plan['dataset'],training_seed=seed,evaluation_seed=42,epsilon=EPSILON)
        for metric in METRICS:
            row.update({metric+'_'+name:value for name,value in summary[metric].items()})
        row['labels_percent'] = summary['feedback']['labels_percent']
        rows.append(row)
    overall = dict(dataset=plan['dataset'],seeds=len(rows),epsilon=EPSILON)
    for key in rows[0]:
        if key in ('dataset','training_seed','evaluation_seed','epsilon'): continue
        values = [r[key] for r in rows]
        overall[key+'_mean'] = sum(values)/len(values)
        overall[key+'_sd'] = run.moments(values)[1] if len(values)>1 else None
    target = output/'Results_final_sequential_eps0p05_v1'/('seeds_'+'_'.join(map(str,seeds)))
    run.csv_write(target/'summary_per_seed.csv',rows)
    run.csv_write(target/'summary_overall.csv',[overall])
    run.write(target/'completed.json',dict(seeds=seeds,plan_sha256=run.digest(output/'plan.json'),
        overall=overall,evaluator_sha256=run.digest(Path(__file__))))
    print('ALL SEQUENTIAL SEEDS COMPLETE '+json.dumps(overall,sort_keys=True),flush=True)
    print('OUTPUT: '+str(target),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,48,54,62,72])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--action',choices=['Evaluate','Collect'],default='Evaluate')
    args = p.parse_args()
    output = run.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'protocol_runs'/'final_multiseed_fixedcnn_v1'
    plan = run.read(output/'plan.json')
    if (plan['dataset']!=args.dataset or plan['evaluation_seed']!=42 or
            not args.seeds or len(set(args.seeds))!=len(args.seeds) or
            any(s not in plan['training_seeds'] for s in args.seeds)):
        p.error('Use distinct existing final training seeds.')
    original,argv = worker.evaluate,sys.argv
    worker.evaluate = evaluate
    try:
        if args.action=='Evaluate':
            for seed in args.seeds:
                sys.argv = ['amcal_final_multiseed_worker.py','--output-root',str(output),
                    '--training-seed',str(seed),'--stage','evaluate','--device',args.device]
                worker.main()
    finally:
        worker.evaluate,sys.argv = original,argv
    collect(output,plan,args.seeds)


if __name__=='__main__':
    main()
