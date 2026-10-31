"""Keep the current bundle and compare a CNN-only post-selector TRAIN 1/3 fine-tune."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import amcal_two_action_paper_gain as base

LEVELS = [5, 7, 10, 12, 15, 17, 20]
VARIANTS = ['Current', 'CNN13']


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def raw_cnn_metrics(model, data, classes):
    if model.training: raise ValueError('CNN validation requires eval mode.')
    x, y = data[:2]; predictions = []; loss = 0.
    if not len(y): raise ValueError('Empty CNN validation data.')
    for start in range(0, len(y), 64):
        target = torch.as_tensor(y[start:start+64], dtype=torch.long, device=x.device)
        logits = model(x[start:start+64])
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite CNN validation logits.')
        loss += float(nn.functional.cross_entropy(logits, target, reduction='sum'))
        predictions.extend(logits.argmax(1).cpu().tolist())
    return dict(loss=loss/len(y), **base.pilot.metrics(y, predictions, classes))


def fit_cnn(model, train, validation, args, directory, protocol):
    """Only this copied CNN is optimized; Context and query models are not inputs."""
    trainer_file = directory/'cnn13_training_state.pth'
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cnn_lr, weight_decay=.0001)
    if trainer_file.exists():
        saved = torch.load(trainer_file, map_location='cpu', weights_only=False)
        if saved['protocol'] != protocol: raise ValueError('CNN resume protocol changed; use a new output.')
        model.load_state_dict(saved['current']); optimizer.load_state_dict(saved['optimizer'])
        history, best, best_state = saved['history'], saved['best'], saved['best_state']
        stale, start = saved['stale'], saved['epoch']+1
        torch.set_rng_state(saved['cpu_rng'])
        if next(model.parameters()).is_cuda and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'], next(model.parameters()).device)
    else:
        history = []; best = float('inf'); best_state = None; stale = 0; start = 1
    x, y = train[:2]
    truth = torch.as_tensor(y, dtype=torch.long, device=x.device)
    batch_size = min(32, len(y))
    while batch_size > 2 and len(y) % batch_size == 1: batch_size -= 1
    if batch_size < 2 or len(y) % batch_size == 1:
        raise ValueError('CNN training requires batches with at least two rows.')
    for epoch in range(start, args.cnn_epochs+1):
        if stale >= args.patience: break
        loader = DataLoader(TensorDataset(x, truth), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed+13000+epoch))
        model.train(); total = 0.; hits = 0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs); loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss): raise ValueError('Nonfinite CNN fine-tuning loss.')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*len(labels)
            hits += int((logits.argmax(1) == labels).sum())
        model.eval(); report = raw_cnn_metrics(model, validation, len(args.classes))
        history.append(dict(epoch=epoch, train_loss=total/len(y), train_AC=100*hits/len(y), validation=report))
        if report['loss'] < best:
            best = report['loss']; best_state = cpu_state(model); stale = 0
        else: stale += 1
        base.parent.save_model(trainer_file, dict(protocol=protocol, epoch=epoch, current=cpu_state(model),
            optimizer=deepcopy(optimizer.state_dict()), history=history, best=best, best_state=best_state,
            stale=stale, cpu_rng=torch.get_rng_state(), cuda_rng=(torch.cuda.get_rng_state(next(model.parameters()).device)
                if next(model.parameters()).is_cuda else None)))
        base.pilot.write_json(directory.parent/'cnn13_training.json', dict(protocol=protocol, epochs=history))
        print(f'DATASET={args.dataset} BETA={args.beta:g} CNN13 EPOCH={epoch}/{args.cnn_epochs} | '
            f'TRAIN_AC={100*hits/len(y):.2f}% | VAL_AC={report["AC"]:.2f}% VAL_F1={report["F1"]:.2f}% '
            f'VAL_CE={report["loss"]:.6f} | Context/DDQN unchanged', flush=True)
    if best_state is None: raise ValueError('No finite fine-tuned CNN epoch was saved.')
    model.load_state_dict(best_state); model.eval().requires_grad_(False)
    selected = min(history, key=lambda row:row['validation']['loss'])['epoch']
    return dict(selected_epoch=selected, epochs=history, selection='minimum CNN CE on VAL 1/3 only; TEST unused')


def development_data(experiment, train_rows, val_rows):
    a = experiment.a
    held_ids = set(map(int, experiment.reference[2])); held_keys = set(base.pilot.feature_keys(experiment.reference[0]))
    groups = {'train_ids':{}, 'val_ids':{}}
    for level in (1, 3):
        for split, limit in [('train_ids',train_rows),('val_ids',val_rows)]:
            data = base.pilot.base.legacy.read_domain(a.burst_dir, experiment.artifact, level, split)
            if split == 'train_ids':
                mask = np.asarray([int(i) not in held_ids and key not in held_keys
                    for i,key in zip(data[2],base.pilot.feature_keys(data[0]))])
                data = base.pilot.subset(data, mask)
            if limit:
                at = base.pilot.pilot_indices(data[1], limit, a.seed+(702 if split == 'train_ids' else 703))
                mask = np.zeros(len(data[1]), dtype=bool); mask[at] = True
                data = base.pilot.subset(data, mask)
            if not len(data[1]) or set(np.unique(data[1])) != set(range(len(a.classes))):
                raise ValueError('Fine-tuning TRAIN/VAL 1/3 must retain every class.')
            groups[split][level] = data
    train_ids = {int(i) for data in groups['train_ids'].values() for i in data[2]}
    val_ids = {int(i) for data in groups['val_ids'].values() for i in data[2]}
    if train_ids.intersection(val_ids): raise ValueError('Fine-tuning TRAIN/VAL IDs overlap.')
    def combine(values):
        return torch.cat([values[level][0] for level in (1,3)]), np.concatenate([values[level][1] for level in (1,3)]), \
            np.concatenate([values[level][2] for level in (1,3)])
    manifest = {split:{str(level):dict(rows=len(data[1]), sha256=base.domain_hash(data))
        for level,data in values.items()} for split,values in groups.items()}
    # Reuse the official source partitions without silently changing the benchmark.
    tk = set().union(*(set(base.pilot.feature_keys(d[0])) for d in groups['train_ids'].values()))
    vk = set().union(*(set(base.pilot.feature_keys(d[0])) for d in groups['val_ids'].values()))
    manifest['feature_duplicate_count_between_official_partitions'] = len(tk.intersection(vk))
    return combine(groups['train_ids']), combine(groups['val_ids']), manifest


def save_bundle(path, variant, cnn, experiment, learner, protocol, tuning=None):
    base.parent.save_model(path, dict(format_version=1, variant=variant, protocol=protocol,
        cnn_state_dict=cpu_state(cnn), context_state_dict=cpu_state(experiment.template),
        query_state_dict=cpu_state(learner.policy), query_spec=deepcopy(experiment.spec),
        config=deepcopy(experiment.healthy['config']), classes=list(map(str, experiment.a.classes)),
        tuning=tuning, prediction='fixed Context -> CNN; query policy frozen during TEST'))


def load_variant(experiment, learner, path, protocol, original_context_hash, original_query_hash):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle['format_version'] != 1 or bundle['protocol'] != protocol or bundle['query_spec'] != experiment.spec:
        raise ValueError('Comparison bundle provenance differs.')
    if (base.weight_hash(bundle['context_state_dict']) != original_context_hash or
            base.weight_hash(bundle['query_state_dict']) != original_query_hash):
        raise ValueError('Context or DDQN changed during CNN-only fine-tuning.')
    experiment.cnn.load_state_dict(bundle['cnn_state_dict']); experiment.cnn.eval().requires_grad_(False)
    learner.policy.load_state_dict(bundle['query_state_dict']); learner.policy.eval().requires_grad_(False)
    return bundle


def compare_case(experiment, learner, variant, level, domain, reference, common_reference,
                 directory, execution):
    a = experiment.a; a.budget_fraction = .2
    tag = f'DATASET={a.dataset} BETA={a.beta:g} VARIANT={variant} LEVEL={level} BUDGET=20%'
    print('\nSTART '+tag+' | RESET=Context/Adam/memory', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    ef, cf, mf, tf = directory/'execution.json', directory/'completed.json', directory/'metrics.json', directory/'trace.csv'
    if ef.exists():
        if json.loads(ef.read_text(encoding='utf-8')) != execution: raise ValueError('CNN comparison case changed.')
    elif any(directory.iterdir()): raise ValueError('Unmatched partial CNN comparison case preserved.')
    else: base.pilot.write_json(ef, execution)
    reused = cf.exists()
    if reused:
        done = json.loads(cf.read_text(encoding='utf-8'))
        if done.get('execution_sha256') != base.pilot.digest(ef) or any(not p.is_file() or
                base.pilot.digest(p) != done.get('sha256',{}).get(p.name) for p in (mf,tf)):
            raise ValueError('Completed CNN comparison metrics/trace changed.')
        result = json.loads(mf.read_text(encoding='utf-8'))
    else:
        before = base.weight_hash(learner.policy.state_dict()); updates, steps = learner.updates, learner.steps
        result, trace = base.run_episode(experiment, learner, domain, reference, a.seed, training=False, mode='DDQN')
        if before != base.weight_hash(learner.policy.state_dict()) or updates != learner.updates or steps != learner.steps:
            raise RuntimeError('Comparison TEST updated the DDQN.')
        if result['td_updates'] != 0 or result['cap'] != int(.2*len(domain[1])) or result['labels'] > result['cap']:
            raise RuntimeError('CNN comparison violated frozen-query or budget accounting.')
        result['cnn_finetune_source_drop'] = {k:common_reference[k]-result['source_before'][k] for k in common_reference}
        result['total_source_drop_from_current_bundle'] = {k:common_reference[k]-result['source_after'][k] for k in common_reference}
        base.pilot.write_csv(tf,trace); base.pilot.write_json(mf,result)
        base.pilot.write_json(cf,dict(execution_sha256=base.pilot.digest(ef),sha256={p.name:base.pilot.digest(p) for p in (mf,tf)}))
    row = dict(variant=variant,beta=a.beta,**base.result_row(variant,level,result),
        cnn_finetune_source_drop=result['cnn_finetune_source_drop']['AC'],
        total_source_drop=result['total_source_drop_from_current_bundle']['AC'])
    for key in ('PR','RC','F1'):
        row['target_final_'+key] = result['target_after'][key]
        row['total_source_drop_'+key] = result['total_source_drop_from_current_bundle'][key]
    print(('REUSED ' if reused else 'DONE ')+tag+
        f' | AC={row["AC"]:.2f}% F1={row["F1"]:.2f}% LABELS={row["labels_percent"]:.2f}% '
        f'FINAL_AC={row["target_final_AC"]:.2f}% | CNN_FT_SOURCE_DROP={row["cnn_finetune_source_drop"]:+.2f} pp '
        f'ONLINE_SOURCE_DROP={row["forgetting"]:+.2f} pp TOTAL_SOURCE_DROP={row["total_source_drop"]:+.2f} pp',flush=True)
    return row


def summarize(rows):
    reports = []
    print('\nMEAN OVER SEVEN LEVELS: CURRENT VERSUS CNN13, SAME CONTEXT AND DDQN', flush=True)
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        if not group: continue
        report = dict(variant=variant,beta=group[0]['beta'],levels=len(group))
        for key in ('AC','PR','RC','F1','labels_percent','target_final_AC','target_final_PR','target_final_RC',
                'target_final_F1','source_final_AC','forgetting','cnn_finetune_source_drop','total_source_drop',
                'total_source_drop_PR','total_source_drop_RC','total_source_drop_F1'):
            report[key] = float(np.mean([r[key] for r in group]))
        reports.append(report)
        print(f'VARIANT={variant} BETA={report["beta"]:g} | '+
            ' | '.join(f'{key}={report[key]:.2f}' for key in ('AC','PR','RC','F1','labels_percent',
                'target_final_AC','target_final_F1','cnn_finetune_source_drop','forgetting','total_source_drop')),flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['IAT','PS'],required=True)
    parser.add_argument('--beta',type=float)
    parser.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-rows',type=int,default=300,help='CNN fine-tuning rows per TRAIN level; zero uses all.')
    parser.add_argument('--val-rows',type=int,default=300)
    parser.add_argument('--test-rows',type=int,default=300)
    parser.add_argument('--cnn-epochs',type=int,default=5)
    parser.add_argument('--cnn-lr',type=float,default=.0001)
    parser.add_argument('--patience',type=int,default=2)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    parser.add_argument('--selector-run',type=Path)
    parser.add_argument('--actor-run',type=Path)
    parser.add_argument('--teacher',type=Path)
    parser.add_argument('--output',type=Path)
    a = parser.parse_args()
    if a.beta is None: a.beta = 26. if a.dataset == 'IAT' else 55.
    if (not np.isfinite(a.beta) or a.beta < 0 or not np.isfinite(a.cnn_lr) or a.cnn_lr <= 0
            or min(a.cnn_epochs,a.patience) < 1 or any(n < 0 or 0 < n < 32 for n in (a.train_rows,a.val_rows,a.test_rows))):
        parser.error('Use finite nonnegative beta, positive CNN LR/epochs/patience and row limits zero or at least 32.')
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    parent_stem = f'two_action_paper_gain_seed{a.seed}_p300_e3_v1' if a.beta in (0.,10.,100.) else f'two_action_paper_gain_extra_seed{a.seed}_p300_e3_v1'
    a.selector_run = a.selector_run or root/'protocol_runs'/parent_stem/'Candidates'/base.beta_name(a.beta)
    source_model = a.selector_run/'Models'/'two_action_ddqn.pth'
    source_completion = json.loads((a.selector_run/'training_complete.json').read_text(encoding='utf-8'))
    source = torch.load(source_model,map_location='cpu',weights_only=False)
    if (source['protocol'] != source_completion['protocol'] or base.pilot.digest(source_model) != source_completion['checkpoint_sha256']
            or float(source['protocol'].get('beta',-1)) != a.beta):
        raise ValueError('Current selector beta/checkpoint provenance differs.')
    source_protocol = source['protocol']['protocol']
    if source_protocol['dataset'] != a.dataset or source_protocol['seed'] != a.seed or 'synthetic' in source_protocol:
        raise ValueError('Use the matching non-synthetic original paper-plus-gain checkpoint.')
    for name,expected in source_protocol['implementation_sha256'].items():
        if base.pilot.digest(Path(__file__).parent/name) != expected: raise ValueError('Source selector implementation changed: '+name)
    a.pilot_rows = source_protocol['pilot_rows']; a.budget_fraction = .2
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = a.output or root/'protocol_runs'/f'post_selector_cnn13_seed{a.seed}_{base.beta_name(a.beta)}_trainp{a.train_rows}_v1'
    experiment = base.PaperGainExperiment(a)
    if (source_protocol['healthy_teacher_sha256'] != experiment.teacher_hash or source_protocol['actor_checkpoint_sha256'] != experiment.actor_sha
            or source_protocol['source_models'] != experiment.healthy['model_hashes']
            or source_protocol['train_files'] != experiment.healthy['training_domain_sha256']):
        raise ValueError('Current selector and source CNN/Context lineage differ.')
    if (source_protocol['train_data'] != experiment.data_manifest('train_ids',base.TRAIN) or
            source_protocol['validation_data'] != experiment.data_manifest('val_ids',base.TRAIN)):
        raise ValueError('Source selector TRAIN/VAL inputs changed.')
    learner = base.QueryDDQN(experiment.spec,experiment.actor_saved['state'],base.pilot.base.legacy.core.device,a.seed)
    learner.policy.load_state_dict(source['policy_state']); learner.policy.eval().requires_grad_(False)
    original_cnn = deepcopy(experiment.cnn).eval().requires_grad_(False)
    context_hash = base.weight_hash(experiment.template.state_dict()); query_hash = base.weight_hash(learner.policy.state_dict())
    train,validation,data_manifest = development_data(experiment,a.train_rows,a.val_rows)
    architecture = torch.load(a.models/'initial_cnn_model.pth',map_location='cpu',weights_only=False).get('architecture','legacy')
    protocol = dict(version=1,dataset=a.dataset,seed=a.seed,beta=a.beta,source_query_sha256=base.pilot.digest(source_model),
        source_query_protocol=source['protocol'],healthy_teacher_sha256=experiment.teacher_hash,
        actor_checkpoint_sha256=experiment.actor_sha,architecture=architecture,development_data=data_manifest,
        cnn_epochs=a.cnn_epochs,cnn_lr=a.cnn_lr,patience=a.patience,
        CNN_training='additional raw preprocessed TRAIN 1/3 only; ordinary CE; AdamW weight_decay .0001; batch32; clip1',
        CNN_selection='minimum raw CNN CE on VAL 1/3 among fine-tuned epochs; TEST unused; baseline metrics retained separately',
        training_rng_seed=a.seed+1313,levels=[1,3],
        initial_CNN_sha256=base.weight_hash(original_cnn.state_dict()),unchanged_context_sha256=context_hash,
        unchanged_query_sha256=query_hash,preprocessing_sha256=base.pilot.digest(a.models/'protocol_preprocessing.joblib'),
        implementation_sha256=base.pilot.digest(Path(__file__)),budget_fraction=.2,
        update='fixed pretrained DDQN; original Context initial weights; batch5; LR .00025; replay .1',
        interpretation='CNN-only post-selector fine-tuning comparison; changing CNN also changes DDQN input signals')
    a.output.mkdir(parents=True,exist_ok=True); protocol_file = a.output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != protocol: raise ValueError('CNN13 recipe changed; use a fresh output.')
    elif any(a.output.iterdir()): raise ValueError('Unmatched partial CNN13 output preserved.')
    else:
        if a.action == 'Evaluate': raise ValueError('Prepare the two bundles before evaluation.')
        base.pilot.write_json(protocol_file,protocol)
    models = a.output/'Models'; models.mkdir(exist_ok=True)
    bundles = {'Current':models/'current_bundle.pth','CNN13':models/'cnn13_bundle.pth'}
    completion_file = a.output/'training_complete.json'
    if a.action in ('Train','All') and not completion_file.exists():
        print(f'DATASET={a.dataset} BETA={a.beta:g} | PRESERVE Current | fine-tune only copied CNN on TRAIN 1/3',flush=True)
        preprocessing = models/'protocol_preprocessing.joblib'
        if preprocessing.exists() and base.pilot.digest(preprocessing) != protocol['preprocessing_sha256']:
            raise ValueError('Copied preprocessing changed.')
        if not preprocessing.exists(): shutil.copy2(a.models/'protocol_preprocessing.joblib',preprocessing)
        save_bundle(bundles['Current'],'Current',original_cnn,experiment,learner,protocol)
        before = raw_cnn_metrics(original_cnn,validation,len(a.classes))
        ft_model = deepcopy(original_cnn)
        base.pilot.base.legacy.core.set_seed(a.seed+1313)
        report = fit_cnn(ft_model,train,validation,a,models,protocol)
        after = raw_cnn_metrics(ft_model,validation,len(a.classes))
        if (context_hash != base.weight_hash(experiment.template.state_dict()) or query_hash != base.weight_hash(learner.policy.state_dict())
                or protocol['initial_CNN_sha256'] != base.weight_hash(original_cnn.state_dict())):
            raise RuntimeError('CNN-only training changed an original source model.')
        save_bundle(bundles['CNN13'],'CNN13',ft_model,experiment,learner,protocol,report)
        base.pilot.write_json(a.output/'finetune_summary.json',dict(before=before,after=after,selected_epoch=report['selected_epoch'],
            context_unchanged=True,query_unchanged=True,original_cnn_unchanged=True,levels=[1,3]))
        base.pilot.write_json(completion_file,dict(protocol=protocol,selected_epoch=report['selected_epoch'],
            bundle_sha256={name:base.pilot.digest(path) for name,path in bundles.items()}))
    if completion_file.exists():
        complete = json.loads(completion_file.read_text(encoding='utf-8'))
        if complete['protocol'] != protocol or any(not path.is_file() or base.pilot.digest(path) != complete['bundle_sha256'][name]
                for name,path in bundles.items()): raise ValueError('CNN13 completed bundles changed.')
    else: raise ValueError('CNN13 training is incomplete.')
    if a.action in ('Evaluate','All'):
        a.pilot_rows = a.test_rows
        domains,reference = experiment.split('test_ids',LEVELS)
        test_manifest = experiment.data_manifest('test_ids',LEVELS)
        common_reference = base.endpoint(experiment,experiment.template,reference)
        results_root = a.output/'Results'/f'testp{a.test_rows}'
        rows = []
        for variant in VARIANTS:
            loaded = load_variant(experiment,learner,bundles[variant],protocol,context_hash,query_hash)
            if loaded['variant'] != variant: raise ValueError('Comparison bundle labels were swapped.')
            print(f'\nCOMPARE DATASET={a.dataset} BETA={a.beta:g} VARIANT={variant} | same Context/DDQN | BUDGET=20%',flush=True)
            for level in LEVELS:
                execution = dict(protocol=protocol,bundle_sha256=complete['bundle_sha256'][variant],variant=variant,level=level,
                    TEST_data=test_manifest,common_source_reference=common_reference)
                rows.append(compare_case(experiment,learner,variant,level,domains[level],reference,common_reference,
                    results_root/variant/f'level_{level}',execution))
                base.pilot.write_csv(results_root/'summary_by_level.csv',rows)
        base.pilot.write_csv(results_root/'summary_mean.csv',summarize(rows))
        base.pilot.write_json(results_root/'completed.json',dict(cases=len(rows),TEST_data=test_manifest,
            common_source_reference=common_reference,protocol_sha256=base.pilot.digest(protocol_file),
            summary_sha256={name:base.pilot.digest(results_root/name) for name in ('summary_by_level.csv','summary_mean.csv')}))
    if (base.pilot.digest(source_model) != protocol['source_query_sha256'] or
            base.pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash or
            base.pilot.digest(a.actor_run/'Models'/'selector.pth') != experiment.actor_sha):
        raise RuntimeError('An original source checkpoint file changed.')
    print('CNN13 COMPARISON OUTPUT: '+str(a.output),flush=True)


if __name__ == '__main__': main()
