"""Opt-in IAT Light CNN control: development 0/1/3, full TEST, learned beta.

Reuse the existing Light IAT healthy teacher and paper-reward Actor. The fixed-beta
warm start, gain probes, DDQN, meta-beta learner and evaluator are shared with the
original-CNN experiment; no existing module or checkpoint is modified.
"""
import argparse
import json
from pathlib import Path

import torch
import amcal_meta_beta_ddqn as meta
import amcal_two_action_paper_gain as gain
import amcal_two_action_ddqn_light_iat as light

ROOT = Path(__file__).resolve().parent
TRAIN = [0, 1, 3]
TEST = [5, 7, 10, 12, 15, 17, 20]
SOURCE_BETA = 26.0
DEPENDENCIES = (
    'amcal_meta_beta_light_iat013.py', 'amcal_meta_beta_ddqn.py',
    'amcal_two_action_paper_gain.py', 'amcal_two_action_ddqn_light_iat.py',
    'amcal_competitive_selector_light_iat.py', 'amcal_competitive_selector.py',
    'amcal_budget_actor.py', 'amcal_minimax_paironly.py',
    'enhancements/cnn_variant.py', 'enhancements/meta_beta_ddqn.py',
    'enhancements/two_action_ddqn.py', 'enhancements/paper_gain_reward.py',
    'enhancements/query_utility_reward.py', 'enhancements/competitive_selector.py',
    'enhancements/minimax_query.py',
)


class LightGainExperiment(light.Experiment):
    set_gain_reward = gain.PaperGainExperiment.set_gain_reward
    reward_for_domain = gain.PaperGainExperiment.reward_for_domain

    def __init__(self, args):
        args.light_cnn = True
        super().__init__(args)
        if set(self.train) != set(TRAIN):
            raise ValueError('Expected TRAIN domains 0/1/3.')
        checkpoint = torch.load(args.models/'initial_cnn_model.pth',
                                map_location='cpu', weights_only=False)
        if checkpoint.get('architecture') != 'light_ps_cnn_v1':
            raise ValueError('This control requires the existing Light IAT CNN.')


def checked_source(exp, protocol, folder):
    path = folder/'Models'/'two_action_ddqn.pth'
    done = json.loads((folder/'training_complete.json').read_text(encoding='utf-8'))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (done['protocol'] != protocol or saved['protocol'] != protocol
            or done['checkpoint_sha256'] != gain.pilot.digest(path)):
        raise ValueError('Light fixed-beta warm-start verification failed.')
    return saved, gain.pilot.digest(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['All', 'Train', 'Evaluate'], default='All')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--teacher', type=Path)
    p.add_argument('--actor-run', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.epochs < 1:
        p.error('Epochs must be positive.')
    runs = ROOT/'ANT_Burst_IAT_TEST'/'protocol_runs'
    a.dataset = 'IAT'
    a.pilot_rows = 300
    a.test_rows = 0
    a.test_budgets = [.2]
    a.budget_fraction = .2
    a.beta = SOURCE_BETA
    a.burst_dir = ROOT/'ANT_Burst_IAT_TEST'/'Dataset_official_v1'/'AdvBurst_FTSC-IAT'
    a.teacher = (a.teacher or runs/
        f'light_iat_minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural').resolve()
    a.actor_run = (a.actor_run or runs/
        f'light_iat_competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward').resolve()
    output = (a.output or runs/
        f'light_iat013_meta_beta_ddqn_seed{a.seed}_p300_e{a.epochs}_v1').resolve()
    for parent, filename in ((a.teacher, 'minimax_query.pth'), (a.actor_run, 'selector.pth')):
        for path in (parent/'Models'/filename, parent/'training_complete.json'):
            if not path.is_file():
                p.error('Required existing Light checkpoint missing: '+str(path))
        if output == parent or parent in output.parents or output in parent.parents:
            p.error('Output must be separate from existing checkpoint directories.')
    a.output = output
    print('IAT LIGHT CONTROL | TRAIN/VAL=0,1,3 | CNN/CONTEXT REUSED | '
          'TEST=5,7,10,12,15,17,20 | ALL ELIGIBLE TEST ROWS | BUDGET=20%', flush=True)
    exp = LightGainExperiment(a)
    source_weights = (gain.weight_hash(exp.cnn.state_dict()),
                      gain.weight_hash(exp.template.state_dict()))
    source_files = dict(exp.healthy['model_hashes'])
    hashes = {name: gain.pilot.digest(ROOT/name) for name in DEPENDENCIES}
    common = dict(version=1, dataset='IAT', seed=a.seed, epochs=a.epochs,
        pilot_rows=300, test_rows=0, training_levels=TRAIN, validation_levels=TRAIN,
        test_levels=TEST, cnn_architecture='light_ps_cnn_v1',
        actor_checkpoint_sha256=exp.actor_sha, healthy_teacher_sha256=exp.teacher_hash,
        source_models=source_files, train_files=exp.healthy['training_domain_sha256'],
        train_data=exp.data_manifest('train_ids', TRAIN),
        validation_data=exp.data_manifest('val_ids', TRAIN),
        context_lr=a.context_lr, replay_weight=a.replay_weight, budget_fraction=.2,
        implementation_sha256=hashes,
        initialization='existing Light IAT healthy CNN/Context and paper Actor; no CNN retraining',
        scope='same final-family learned-beta pipeline; TRAIN300 per known level, full eligible TEST',
        evaluation='pre-feedback; fixed Context path; frozen Q/beta; reset Context/Adam/memory/budget per level')
    warm_root = output/'WarmStart'/'beta_26'
    a.output = warm_root
    warm_probes, retention, anchors, probe_manifest = gain.prepare_train_probes(
        exp, gain.pilot, 24, 32)
    exp.train_anchors = anchors
    exp.reward_probe_keys = set().union(*(set(gain.pilot.feature_keys(d[0]))
                                        for d in warm_probes))
    exp.set_gain_reward(warm_probes, retention, SOURCE_BETA)
    warm_protocol = dict(protocol=dict(common, probes=probe_manifest), beta=SOURCE_BETA)
    if a.action in ('Train', 'All'):
        print('STAGE=FIXED_BETA_WARMSTART | BETA=26 | TRAIN/VAL=0,1,3', flush=True)
        gain.pilot.base.legacy.core.set_seed(a.seed)
        learner = gain.QueryDDQN(exp.spec, exp.actor_saved['state'],
                                gain.pilot.base.legacy.core.device, a.seed)
        gain.train(exp, learner, warm_protocol)
    source, source_hash = checked_source(exp, warm_protocol, warm_root)
    # Recreate the experiment: do not reserve probes twice from a mutated stream.
    a.output = output
    exp = LightGainExperiment(a)
    inner, retention, anchors, probe_manifest = gain.prepare_train_probes(
        exp, gain.pilot, 24, 32)
    exp.train_anchors = anchors
    streams, probes, anchors, meta_manifest = meta.reserve_meta(exp, inner)
    exp.train_anchors = anchors
    exp.reward_probe_keys = set().union(*(set(gain.pilot.feature_keys(d[0]))
                                        for d in [*inner, *probes.values()]))
    protocol = dict(common, source_beta=SOURCE_BETA, source_checkpoint_sha256=source_hash,
        source_protocol=warm_protocol, inner_probe_manifest=probe_manifest,
        meta_TRAIN_manifest=meta_manifest,
        inner_probe_sha256=[gain.domain_hash(d) for d in inner],
        meta_replay_sha256=gain.domain_hash(anchors),
        beta='unchanged TRAIN-only MetaBetaDDQN; TEST beta and Q frozen',
        validation='mean online AC minus positive source forgetting on VAL0/1/3')
    pf = output/'protocol.json'
    if pf.exists():
        if json.loads(pf.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protocol changed; preserve output and choose a fresh --output.')
    elif a.action == 'Evaluate':
        raise ValueError('Train this separate Light0/1/3 learned-beta checkpoint first.')
    elif any(path.name != 'WarmStart' for path in output.iterdir()):
        raise ValueError('Unmatched partial output preserved; choose a fresh --output.')
    else:
        gain.pilot.write_json(pf, protocol)
    learner = meta.learner_from_source(exp, source)
    if a.action in ('Train', 'All'):
        print('STAGE=LEARNED_BETA | TRAIN/VAL=0,1,3 | SAME DDQN/GAIN REWARD', flush=True)
        meta.train(exp, learner, protocol, inner, streams, probes, retention)
    if a.action in ('Evaluate', 'All'):
        print('STAGE=FULL_TEST | BUDGET=20% | NO TEST ROW LIMIT | '
              'POLICIES=LearnedBeta,SourceFixed,Uncertainty', flush=True)
        meta.evaluate(exp, source, protocol)
    if source_weights != (gain.weight_hash(exp.cnn.state_dict()),
                          gain.weight_hash(exp.template.state_dict())):
        raise RuntimeError('Initial CNN/Context weights changed.')
    for name, digest in source_files.items():
        if gain.pilot.digest(a.models/name) != digest:
            raise RuntimeError('Source model file changed: '+name)
    if (gain.pilot.digest(a.teacher/'Models'/'minimax_query.pth') != exp.teacher_hash
            or gain.pilot.digest(a.actor_run/'Models'/'selector.pth') != exp.actor_sha):
        raise RuntimeError('Teacher or Actor checkpoint changed.')
    print('OUTPUT: '+str(output), flush=True)


if __name__ == '__main__':
    main()
