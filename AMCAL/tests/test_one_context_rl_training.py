"""Dedicated one-Context policy training: causal feedback and real CLI provenance."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_final_enhancement as runner
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EnhancementAgent
from enhancements.config import EnhancementConfig
from enhancements.stream import episode
import test_final_enhancement as fixtures


class OneContextRLTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        legacy.core.device = torch.device('cpu')

    def test_incompatible_training_modes_fail_before_writing(self):
        base = ['train','--output','unused','--burst-dir','unused','--profile','accuracy',
                '--warm-start-enhancement','--aligned-sequence','--one-context-rl-training',
                '--context-count','1','--query-schedule','policy','--update-policy','learned']
        variants = [base+['--context-count','2'], base+['--query-schedule','uniform'],
                    base+['--counterfactual-utility'],base+['--joint-router'],
                    [x for x in base if x!='--warm-start-enhancement'],
                    [x for x in base if x!='--one-context-rl-training']+['--query-value-only-training']]
        with patch.object(runner,'train') as train, redirect_stderr(io.StringIO()):
            for argv in variants:
                with self.subTest(argv=argv), self.assertRaises(SystemExit): runner.main(argv)
            train.assert_not_called()

    def test_policy_train_never_learns_from_unqueried_labels(self):
        self.check_policy_causality(False)

    def test_query_value_only_train_never_learns_from_unqueried_labels(self):
        self.check_policy_causality(True)

    def check_policy_causality(self, query_only):
        torch.manual_seed(12)
        cnn,bank,initial,a,domain,anchors,parent = fixtures.FinalEnhancementTests().fixture(query='policy',policy='learned')
        bank = bank[:1]; a.train_epsilon = 1.
        start = deepcopy(parent.policy.state_dict())
        def run(labels):
            torch.manual_seed(71)
            agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start,slots=1,query_value_only=query_only)
            continuation={}
            metrics,traces=episode(a,cnn,bank,initial,agent,(domain[0],labels,domain[2]),anchors,
                training=True,paid_supervision=True,continuation=continuation)
            return metrics,traces,deepcopy(agent.policy.state_dict()),deepcopy(bank[0].state_dict()),continuation
        m,trace,policy,context,continuation=run(domain[1])
        self.assertGreater(m['labels'],0)
        self.assertLessEqual(m['labels'],m['label_cap'])
        self.assertEqual(m['route_updates'],0 if query_only else m['labels'])
        self.assertEqual(len(continuation['history'].updates),2)
        altered=domain[1].copy(); paid=set(m['queried_source_ids'])
        for i,identity in enumerate(domain[2]):
            if identity not in paid: altered[i]=1-altered[i]
        _,again,policy2,context2,_=run(altered)
        self.assertEqual([r['action'] for r in trace],[r['action'] for r in again])
        for left,right in ((policy,policy2),(context,context2)):
            for key in left: self.assertTrue(torch.equal(left[key],right[key]),key)

    def test_real_cli_train_select_and_both_evaluations(self):
        self.check_real_cli(False)

    def test_real_cli_query_value_only_train_and_both_evaluations(self):
        self.check_real_cli(True)

    def test_real_cli_paired_only_train_and_both_evaluations(self):
        self.check_real_cli(True,paired=True)

    def test_real_cli_connected_long_horizon_and_both_evaluations(self):
        self.check_real_cli(True,paired=True,connected=True,gamma=.99)

    def check_real_cli(self, query_only, paired=False, connected=False, gamma=.95):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; models=source/'Models'; models.mkdir(parents=True)
            burst=root/'burst'; burst.mkdir(); output=root/'specialized'
            rng=np.random.default_rng(42); raw=rng.normal(size=(104,8)).astype(np.float32)
            labels=np.arange(104)%2; features=[f'f{i}' for i in range(8)]
            for level in (0,1,5,7):
                path=burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
                frame=pd.DataFrame(raw+level*.02,columns=features); frame['label']=labels
                frame.to_csv(path,index=False)
                pd.DataFrame({'source_row':np.arange(104)}).to_csv(path.with_name(path.stem+'_rows.csv'),index=False)
            artifact=dict(feature_names=features,label_name='label',source_rows=104,source_labels=labels,
                train_ids=np.arange(64),val_ids=np.arange(64,84),test_ids=np.arange(84,104),
                scaler=StandardScaler().fit(raw[:64]),label_encoder=LabelEncoder().fit(labels),
                source_sha256=legacy.core.file_sha256(burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv'))
            joblib.dump(artifact,models/'protocol_preprocessing.joblib')
            cnn=legacy.core.CNN1D(8,2); context=legacy.core.ContextAwareNetwork(8)
            torch.save({'model_state_dict':cnn.state_dict()},models/'initial_cnn_model.pth')
            torch.save({'context_model_state_dict':context.state_dict()},models/'adversarial_best_context.pth')
            torch.save({'initial_state_dict':context.state_dict()},models/'enhanced_context.pth')
            config=EnhancementConfig.preset('accuracy')
            parent=EnhancementAgent(150,128,2,torch.device('cpu'),42,config)
            with torch.no_grad():
                for parameter in parent.policy.update_value.parameters(): parameter.zero_()
                parent.policy.update_value[-1].bias.fill_(2.)
            hashes={name:legacy.core.file_sha256(models/name) for name in (*runner.SOURCE_FILES,'enhanced_context.pth')}
            parent_state=deepcopy(parent.policy.state_dict())
            torch.save(dict(format_version=1,config=config.as_dict(),training_levels=[0,1],model_hashes=hashes,
                training_domain_sha256={str(k):legacy.core.file_sha256(burst/f'AdvBurstTrainedBurst_size_{k}_Step_2000.csv') for k in (0,1)},
                policy_state_dict=parent_state,prerouter_state_dict=parent.pre_router.state_dict()),models/'final_enhancement.pth')
            source_hash=legacy.core.file_sha256(models/'final_enhancement.pth')
            common=['--output',str(output),'--burst-dir',str(burst),'--device','cpu',
                '--context-count','1','--query-schedule','policy','--update-policy','learned',
                '--budget-fraction','0.5','--eval-epsilon','0.05','--gamma',str(gamma)]
            original=runner.episode; train_stages=[]; trained_states=[]
            def checked_episode(*args,**kwargs):
                a,_,bank,_,agent,domain=args[:6]
                self.assertEqual(len(bank),1); self.assertEqual(agent.slots,1)
                self.assertEqual(a.query_schedule,'policy')
                ids=set(map(int,domain[2]))
                self.assertFalse(ids & set(artifact['test_ids']) if a.command=='train' else False)
                if kwargs.get('training'):
                    self.assertTrue(kwargs['paid_supervision'])
                    self.assertEqual(agent.query_value_only,query_only)
                    self.assertEqual(agent.paired_query_only,paired)
                    if query_only:
                        expected={id(p) for p in agent.policy.update_value.parameters()}
                        actual={id(p) for group in agent.td_optimizer.param_groups for p in group['params']}
                        self.assertEqual(expected,actual)
                        for name,parameter in agent.policy.named_parameters():
                            self.assertEqual(parameter.requires_grad,name.startswith('update_value.'))
                    continuing=bool(kwargs.get('continuation'))
                    if continuing:
                        self.assertGreater(kwargs['continuation']['history'].paid,0)
                        self.assertTrue(kwargs['continuation']['optimizers'][0].state)
                    train_stages.append(continuing)
                    self.assertTrue(ids.issubset(set(artifact['train_ids'])))
                else:
                    self.assertFalse(agent.query_value_only)
                    self.assertTrue(all(p.requires_grad for p in agent.policy.parameters()))
                result=original(*args,**kwargs)
                if kwargs.get('training'): trained_states.append(deepcopy(agent.policy.state_dict()))
                return result
            with redirect_stdout(io.StringIO()), patch.object(runner,'episode',checked_episode), \
                    patch('enhancements.sequence.episode',checked_episode), \
                    patch('enhancements.stream.alternatives',side_effect=AssertionError('No counterfactual simulation')):
                runner.main(['train',*common,'--source-run',str(source),'--profile','accuracy',
                    '--warm-start-enhancement','--aligned-sequence','--one-context-rl-training',
                    '--levels','0','1','--epochs','1',
                    *(['--query-value-only-training'] if query_only else []),
                    *(['--paired-query-only-training'] if paired else []),
                    *(['--connected-sequence-td'] if connected else [])])
            self.assertEqual(train_stages,[False,True])
            saved=torch.load(output/'Models'/'final_enhancement.pth',weights_only=False)
            report=json.loads((output/'enhancement_training.json').read_text())
            protocol=saved['training_protocol']
            self.assertEqual(saved['training_context_count'],1)
            self.assertEqual(saved['parent_checkpoint_sha256'],source_hash)
            self.assertEqual(protocol['query_schedule'],'policy')
            self.assertEqual(protocol['context_count'],1)
            self.assertFalse(protocol['selector_specialization']['damage_guard'])
            audits=set(protocol['validation_audit_source_ids'])
            for ids in protocol['validation_adaptation_source_ids'].values(): self.assertFalse(audits & set(ids))
            trained=report['epochs'][1]
            self.assertGreater(sum(r['td_updates'] for r in trained['train_stages']),0)
            self.assertTrue(all(r['route_updates']==(0 if query_only else r['labels']) for r in trained['train_stages']))
            if paired:
                self.assertEqual(protocol['temporal_credit']['connected_levels'],connected)
                self.assertEqual(protocol['temporal_credit']['gamma'],gamma)
                for stage in trained['train_stages']:
                    d=stage['paired_target_diagnostics']
                    self.assertGreater(d['sampled_rows'],0)
                    self.assertAlmostEqual(d['target_mean'],d['utility_mean']+d['future_mean'],places=5)
            if query_only:
                self.assertEqual(protocol['selector_specialization']['trained_parameters'],'update_value only')
                self.assertTrue(all(row['frozen_selector_unchanged'] for row in report['epochs']))
                for key in parent_state:
                    if not key.startswith('update_value.'):
                        self.assertTrue(torch.equal(parent_state[key],trained_states[-1][key]),key)
                        self.assertTrue(torch.equal(parent_state[key],saved['policy_state_dict'][key]),key)
                self.assertTrue(any(not torch.equal(parent_state[key],trained_states[-1][key])
                    for key in parent_state if key.startswith('update_value.')))
            self.assertLessEqual(saved['selected_epoch'],1)
            digest=legacy.core.file_sha256(output/'Models'/'final_enhancement.pth')
            with redirect_stdout(io.StringIO()):
                runner.main(['evaluate',*common,'--levels','5','7','--modes','learned'])
                runner.main(['evaluate',*common,'--levels','5','7','--modes','learned','--continual'])
            results=list((output/'Results').glob('evaluation_*')); self.assertEqual(len(results),2)
            kinds=set()
            for result in results:
                execution=json.loads((result/'execution.json').read_text())
                kinds.add(execution['continual'])
                self.assertEqual(execution['checkpoint_sha256'],digest)
                self.assertEqual(execution['training_protocol'],protocol)
                self.assertIn('dedicated',execution['context_count_protocol'])
                self.assertNotIn('context_damage_guard',execution)
                if paired:
                    self.assertEqual(execution['temporal_credit']['gamma'],gamma)
                    self.assertEqual(execution['temporal_credit']['active_boundary_links'],connected and execution['continual'])
                if connected:
                    trace=pd.read_csv(result/'learned_level5_trace.csv')
                    self.assertEqual(bool(trace.iloc[-1]['td_boundary_successor']),execution['continual'])
                    self.assertEqual(bool(trace.iloc[-1]['td_terminal']),not execution['continual'])
                    final=pd.read_csv(result/'learned_level7_trace.csv')
                    self.assertTrue(bool(final.iloc[-1]['td_terminal']))
                for level in (5,7):
                    metrics=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertLessEqual(metrics['labels'],metrics['label_cap'])
                    self.assertTrue(metrics['classifier_unchanged'])
            self.assertEqual(kinds,{False,True})
            with self.assertRaisesRegex(ValueError,'Context count'):
                runner.main(['evaluate',*common,'--context-count','3','--levels','5','7'])
            self.assertEqual(legacy.core.file_sha256(models/'final_enhancement.pth'),source_hash)
            self.assertEqual(legacy.core.file_sha256(output/'Models'/'final_enhancement.pth'),digest)
            for name,value in hashes.items(): self.assertEqual(legacy.core.file_sha256(output/'Models'/name),value)


if __name__=='__main__': unittest.main()
