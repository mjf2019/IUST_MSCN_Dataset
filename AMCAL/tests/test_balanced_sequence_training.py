"""Original balanced v1: persistent paid-feedback training and VAL audits."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as bank
import amcal_balanced_sequence_training as runner
import amcal_burst_sequential_retention as audit_runner
import test_structured_bank_protocol as fixtures


class BalancedSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); bank.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def fixture(self):
        cnn, contexts, initial, a, domain = fixtures.StructuredBankTests().fixture()
        a.update_policy = 'learned'; a.balanced_reward_reference = True
        a.gamma = .95; a.selection_cost = .0001; a.gain_weight = a.retention_weight = 1.
        agent = bank.Agent(34, bank.core.device, 42)
        return cnn, contexts, initial, a, domain, agent

    def test_original_memory_and_optimizer_survive_domain_transition(self):
        cnn, contexts, initial, a, domain, agent = self.fixture()
        continuation = {}
        bank.episode(a, cnn, contexts, initial, agent, domain, continuation=continuation)
        memory = continuation['reward_reference']; optimizers = continuation['optimizers']
        self.assertIsInstance(memory, bank.BalancedRewardReference)
        self.assertIsInstance(memory.source, bank.core.PaidHistory)
        before_steps = [max((float(s['step']) for s in o.state.values()), default=0) for o in optimizers]
        bank.episode(a, cnn, contexts, initial, agent, domain, continuation=continuation)
        self.assertIs(memory, continuation['reward_reference'])
        self.assertIs(optimizers, continuation['optimizers'])
        self.assertEqual(continuation['history'].paid, 16)
        self.assertEqual(agent.route_updates, 16)
        after_steps = [max((float(s['step']) for s in o.state.values()), default=0) for o in optimizers]
        self.assertGreater(sum(after_steps), sum(before_steps))

    def test_train_rollout_does_not_learn_from_unpaid_labels(self):
        cnn, contexts, initial, a, domain, agent = self.fixture()
        start = deepcopy(agent.policy.state_dict())
        stages, _ = runner.run_sequence(a, cnn, contexts, initial, agent, {0:domain}, [0], 42)
        self.assertEqual(stages[0]['route_updates'], stages[0]['labels'])
        after = deepcopy(agent.policy.state_dict())
        order = np.random.default_rng(42).permutation(len(domain[1]))
        positions = np.random.default_rng(42+819003).choice(16, 8, replace=False)
        paid = set(order[positions]); y = domain[1].copy()
        for i in range(len(y)):
            if i not in paid: y[i] = 1-y[i]
        other = bank.Agent(34, bank.core.device, 42, policy_state=start)
        runner.run_sequence(a, cnn, contexts, initial, other, {0:(domain[0],y,domain[2])}, [0], 42)
        for k, v in after.items(): self.assertTrue(torch.equal(v, other.policy.state_dict()[k]))

    def test_validation_audit_labels_do_not_change_adaptation(self):
        cnn, contexts, initial, a, domain, agent = self.fixture()
        streams, audits, clean, ids = runner.split_validation({0:domain,1:domain}, domain, 42, .2)
        for d in streams.values(): self.assertFalse(set(ids) & set(d[2]))
        start = deepcopy(agent.policy.state_dict()); collected = []
        original_episode = bank.episode
        def episode(*args, **kwargs):
            result = original_episode(*args, **kwargs)
            collected.append(deepcopy(args[4].policy.state_dict()))
            return result
        with patch.object(bank, 'models_from_source', return_value=(cnn, contexts, initial, 34)), patch.object(bank, 'episode', episode):
            first = runner.validation_sequence(a, None, None, start, streams, audits, clean, [0,1])
            policy_after = collected[-1]
            flipped = {k:(x,1-y) for k,(x,y) in audits.items()}
            runner.validation_sequence(a, None, None, start, streams, flipped, (clean[0],1-clean[1]), [0,1])
            for k, v in policy_after.items(): self.assertTrue(torch.equal(v, collected[-1][k]))
        self.assertEqual(len(first['audit_accuracy_by_stage']), 3)

    def test_real_original_checkpoint_training_and_sequential_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; models = source/'Models'; models.mkdir(parents=True)
            burst = root/'burst'; burst.mkdir()
            raw = np.random.default_rng(42).normal(size=(80,8)).astype(np.float32)
            y = np.arange(80)%2; features = [f'f{i}' for i in range(8)]
            for level in (0,1,5,7):
                path = burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
                frame = pd.DataFrame(raw+level*.02, columns=features); frame['label'] = y
                frame.to_csv(path, index=False)
                pd.DataFrame({'source_row':np.arange(80)}).to_csv(path.with_name(path.stem+'_rows.csv'), index=False)
            artifact = dict(feature_names=features,label_name='label',source_rows=80,source_labels=y,
                train_ids=np.arange(40),val_ids=np.arange(40,60),test_ids=np.arange(60,80),
                scaler=StandardScaler().fit(raw[:40]),label_encoder=LabelEncoder().fit(y),
                source_sha256=bank.core.file_sha256(burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv'))
            joblib.dump(artifact, models/'protocol_preprocessing.joblib')
            cnn = bank.core.CNN1D(8,2); context = bank.core.ContextAwareNetwork(8)
            torch.save({'model_state_dict':cnn.state_dict()},models/'initial_cnn_model.pth')
            torch.save({'context_model_state_dict':context.state_dict()},models/'adversarial_best_context.pth')
            hashes = {n:bank.core.file_sha256(models/n) for n in runner.SOURCE_FILES}
            provenance = {str(l):bank.core.file_sha256(burst/f'AdvBurstTrainedBurst_size_{l}_Step_2000.csv') for l in (0,1)}
            config = dict(balanced_reward_reference=True, decoupled_query=False, selection_cost=.0001,
                gain_weight=1.,retention_weight=1.,gamma=.95,train_epsilon=.2,context_alpha=.8)
            torch.save(dict(version=1,route_width=150,policy_state_dict=bank.StructuredQ(150).state_dict(),
                model_hashes=hashes,training_levels=[0,1],training_domain_sha256=provenance,config=config),
                models/'structured_bank_ddqn.pth')
            parent_hash = bank.core.file_sha256(models/'structured_bank_ddqn.pth')
            output = root/'balanced'; observed_splits = []; read_original = bank.read_domain
            def read(*args, **kwargs):
                observed_splits.append(args[3]); return read_original(*args, **kwargs)
            with redirect_stdout(io.StringIO()), patch.object(bank, 'read_domain', read):
                runner.main(['--source-run',str(source),'--output',str(output),'--burst-dir',str(burst),
                    '--levels','0','1','--epochs','1','--device','cpu'])
            self.assertNotIn('test_ids', observed_splits)
            saved = torch.load(output/'Models'/'structured_bank_ddqn.pth',weights_only=False)
            self.assertEqual(saved['version'],1)
            self.assertEqual(saved['parent_checkpoint_sha256'],parent_hash)
            self.assertTrue(saved['config']['balanced_reward_reference'])
            self.assertFalse(any(k.startswith('comparison.') for k in saved['policy_state_dict']))
            self.assertEqual(bank.core.file_sha256(models/'structured_bank_ddqn.pth'),parent_hash)
            for name,digest in hashes.items(): self.assertEqual(bank.core.file_sha256(output/'Models'/name),digest)
            log = json.loads((output/'structured_sequence_training.json').read_text())
            self.assertEqual(log['epochs'][0]['train_labels'],0)
            for stage in log['epochs'][1]['train_stages']:
                self.assertEqual(stage['labels'],stage['cap'])
                self.assertEqual(stage['route_updates'],stage['labels'])
            protocol = log['training_protocol']; ids = set(protocol['validation_audit_source_ids'])
            for adaptation in protocol['validation_adaptation_source_ids'].values(): self.assertFalse(ids & set(adaptation))
            with redirect_stdout(io.StringIO()), patch.object(sys,'argv',['audit','--source-run',str(output),
                    '--output',str(output/'Results'),'--burst-dir',str(burst),'--levels','5','7',
                    '--modes','learned','--query-schedule','uniform','--device','cpu']):
                audit_runner.main()
            self.assertTrue((output/'Results'/'combined_sequential_summary.csv').is_file())
            metadata = json.loads((output/'Results'/'sequential_config.json').read_text())
            self.assertEqual(metadata['query_schedule'],'uniform')


if __name__ == '__main__': unittest.main()
