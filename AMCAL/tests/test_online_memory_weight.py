"""Online replay CE ablation changes Context updates, not label availability."""
from copy import deepcopy
from dataclasses import replace
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
import amcal_final_enhancement as runner
from enhancements.agent import EnhancementAgent
from enhancements.stream import episode
import test_final_enhancement as fixtures


class OnlineMemoryWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); legacy.core.device=torch.device('cpu')

    def test_weight_changes_context_updates_with_same_labels_and_memory_capacity(self):
        torch.manual_seed(12)
        cnn,bank,initial,a,domain,anchors,source=fixtures.FinalEnhancementTests().fixture(policy='learned')
        frozen=deepcopy(cnn.state_dict()); state=deepcopy(source.policy.state_dict())
        results=[]
        for weight in (.1,.3):
            a.config=replace(a.config,memory_ce_weight=weight)
            current=[deepcopy(bank[0])]
            agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=state,slots=1)
            m,trace=episode(a,cnn,current,initial,agent,domain,anchors)
            results.append((m,trace,deepcopy(current[0].state_dict())))
            for key,value in frozen.items(): self.assertTrue(torch.equal(value,cnn.state_dict()[key]))
        low,high=results
        self.assertEqual(low[0]['queried_source_ids'],high[0]['queried_source_ids'])
        self.assertEqual(low[0]['memory'],high[0]['memory'])
        self.assertEqual(low[0]['labels'],low[0]['label_cap'])
        self.assertEqual(high[0]['labels'],high[0]['label_cap'])
        self.assertEqual([r['queried'] for r in low[1]],[r['queried'] for r in high[1]])
        self.assertTrue(any(not torch.equal(value,high[2][key]) for key,value in low[2].items()))

    def test_invalid_weights_and_unmatched_experiments_are_rejected_before_loading(self):
        common=['evaluate','--output','unused','--burst-dir','unused','--context-count','1',
                '--query-schedule','uniform']
        for value in ('-1','nan','inf'):
            with self.subTest(value=value),self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):
                runner.main([*common,'--online-memory-ce-weight',value])
        for extra in (['--context-count','2'],['--query-schedule','policy'],['--router-refresh']):
            with self.subTest(extra=extra),self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):
                runner.main([*common,'--online-memory-ce-weight','.3',*extra])


if __name__=='__main__': unittest.main()
