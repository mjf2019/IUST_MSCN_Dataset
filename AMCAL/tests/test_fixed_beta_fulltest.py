"""User-run candidate reuse and provenance tests; no real training or inference."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_fixed_beta_fulltest as runner


class FullBetaChecks(unittest.TestCase):
    def test_candidate_search_reuses_primary_then_extra_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            runs=Path(temp); own=runs/'fixed_beta_fulltest_seed42_v1'
            roots=runner.candidate_roots(runs,42,own,10.)
            self.assertIsNone(runner.locate_candidate(runs,42,own,10.))
            roots[1].joinpath('Models').mkdir(parents=True)
            roots[1].joinpath('Models/two_action_ddqn.pth').write_bytes(b'fixture')
            # A model without a completion record is not reused.
            self.assertIsNone(runner.locate_candidate(runs,42,own,10.))
            roots[1].joinpath('training_complete.json').write_text('{}')
            self.assertEqual(runner.locate_candidate(runs,42,own,10.),roots[1])
            roots[0].joinpath('Models').mkdir(parents=True)
            roots[0].joinpath('Models/two_action_ddqn.pth').write_bytes(b'fixture')
            roots[0].joinpath('training_complete.json').write_text('{}')
            self.assertEqual(runner.locate_candidate(runs,42,own,10.),roots[0])

    def test_matching_source_recipe_is_required_for_every_beta(self):
        parent=dict(dataset='IAT',seed=42,budget_fraction=.2,context_lr=.00025,source_models={'CNN':'fixture'})
        for beta in runner.BETAS:
            protocol=dict(protocol=deepcopy(parent),beta=beta)
            saved=dict(protocol=protocol); done=dict(protocol=deepcopy(protocol))
            self.assertEqual(runner.validate_candidate(saved,done,parent,beta),protocol)
            with self.assertRaisesRegex(ValueError,'beta/protocol'):
                runner.validate_candidate(saved,done,parent,beta+1)
            changed=deepcopy(parent); changed['source_models']={'CNN':'different'}
            with self.assertRaisesRegex(ValueError,'source_models'):
                runner.validate_candidate(saved,done,changed,beta)
            changed=deepcopy(parent); changed['budget_fraction']=.3
            with self.assertRaisesRegex(ValueError,'budget_fraction'):
                runner.validate_candidate(saved,done,changed,beta)

    def test_full_test_summary_uses_all_sample_and_label_counts(self):
        rows=[dict(dataset='IAT',beta=5.,samples=968,labels=193,**{
            key:1. for key in ('AC','PR','RC','F1','labels_percent','budget_used_percent',
                'target_final_AC','target_final_PR','target_final_RC','target_final_F1','source_final_AC','forgetting')})
            for _ in range(7)]
        summary=runner.report_mean(rows)
        self.assertEqual(summary['total_samples'],6776)
        self.assertEqual(summary['total_labels'],1351)
        self.assertEqual(summary['levels'],7)
        self.assertEqual(summary['budget_percent'],20)


if __name__=='__main__': unittest.main()

