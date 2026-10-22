"""A directory alone is not evidence of a complete, reusable evaluation."""
from pathlib import Path
import json
import tempfile
import unittest
import amcal_final_enhancement as runner


class EvaluationCompletionTests(unittest.TestCase):
    def test_matches_metadata_and_rejects_missing_or_truncated_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            execution=dict(modes=['learned'],levels=[5],continual=False,seed=42)
            self.assertFalse(runner.completed_evaluation(root,execution))
            (root/'execution.json').write_text(json.dumps(execution))
            (root/'combined_results.csv').write_text('Level,Mode,Accuracy\n5,learned,0.5\n')
            metric=dict(mode='learned',level=5,samples=2,labels=1,label_cap=1)
            (root/'learned_level5_metrics.json').write_text(json.dumps(metric))
            trace=root/'learned_level5_trace.csv'
            trace.write_text('source_row,prediction,queried\n1,0,True\n2,1,False\n')
            self.assertTrue(runner.completed_evaluation(root,execution))
            self.assertFalse(runner.completed_evaluation(root,dict(execution,seed=43)))
            trace.write_text('source_row,prediction,queried\n1,0,True\n')
            self.assertFalse(runner.completed_evaluation(root,execution))
            (root/'execution.json').write_text('[]')
            self.assertFalse(runner.completed_evaluation(root,execution))
            (root/'execution.json').write_text('{')
            self.assertFalse(runner.completed_evaluation(root,execution))


if __name__=='__main__':unittest.main()
