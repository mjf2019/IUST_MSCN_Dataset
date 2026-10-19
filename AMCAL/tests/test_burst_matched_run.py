"""Check PS dataset routing and strict separation of training and testing."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_matched_run as m


class MatchedRunTests(unittest.TestCase):
    def fixture(self,root,command):
        return SimpleNamespace(command=command,representation='PS',seed=42,device='cuda',
            dataset=root/'Dataset',source_run=root/'source',amcal_run=root/'bank',baseline_run=root/'baselines')

    def test_train_then_test_keeps_ps_source_and_heldout_domains_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            a=self.fixture(Path(directory),'train');train=m.commands(a)
            self.assertEqual(len(train),3)
            self.assertTrue(all(c[1]=='train' for c in train))
            self.assertTrue(all('AdvBurst_FTSC-PS' in c[c.index('--burst-dir')+1] for c in train))
            self.assertIn('--stage2-pretrain-epochs',train[0])
            self.assertEqual(train[0][train[0].index('--multishift-levels')+1:train[0].index('--multishift-levels')+4],['0','1','3'])
            a.source_run.mkdir()
            self.assertEqual(len(m.commands(a)),2)
            a.command='evaluate';evaluation=m.commands(a)
            self.assertEqual(len(evaluation),2)
            self.assertTrue(all(c[1]=='evaluate' for c in evaluation))
            for c in evaluation:
                i=c.index('--levels')
                self.assertEqual(c[i+1:i+8],['5','7','10','12','15','17','20'])
            self.assertEqual(evaluation[0][evaluation[0].index('--eval-epsilon')+1],'0.05')

    def test_failed_child_stops_sequence_and_prepared_data_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=self.fixture(root,'train')
            a.dataset.mkdir();(a.dataset/'official_partitions.json').write_text('{}')
            burst=a.dataset/'AdvBurst_FTSC-PS';burst.mkdir()
            for level in (0,1,3,5,7,10,12,15,17,20):
                stem=f'AdvBurstTrainedBurst_size_{level}_Step_2000'
                for suffix in ('.csv','_rows.csv'):(burst/(stem+suffix)).touch()
            argv=['runner','train','--representation','PS','--dataset',str(a.dataset),
                  '--source-run',str(a.source_run),'--amcal-run',str(a.amcal_run),
                  '--baseline-run',str(a.baseline_run)]
            with patch.object(sys,'argv',argv),patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=7)) as run:
                with self.assertRaises(SystemExit) as error:m.main()
            self.assertEqual(error.exception.code,7)
            self.assertEqual(run.call_count,1)
            self.assertIn('amcal_burst_two_stage_protocol.py',run.call_args.args[0][1])


if __name__=='__main__':unittest.main()
