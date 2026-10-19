"""Check shared-data dispatch without requiring PyTorch or lengthy model training."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_adversarial_baselines as runner


class BaselineDispatchTests(unittest.TestCase):
    def options(self, root, command):
        return SimpleNamespace(command=command, attack='all', amcal_run=root/'amcal',
                               output=root/'baselines', data_dir=root/'data', source_file='source.csv',
                               levels=[0,1,2,3,4,5], seed=42, device='cuda', methods=list(runner.METHODS),
                               budget_fraction=.2, jobs=-1, dfe_epochs=200, af_epochs=30,
                               patience=10, batch_size=32)

    def test_training_reuses_exact_domains_and_bank_cnn_without_evaluation(self):
        a=self.options(Path('/tmp/test'), 'train')
        with patch.object(runner,'validate_run',return_value={}), patch.object(runner.protocol,'prepare_evaluation_domains') as prepare:
            command, output=runner.command_for(a,'PGD')
        prepare.assert_not_called()
        self.assertEqual(command[command.index('--burst-dir')+1],str(a.amcal_run/'PGD/TrainingDomains'))
        self.assertEqual(command[command.index('--source-run')+1],str(a.amcal_run/'PGD/Bank'))
        self.assertEqual(command[command.index('--train-levels')+1:command.index('--train-levels')+3],['0','2'])
        self.assertEqual(output,a.output/'PGD')

    def test_evaluation_keeps_training_and_test_files_separate(self):
        a=self.options(Path('/tmp/test'),'evaluate')
        manifest={'training_settings':{'context_lr':.0005}}
        with patch.object(runner,'validate_run',return_value=manifest), patch.object(runner.protocol,'prepare_evaluation_domains',return_value=Path('/tmp/evaluation')):
            command,_=runner.command_for(a,'CW')
        self.assertEqual(command[command.index('--burst-dir')+1],'/tmp/evaluation')
        self.assertEqual(command[command.index('--training-domain-dir')+1],str(a.amcal_run/'CW/TrainingDomains'))
        self.assertIn('--include-seen-levels',command)

    def test_aggregate_reads_execution_seed_and_method_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            a=self.options(Path(directory),'evaluate')
            result=a.output/'PGD/Results';result.mkdir(parents=True)
            metrics=dict(execution_config=dict(seed=42,budget=.2),level=2,method='River-ARFClassifier',
                         reference=dict(drop_pp=.5),accuracy=.8,macro_f1=.7,labels=239,label_cap=239,actual_label_fraction=.2)
            (result/'arf_per2_seed42_tag_metrics.json').write_text(json.dumps(metrics))
            runner.aggregate(a,['PGD'])
            with (a.output/'Results/all_baselines_seed42.csv').open(encoding='utf-8-sig') as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['Method'],'arf')
            self.assertEqual(rows[0]['Domain'],'Seen')


if __name__=='__main__':unittest.main()
