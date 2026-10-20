import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_supplied_attack_protocol as protocol

class SuppliedTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(x=np.arange(600,dtype=float),label=np.repeat(['a','b','c','d','e','f'],100)))

    def test_source_and_attack_partitions_are_disjoint_and_share_ids(self):
        frame=self.frame();pool,parts=protocol.source_pools(frame)
        groups=[set(parts[k]) for k in ('train','validation','test')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(set.union(*groups),set(range(600)))
        self.assertEqual([len(set(pool)&g) for g in groups],[72,24,24])
        self.assertTrue(groups[2]<=set(pool))
        for group in groups:self.assertEqual(set(frame.iloc[sorted(group),-1]),set('abcdef'))

    def test_preparation_copies_existing_attacks_without_modification(self):
        frame=self.frame();pool,_=protocol.source_pools(frame)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';(data/'CW').mkdir(parents=True)
            original=data/'original.csv';frame.to_csv(original,index=False)
            for radius in protocol.LEVELS['CW']:
                attack=frame.iloc[pool].copy();attack['x']+=radius
                attack.to_csv(data/'CW'/f'CW_eps_{radius}.csv',index=False)
            a=SimpleNamespace(output=root/'run',data_dir=data,source_file='original.csv',attack='CW',seed=42)
            manifest=protocol.prepare(a)
            self.assertFalse(manifest['attack_generation'])
            self.assertEqual(manifest['training_levels'],[0,2])
            target=a.output/'Domains'/protocol.domain_name(2)
            self.assertEqual(target.read_bytes(),(data/'CW/CW_eps_0.5.csv').read_bytes())
            mapping=pd.read_csv(target.with_name(target.stem+'_rows.csv'))
            self.assertEqual(mapping.source_row.tolist(),pool.tolist())
            self.assertIn('ASSUMED',manifest['alignment'])
            with self.assertRaises(ValueError):protocol.prepare(a)

    def test_independent_baselines_never_require_amcal_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'supplied_attack_training.json').write_text(json.dumps(dict(purpose='independent_baselines')))
            a=SimpleNamespace(output=root,command='train-baselines',independent_baselines=True,
                              methods=['cnn','arf'],budget_fraction=.2,seed=42,device='cuda')
            with patch.object(protocol,'independent_baseline_source',return_value=root/'Domains') as prepare, patch.object(protocol,'check') as check, patch.object(protocol.subprocess,'run') as run:
                protocol.baseline(a)
            prepare.assert_called_once()
            check.assert_not_called()
            command=run.call_args.args[0]
            self.assertIn('--independent-source',command)
            self.assertEqual(command[command.index('--source-run')+1],str(root/'BaselineSource'))
            self.assertFalse((root/'Bank').exists())


    def test_full_unseen_keeps_entire_file_and_seen_keeps_holdout(self):
        from baselines.adversarial_domains import evaluation_row_ids
        available=np.arange(1200)
        heldout=np.arange(960,1200)
        full=evaluation_row_ids(available,heldout,True)
        seen=evaluation_row_ids(available,heldout,False)
        self.assertEqual(full.tolist(),list(range(1200)))
        self.assertEqual(seen.tolist(),list(range(960,1200)))
        self.assertEqual(int(.2*len(full)),240)
        self.assertEqual(int(.2*len(seen)),48)


if __name__=='__main__':unittest.main()
