import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
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

if __name__=='__main__':unittest.main()
