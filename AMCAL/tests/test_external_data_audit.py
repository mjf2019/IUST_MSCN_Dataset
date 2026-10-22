import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amcal_external_data_audit import audit, audit_file, excluded, roles_and_targets


class ExternalDataAuditTests(unittest.TestCase):
    def test_npbo_and_tun_labels_and_metadata(self):
        for col in ('label', 'label_3'):
            frame = pd.DataFrame({col: [3, 1, 2], 'label_2': [False, True, True],
                                  'label_poly': ['3_Apache', '1_Apache', '2_Apache']})
            role, y = roles_and_targets(frame, 'ASNM_NPBO_TEST', Path('x.csv'))
            self.assertEqual(role.tolist(), ['benign', 'direct', 'adversarial'])
            self.assertEqual(y.tolist(), [0, 1, 1])
            frame.loc[2, 'label_2'] = False
            with self.assertRaisesRegex(ValueError, 'disagree'): roles_and_targets(frame, 'ASNM_NPBO_TEST', Path('x.csv'))
        for col in ('label', 'label_2', 'label_poly_o', 'id', 'Unnamed: 0', 'n_actions', 'srcIP', 'SrcMAC', 'TSesStart'):
            self.assertTrue(excluded(col), col)
        self.assertFalse(excluded('srcIPInVlan'))
        self.assertFalse(excluded('MeanPktLenIn'))

    def test_drelab_path_and_adversarial_target(self):
        frame = pd.DataFrame({'Label': ['adversarial', '1']})
        role, y = roles_and_targets(frame, 'DReLAB_TEST', Path('CTU/RF/DDQN/Neris.csv'))
        self.assertEqual(role.tolist(), ['adversarial']*2); self.assertEqual(y.tolist(), [1, 1])
        frame.loc[0, 'Label'] = '0'
        with self.assertRaises(ValueError): roles_and_targets(frame, 'DReLAB_TEST', Path('CTU/RF/DDQN/Neris.csv'))

    def test_full_csv_not_only_first_chunk_and_no_raw_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root/'AMCAL/DReLAB_TEST/Dataset_official_v1/DReLAB/CTU/RF/DDQN/Neris.csv'
            d.parent.mkdir(parents=True)
            pd.DataFrame({'Unnamed: 0': [0,1,2], 'Dur': [1,2,3], 'n_actions': [4,5,6],
                          'Label': ['adversarial']*3}).to_csv(d,index=False)
            for name, col in [('ASNM_NPBO_TEST','label'), ('ASNM_TUN_TEST','label_3')]:
                path=root/'AMCAL'/name/'Dataset_official_v1'/'x.csv'
                path.parent.mkdir(parents=True)
                pd.DataFrame({'id':[1,2,3],col:[3,1,2], 'label_2':[False,True,True],
                    'srcIP':['a','b','c'], 'F':[1,2,float('inf')]}).to_csv(path,index=False,sep=';')
            before={p: p.read_bytes() for p in root.rglob('*.csv')}
            with contextlib.redirect_stdout(io.StringIO()): result=audit(root,chunksize=1)
            self.assertTrue(all(x['ready_for_split_design'] for x in result['datasets']))
            files=[x['files'][0] for x in result['datasets']]
            self.assertEqual(files[0]['candidate_features'], ['Dur'])
            self.assertEqual(files[0]['rows'], 3)
            self.assertEqual(files[1]['roles'], {'benign':1,'direct':1,'adversarial':1})
            self.assertEqual(files[1]['infinite_values'], {'F':1})
            self.assertEqual(files[1]['separator'], ';')
            for path, content in before.items(): self.assertEqual(path.read_bytes(),content)

    def test_missing_files_report_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()): result=audit(Path(tmp))
            self.assertTrue(all(x['errors'] for x in result['datasets']))


if __name__ == '__main__': unittest.main()
