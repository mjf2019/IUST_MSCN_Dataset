import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amcal_external_data_audit import audit, audit_file, excluded, roles_and_targets, separator


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
        for col in ('label', 'label_2', 'label_poly_o', 'id', 'Unnamed: 0', 'n_actions', 'srcIP', 'SrcMAC', 'TSesStart', 'IPSrcType', 'IPDstType',
                    'srcIPInVlan', 'DstIPInVlan', 'source_ip', 'destination_ip', 'Flow ID', 'Timestamp', 'srcPort'):
            self.assertTrue(excluded(col), col)
        self.assertTrue(excluded('srcIPInVlan'))
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

    def test_wide_complete_records_and_quoted_delimiter(self):
        with tempfile.TemporaryDirectory() as tmp:
            for delimiter in (',', ';', '\t'):
                path = Path(tmp)/'wide.csv'
                # A single record exceeds the old 32768-byte sample.
                columns = [f'feature_{i:04d}_long_name' for i in range(904)]
                frame = pd.DataFrame([['a,b;c']*904, ['z'*50]*904], columns=columns)
                frame.to_csv(path, index=False, sep=delimiter)
                self.assertEqual(separator(path), delimiter)

    def test_auxiliary_metadata_is_diagnostic_not_a_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'npbo.csv'
            pd.DataFrame({'label':[1,2,3], 'label_2':[True,True,False],
                'label_poly_o':['1_Apache','2a_variant_Apache','3_Other'],
                'F':[1,2,3]}).to_csv(path,index=False)
            result = audit_file(path,'ASNM_NPBO_TEST',root,chunksize=1)
            self.assertEqual(result['binary_classes'], {'1':2,'0':1})
            self.assertEqual(result['auxiliary_label_diagnostics']['label_poly_o']['prefix_mismatch_rows'],1)
            self.assertEqual(result['candidate_features'],['F'])
            self.assertTrue(result['warnings'])

    def test_dataset_filter_preserves_existing_report_entries(self):
        import json
        from unittest.mock import patch
        from amcal_external_data_audit import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path=root/'AMCAL/ASNM_NPBO_TEST/Dataset_official_v1/x.csv'
            path.parent.mkdir(parents=True)
            pd.DataFrame({'label':[1,2,3], 'label_2':[True,True,False], 'F':[1,2,3]}).to_csv(path,index=False)
            output=root/'AMCAL/external_datasets_audit.json'
            preserved={'name':'DReLAB_TEST','files':[{'marker':'retained'}],'errors':[]}
            output.write_text(json.dumps({'datasets':[preserved]}))
            with patch.object(sys,'argv',['audit','--repo-root',str(root),'--datasets','ASNM_NPBO_TEST']):
                with contextlib.redirect_stdout(io.StringIO()): main()
            result=json.loads(output.read_text())
            self.assertEqual(result['datasets'][0],preserved)
            self.assertEqual(result['datasets'][1]['role_rows'],{'direct':1,'adversarial':1,'benign':1})

    def test_missing_files_report_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()): result=audit(Path(tmp))
            self.assertTrue(all(x['errors'] for x in result['datasets']))


if __name__ == '__main__': unittest.main()
