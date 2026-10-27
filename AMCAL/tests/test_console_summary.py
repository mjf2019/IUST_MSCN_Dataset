import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.console_summary import print_level_result, print_saved_summaries


class ConsoleSummaryTests(unittest.TestCase):
    def test_existing_bom_csv_displays_mean_without_rescaling_percentages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'UniformResults';path.mkdir()
            rows=[dict(method='Uniform',budget_fraction=.05,AC=ac,PR=60,RC=55,F1=57,
                target_final_AC=70,source_final_AC=68,forgetting_AC=-2,labels_percent=5)
                for ac in (60,64)]
            with (path/'summary.csv').open('w',encoding='utf-8-sig',newline='') as output:
                writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture):print_saved_summaries(tmp)
            text=capture.getvalue()
            self.assertIn('Uniform',text);self.assertIn('62.00',text)
            self.assertIn('-2.00',text);self.assertNotIn('6200.00',text)

    def test_level_prints_online_feedback_and_final_endpoints(self):
        result=dict(level=5,online=dict(AC=60,PR=58,RC=56,F1=57),labels=15,cap=15,
            labels_percent=5,original_domain_forgetting=dict(AC=-2),target_after=dict(AC=70),
            source_before=dict(AC=66),source_after=dict(AC=68))
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):print_level_result('Uniform',result)
        self.assertIn('labels 15/15 (5.00%)',capture.getvalue())
        self.assertIn('Final target AC 70.00%',capture.getvalue())
        self.assertIn('66.00% -> 68.00%',capture.getvalue())


if __name__=='__main__':unittest.main()
