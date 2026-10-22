"""PS pipeline: data provenance, complete real training chain and both protocols."""
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,LabelEncoder
import amcal_final_enhancement as runner
import amcal_burst_structured_bank_protocol as legacy
import amcal_paironly_burst_ps as pipeline


class PairOnlyPSTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);legacy.core.device=torch.device('cpu');torch.manual_seed(42)

    def source(self,root):
        source=root/'PS_source';models=source/'Models';models.mkdir(parents=True)
        burst=root/'AdvBurst_FTSC-PS';burst.mkdir()
        raw=np.random.default_rng(42).normal(size=(80,8)).astype(np.float32)
        labels=np.arange(80)%2;features=[f'f{i}' for i in range(8)]
        for level in pipeline.TRAIN_LEVELS+pipeline.TEST_LEVELS:
            path=burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
            frame=pd.DataFrame(raw+level*.02,columns=features);frame['label']=labels;frame.to_csv(path,index=False)
            pd.DataFrame({'source_row':np.arange(80)}).to_csv(path.with_name(path.stem+'_rows.csv'),index=False)
        artifact=dict(feature_names=features,label_name='label',source_rows=80,source_labels=labels,
            train_ids=np.arange(40),val_ids=np.arange(40,60),test_ids=np.arange(60,80),
            scaler=StandardScaler().fit(raw[:40]),label_encoder=LabelEncoder().fit(labels),
            source_sha256=pipeline.digest(burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv'))
        joblib.dump(artifact,models/'protocol_preprocessing.joblib')
        torch.save({'model_state_dict':legacy.core.CNN1D(8,2).state_dict()},models/'initial_cnn_model.pth')
        torch.save({'context_model_state_dict':legacy.core.ContextAwareNetwork(8).state_dict()},models/'adversarial_best_context.pth')
        selector=legacy.StructuredQ(150)
        torch.save(dict(version=1,route_width=150,policy_state_dict=selector.state_dict(),
            model_hashes={n:pipeline.digest(models/n) for n in runner.SOURCE_FILES}),models/'structured_bank_ddqn.pth')
        return source,burst

    def test_real_ps_recipe_no_iat_weights_no_routing_protection_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,burst=self.source(root);output=root/'PS_pipeline'
            args=['--source-run',str(source),'--burst-dir',str(burst),'--output-root',str(output),
                '--device','cpu','--base-epochs','1','--utility-epochs','1','--pair-epochs','1',
                '--base-warmup-episodes','0','--patience','1','--seed','42']
            with redirect_stdout(io.StringIO()):pipeline.main(args)
            plan=json.loads((output/'pipeline_plan.json').read_text())
            self.assertEqual(plan['train_levels'],[0,1,3]);self.assertEqual(plan['test_levels'],pipeline.TEST_LEVELS)
            self.assertEqual(plan['representation'],'Burst_PS')
            for stage in plan['stages']:
                self.assertNotIn('IAT',' '.join(stage['args']))
                self.assertNotIn('--router-protected-replay',stage['args'])
                self.assertTrue((Path(stage['output'])/'pipeline_stage_complete.json').is_file())
            checkpoint=output/'paironly/Models/final_enhancement.pth'
            digest=pipeline.digest(checkpoint)
            saved=torch.load(checkpoint,weights_only=False)
            self.assertEqual(saved['training_context_count'],1)
            protocol=saved['training_protocol']['selector_specialization']
            self.assertEqual(protocol['trained_parameters'],'update_value only')
            self.assertIn('paired objective only',protocol['optimizer'])
            self.assertEqual(saved['parent_checkpoint_sha256'],pipeline.digest(output/'counterfactual/Models/final_enhancement.pth'))
            results=list((output/'paironly/Results').glob('evaluation_*'));self.assertEqual(len(results),2)
            kinds=set()
            for result in results:
                ex=json.loads((result/'execution.json').read_text());kinds.add(ex['continual'])
                self.assertEqual(ex['context_count'],1);self.assertEqual(ex['modes'],['learned'])
                self.assertNotIn('router_protected_replay',ex)
                self.assertNotIn('router_level_balanced_history',ex)
                self.assertEqual(ex['checkpoint_sha256'],digest)
                self.assertTrue((result/'routing_oracle_summary.csv').is_file())
                for level in pipeline.TEST_LEVELS:
                    metric=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertLessEqual(metric['labels'],metric['label_cap'])
                    self.assertTrue(metric['classifier_unchanged'])
                    self.assertTrue(set(metric['source_ids']).issubset(set(range(60,80))))
            self.assertEqual(kinds,{False,True})
            with redirect_stdout(io.StringIO()),patch.object(runner,'train',side_effect=AssertionError('No retraining')), \
                    patch.object(runner,'episode',side_effect=AssertionError('No repeated evaluation')):
                pipeline.main(args)
            self.assertEqual(pipeline.digest(checkpoint),digest)
            with self.assertRaisesRegex(ValueError,'recipe/provenance differs'),redirect_stdout(io.StringIO()):
                pipeline.main([*args,'--pair-epochs','2'])

    def test_ps_source_mismatch_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,burst=self.source(root);output=root/'unused'
            path=burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv'
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Source data differ'):
                pipeline.main(['--source-run',str(source),'--burst-dir',str(burst),'--output-root',str(output),'--device','cpu'])
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
