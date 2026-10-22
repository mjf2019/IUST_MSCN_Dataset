from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import amcal_final_enhancement as runner
import amcal_batch5_pipeline as pipeline
from amcal_light_cnn import LightCNN1D, classifier_for_checkpoint, ARCHITECTURE
from test_paironly_burst_ps import PairOnlyPSTests


class Batch5PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.nnpack.set_flags(False)
        runner.legacy.core.device=torch.device('cpu')
        torch.manual_seed(42)

    def common(self,root,source,burst,dataset):
        return ['--dataset',dataset,'--source-run',str(source),'--burst-dir',str(burst),
            '--output-root',str(root),'--device','cpu','--epochs','1','--patience','1',
            '--utility-epochs','1','--cnn-epochs','1','--context-epochs','1']

    def check_final(self,root):
        models=root/'paironly_batch5/Models';checkpoint=models/'final_enhancement.pth'
        saved=torch.load(checkpoint,weights_only=False)
        self.assertEqual(saved['training_context_count'],1)
        self.assertEqual(saved['training_protocol']['context_update']['mode'],'batch5')
        self.assertEqual(saved['training_protocol']['selector_specialization']['trained_parameters'],'update_value only')
        self.assertIn('paired objective only',saved['training_protocol']['selector_specialization']['optimizer'])
        results=list((root/'paironly_batch5/Results').glob('evaluation_*'))
        self.assertEqual(len(results),2)
        self.assertEqual({json.loads((r/'execution.json').read_text())['continual'] for r in results},{False,True})
        for result in results:
            ex=json.loads((result/'execution.json').read_text())
            self.assertEqual(ex['paid_replay']['mode'],'batch5')
            self.assertEqual(ex['paid_replay']['count'],4)
            for level in pipeline.TEST:
                m=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                self.assertTrue(m['classifier_unchanged']);self.assertLessEqual(m['labels'],m['label_cap'])
                self.assertTrue(set(m['source_ids'])<=set(range(60,80)))
        return pipeline.digest(checkpoint)

    def test_real_light_ps_train_and_both_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source,burst=PairOnlyPSTests().source(base);out=base/'light'
            args=self.common(out,source,burst,'PS')
            original_hash=pipeline.digest(source/'Models/initial_cnn_model.pth')
            read_domain=runner.legacy.read_domain
            def training_domains(root,artifact,level,split,*rest,**kw):
                self.assertNotEqual(split,'test_ids','No TEST access during TRAIN')
                return read_domain(root,artifact,level,split,*rest,**kw)
            with redirect_stdout(io.StringIO()),patch.object(runner.legacy,'read_domain',side_effect=training_domains):
                pipeline.main([*args,'--action','Train'])
            self.assertFalse((out/'paironly_batch5/Results').exists())
            with redirect_stdout(io.StringIO()):
                pipeline.main([*args,'--action','Evaluate']);pipeline.main([*args,'--action','Audit'])
            digest=self.check_final(out)
            src=torch.load(out/'source/Models/initial_cnn_model.pth',weights_only=False)
            self.assertEqual(src['architecture'],ARCHITECTURE)
            log=json.loads((out/'source/source_training.json').read_text())
            self.assertLess(log['light_parameters'],log['original_parameters'])
            self.assertTrue((out/'Results/frozen_generalization.csv').exists())
            self.assertEqual(original_hash,pipeline.digest(source/'Models/initial_cnn_model.pth'))
            with redirect_stdout(io.StringIO()),patch.object(runner,'train',side_effect=AssertionError('No retraining')):
                pipeline.main(args)
            self.assertEqual(digest,pipeline.digest(out/'paironly_batch5/Models/final_enhancement.pth'))
            with self.assertRaisesRegex(ValueError,'recipe changed'),redirect_stdout(io.StringIO()):
                pipeline.main([*args,'--epochs','2'])

    def test_iat_specialization_reuses_only_matching_source(self):
        import amcal_paironly_burst_ps as old
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source,burst=PairOnlyPSTests().source(base)
            prior=base/'prior'
            with redirect_stdout(io.StringIO()):
                old.main(['--source-run',str(source),'--burst-dir',str(burst),'--output-root',str(prior),
                    '--action','Train','--device','cpu','--base-epochs','1','--utility-epochs','1',
                    '--pair-epochs','1','--base-warmup-episodes','0','--patience','1'])
            out=base/'iat';args=self.common(out,prior/'counterfactual',burst,'IAT')
            with redirect_stdout(io.StringIO()):pipeline.main(args)
            self.check_final(out)
            # A trained batch5 checkpoint inherits its update rule when no override is supplied.
            a=pipeline.argparse.Namespace(dataset='IAT',output_root=out,source_run=prior/'counterfactual',
                burst_dir=burst,seed=42,device='cpu',epochs=1,patience=1,utility_epochs=1,evaluation_tag='')
            _,evaluation=pipeline.make_plan(a)
            index=evaluation.index('--paid-replay-mode');del evaluation[index:index+2]
            with redirect_stdout(io.StringIO()):runner.main(evaluation)
            with self.assertRaisesRegex(ValueError,'differs'):
                runner.main([*evaluation,'--paid-replay-mode','batch8'])
            self.assertFalse((out/'source').exists())
            saved=torch.load(out/'paironly_batch5/Models/final_enhancement.pth',weights_only=False)
            self.assertEqual(saved['parent_checkpoint_sha256'],pipeline.digest(prior/'counterfactual/Models/final_enhancement.pth'))

    def test_factory_shapes_embedding_and_unknown_architecture(self):
        model=LightCNN1D(100,5).eval();x=torch.randn(3,1,100)
        logits,embedding=runner.legacy.cnn_with_embedding(model,x)
        self.assertEqual(tuple(logits.shape),(3,5));self.assertEqual(tuple(embedding.shape),(3,128))
        torch.testing.assert_close(logits,model(x))
        with self.assertRaisesRegex(ValueError,'Unknown'):classifier_for_checkpoint({'architecture':'bad'},100,5,LightCNN1D)


if __name__=='__main__':unittest.main()
