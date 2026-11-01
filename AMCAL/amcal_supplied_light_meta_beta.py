"""Final architecture control: clean-only supplied attacks with Light CNN.

Reuse the original experiment's partitions, preprocessing, DDQN/beta algorithm
and evaluation. Train separate Light CNN/Context weights on clean TRAIN only.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
import torch

import amcal_clean_meta_beta as shared
import amcal_batch5_pipeline as source_fit
from amcal_light_cnn import LightCNN1D, ARCHITECTURE
from enhancements.config import EnhancementConfig

api = shared.api
base = shared.base
ROOT = Path(__file__).resolve().parent
MODEL_FILES = ('protocol_preprocessing.joblib','initial_cnn_model.pth','adversarial_best_context.pth')


def verify_input(a):
    manifest = shared.read(a.source_root/'clean_supplied_protocol.json')
    recipe = manifest['recipe']
    shared.validate_recipe(recipe, a.dataset, a.seed)
    marker = shared.read(a.source_root/'source'/'clean_source_complete.json')
    if marker['recipe'] != recipe:
        raise ValueError('Original clean source recipe mismatch.')
    for name, digest in marker['outputs'].items():
        if api.digest(a.source_root/'source'/name) != digest:
            raise ValueError('Original source changed: '+name)
    clean = {k:v for k,v in manifest['prepared_hashes'].items() if k.startswith('TrainingDomains/')}
    if not clean:
        raise ValueError('Missing prepared clean TRAIN hashes.')
    for name, digest in clean.items():
        if api.digest(a.source_root/name) != digest:
            raise ValueError('Prepared clean data changed: '+name)
    artifact = api.base.load_artifact(a.source_root/'source'/'Models', a.burst_dir)
    for key, name in [('train_ids','train'),('val_ids','validation'),('test_ids','test')]:
        if not np.array_equal(artifact[key], recipe['partitions'][name]):
            raise ValueError('Original partition mismatch: '+key)
    if any(shared.excluded(name) for name in artifact['feature_names']):
        raise ValueError('Excluded identifiers found in source features.')
    return manifest, artifact, marker


def source_stage(a, manifest, artifact):
    source = a.output/'source'; models = source/'Models'
    preprocessor = a.source_root/'source'/'Models'/'protocol_preprocessing.joblib'
    recipe = dict(version=1, dataset=a.dataset, seed=a.seed, architecture=ARCHITECTURE,
        training_levels=[0], validation_levels=[0],
        original_source_recipe=manifest['recipe'],
        preprocessing_sha256=api.digest(preprocessor),
        cnn_epochs=a.cnn_epochs, context_epochs=a.context_epochs, patience=a.patience,
        source_fit='same amcal_batch5_pipeline.fit as original source; CNN AdamW .001; Context AdamW .0003',
        initialization='fresh Light CNN and Context; no original/other-dataset model weights imported',
        implementation_sha256={name:api.digest(ROOT/name) for name in
            ('amcal_supplied_light_meta_beta.py','amcal_light_cnn.py','amcal_batch5_pipeline.py')})
    pf = source/'recipe.json'; done = source/'training_complete.json'
    if pf.exists():
        if shared.read(pf) != recipe:
            raise ValueError('Light source recipe changed; use a fresh --output.')
    elif source.exists() and any(source.iterdir()):
        raise ValueError('Unmatched source preserved; use a fresh --output.')
    elif a.action == 'Evaluate':
        raise ValueError('Train the separate Light source first.')
    else:
        shared.write(pf, recipe)
    if done.exists():
        record = shared.read(done)
        if record['recipe'] != recipe or any(api.digest(models/name) != sha
                for name,sha in record['sha256'].items()):
            raise ValueError('Completed Light source changed.')
        print('Skipping verified Light CNN/Context source.', flush=True)
        return recipe
    if a.action == 'Evaluate':
        raise ValueError('Light source stage incomplete.')
    models.mkdir(parents=True, exist_ok=True)
    saved_preprocessor = models/preprocessor.name
    if saved_preprocessor.exists():
        if api.digest(saved_preprocessor) != api.digest(preprocessor):
            raise ValueError('Existing Light preprocessing differs from original.')
    else:
        shutil.copy2(preprocessor, saved_preprocessor)
    a.classes = list(artifact['label_encoder'].classes_)
    domains = [api.base.legacy.read_domain(a.burst_dir, artifact, 0, split)
               for split in ('train_ids','val_ids')]
    for d in domains:
        shared.check_classes(d, len(a.classes), 'Light clean source')
    tensors = [(d[0],torch.as_tensor(d[1],dtype=torch.long,device=d[0].device)) for d in domains]
    cnn_file = models/'initial_cnn_model.pth'; cnn_done = source/'cnn_complete.json'
    if cnn_done.exists():
        record = shared.read(cnn_done)
        if record['recipe'] != recipe or record['sha256'] != api.digest(cnn_file):
            raise ValueError('Completed Light CNN changed.')
        cnn = api.base.load_cnn(models, artifact)
        cnn_logs = record['logs']
    else:
        api.base.legacy.core.set_seed(a.seed)
        cnn = LightCNN1D(len(artifact['feature_names']),len(a.classes)).to(api.base.legacy.core.device)
        print(f'LIGHT SOURCE CNN DATASET={a.dataset} TRAIN=[0] VAL=[0] MAX_EPOCHS={a.cnn_epochs}',flush=True)
        cnn_logs = source_fit.fit(a, cnn, None, *tensors)
        shared.save(cnn_file,dict(architecture=ARCHITECTURE, model_state_dict=cnn.state_dict(),
            preprocessing_sha256=api.digest(saved_preprocessor),
            input_length=len(artifact['feature_names']),num_classes=len(a.classes),training_levels=[0]))
        # Restore this RNG before Context initialization on an interrupted rerun.
        shared.save(source/'source_rng.pth',dict(cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        shared.write(cnn_done,dict(recipe=recipe,sha256=api.digest(cnn_file),logs=cnn_logs,
            rng_sha256=api.digest(source/'source_rng.pth')))
    cnn.eval().requires_grad_(False)
    record = shared.read(cnn_done)
    if api.digest(source/'source_rng.pth') != record['rng_sha256']:
        raise ValueError('CNN-to-Context RNG checkpoint changed.')
    rng = torch.load(source/'source_rng.pth',map_location='cpu',weights_only=False)
    torch.set_rng_state(rng['cpu'])
    if rng['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    context = api.base.legacy.core.ContextAwareNetwork(len(artifact['feature_names'])).to(api.base.legacy.core.device)
    print(f'LIGHT SOURCE CONTEXT DATASET={a.dataset} TRAIN=[0] VAL=[0] MAX_EPOCHS={a.context_epochs}',flush=True)
    context_logs = source_fit.fit(a, context, cnn, *tensors, context=True)
    shared.save(models/'adversarial_best_context.pth',dict(context_model_state_dict=context.state_dict(),
        preprocessing_sha256=api.digest(saved_preprocessor),training_levels=[0]))
    shared.write(source/'source_training.json',dict(recipe=recipe,architecture=ARCHITECTURE,
        training_levels=[0],cnn=cnn_logs,context=context_logs,selection='clean VAL loss only'))
    shared.write(done,dict(recipe=recipe,sha256={name:api.digest(models/name) for name in MODEL_FILES}))
    return recipe


class LightExperiment(shared.CleanExperiment):
    def __init__(self,a,manifest,recipe):
        self.a = a; self.source_root = a.source_root; self.training_dir = a.burst_dir
        self.models = a.output/'source'/'Models'
        self.manifest = deepcopy(manifest)
        self.manifest['recipe'] = dict(manifest['recipe'],architecture=ARCHITECTURE,light_source_recipe=recipe)
        self.artifact = api.base.load_artifact(self.models,a.burst_dir)
        a.classes = list(self.artifact['label_encoder'].classes_)
        a.config = EnhancementConfig.preset('accuracy').validated()
        self.cnn = api.base.load_cnn(self.models,self.artifact)
        saved = torch.load(self.models/'adversarial_best_context.pth',map_location='cpu',weights_only=False)
        self.template = api.base.build_bank(self.models,self.artifact,a.config,
            saved['context_model_state_dict'],slots=1)[0].eval()
        self.original_weights = (base.weight_hash(self.cnn.state_dict()),base.weight_hash(self.template.state_dict()))
        self.model_files = {name:api.digest(self.models/name) for name in MODEL_FILES}
        self.full_train = self.domain(a.burst_dir,0,'train_ids')
        self.validation = self.domain(a.burst_dir,0,'val_ids')
        self.parts = shared.reserve_clean(self.full_train,a.seed,len(a.classes),a.meta_rows)
        self.train = self.parts['inner']; self.reference = self.parts['reference']
        self.anchors = self.full_train; self.train_anchors = self.train
        self.reward_probe_keys = set().union(*(set(api.feature_keys(self.parts[name][0]))
            for name in ('gain','meta_gain')))
        self.gain_by_domain = {}
        self.val_stream,self.val_reference,_ = api.reference_partition(self.validation,a.seed,.2)
        self.val_stream = shared.without(self.val_stream,[self.val_reference])
        for d in (self.val_stream,self.val_reference):
            shared.check_classes(d,len(a.classes),'Light clean VAL')

    def assert_source_unchanged(self):
        if self.original_weights != (base.weight_hash(self.cnn.state_dict()),base.weight_hash(self.template.state_dict())):
            raise RuntimeError('Light CNN or initial Context changed.')
        if any(api.digest(self.models/name) != sha for name,sha in self.model_files.items()):
            raise RuntimeError('Light source file changed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=list(shared.DATASETS),required=True)
    p.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    p.add_argument('--protocol',choices=['Both','Online','Independent'],default='Both')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--source-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--warmup-epochs',type=int,default=3)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--meta-rows',type=int,default=300)
    a = p.parse_args()
    if min(a.warmup_epochs,a.epochs) < 1 or int(.2*a.meta_rows) < 32:
        p.error('Positive epochs and at least 32 paid meta observations required.')
    runs = ROOT/shared.DATASETS[a.dataset][0]/'protocol_runs'
    a.source_root = (a.source_root or runs/f'clean0_AMCAL_attacks_seed{a.seed}').resolve()
    a.output = (a.output or runs/f'clean0_light_meta_beta_ddqn_seed{a.seed}_v1').resolve()
    if a.output == a.source_root or a.output in a.source_root.parents or a.source_root in a.output.parents:
        p.error('Light output must be separate from the original source.')
    a.burst_dir = a.source_root/'TrainingDomains'
    a.budget_fraction = .2; a.context_lr = .00025; a.replay_weight = .1; a.beta = 0.
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; choose --device cpu.')
    api.base.legacy.core.device = torch.device(device); api.base.legacy.core.set_seed(a.seed)
    manifest,artifact,marker = verify_input(a)
    settings = manifest['recipe']['training_settings']
    a.cnn_epochs = settings['cnn_epochs']; a.context_epochs = settings['context_epochs']; a.patience = settings['patience']
    pf = a.output/'protocol.json'
    if pf.exists():
        existing = shared.read(pf)
        if (existing.get('cnn_architecture') != ARCHITECTURE
                or existing.get('dataset') != a.dataset or existing.get('seed') != a.seed
                or existing.get('implementation_extra_sha256') != api.digest(Path(__file__))):
            p.error('Existing output belongs to another experiment; use a fresh --output.')
    if a.output.exists() and not pf.exists() and any(x.name != 'source' for x in a.output.iterdir()):
        p.error('Unmatched Light output preserved; use a fresh --output.')
    recipe = source_stage(a,manifest,artifact)
    exp = LightExperiment(a,manifest,recipe)
    protocol = shared.protocol_for(exp)
    protocol.update(initialization='fresh Light source on same clean TRAIN/VAL; unchanged DDQN/beta pipeline',
        cnn_architecture=ARCHITECTURE,implementation_extra_sha256=api.digest(Path(__file__)))
    if pf.exists():
        if shared.read(pf) != protocol:
            p.error('Light training recipe changed; use a fresh --output.')
    elif a.action == 'Evaluate':
        p.error('Train the Light DDQN/beta first.')
    else:
        shared.write(pf,protocol)
    print(f'FINAL LIGHT CONTROL DATASET={a.dataset} TRAIN=[0] VAL=[0] '
        f'FULL_TRAIN={len(exp.full_train[1])} FULL_TEST_PER_INTENSITY={len(exp.artifact["test_ids"])} '
        'BUDGET=20% | unchanged reward/DDQN/beta; no TEST cap',flush=True)
    if a.action in ('All','Train'):
        shared.train(exp,protocol)
    if a.action in ('All','Evaluate'):
        shared.evaluate(exp,protocol)
    exp.assert_source_unchanged()
    if any(api.digest(a.source_root/'source'/name) != sha for name,sha in marker['outputs'].items()):
        raise RuntimeError('An original source file changed.')
    print('OUTPUT: '+str(a.output),flush=True)


if __name__ == '__main__':
    main()
