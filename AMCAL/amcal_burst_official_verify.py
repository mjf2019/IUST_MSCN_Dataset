"""Compare existing Burst CSVs with the original authors' data and level-3 UAP.
Downloads public source files into a cache; never modifies input CSVs.
Uses restricted pickle class resolution and selected reviewed preprocessing helpers.
"""
import argparse
import ast
import hashlib
import html
import json
import pickle
import re
from pathlib import Path
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

class DATASET:
    pass

class Restricted(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='__main__' and name=='DATASET':return DATASET
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            from numpy.core import multiarray
            return getattr(multiarray,name)
        if module=='numpy' and name in ('ndarray','dtype'):return getattr(np,name)
        raise pickle.UnpicklingError(f'Blocked pickle global: {module}.{name}')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def download(url,path,drive=False):
    if path.exists():return
    request=urllib.request.urlopen(url,timeout=60)
    if drive and 'text/html' in request.headers.get('Content-Type',''):
        body=request.read().decode();request.close()
        action=re.search(r'<form[^>]*action="([^"]+)"',body)
        if not action:raise RuntimeError('Drive download unavailable; provide the source file in --cache.')
        endpoint=html.unescape(action.group(1))
        if urllib.parse.urlparse(endpoint).hostname!='drive.usercontent.google.com':
            raise RuntimeError('Unexpected Drive confirmation endpoint.')
        params=dict(re.findall(r'type="hidden" name="([^"]+)" value="([^"]+)"',body))
        request=urllib.request.urlopen(endpoint+'?'+urllib.parse.urlencode(params),timeout=60)
    if 'text/html' in request.headers.get('Content-Type',''):
        raise RuntimeError('Received HTML instead of source data.')
    temporary=path.with_suffix(path.suffix+'.part')
    with request,open(temporary,'wb') as f:
        while True:
            b=request.read(1024*1024)
            if not b:break
            f.write(b)
    temporary.rename(path)
    print('Downloaded:',path.name,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    if args.output.exists():p.error('Report already exists; choose a new output filename.')
    paths={
        'dataset':args.cache/'FTSC_Dataset.pickle',
        'uap':args.cache/'AdvBurstTrainedBurst_size_3_Step_2000.npy',
        'notebook':args.cache/'FTSC.ipynb'}
    download('https://drive.google.com/uc?export=download&id=1CVB0w-WvY6MIyG7TTNa4IL9cW90FdJyE',paths['dataset'],True)
    download('https://drive.google.com/uc?export=download&id=1Ens8YjAdJfwJYx_IeEPRYNcJV9cE3qNI',paths['uap'],True)
    download('https://raw.githubusercontent.com/amsadeghzadeh/AdversarialNetworkTraffic/master/FTSC.ipynb',paths['notebook'])
    with open(paths['dataset'],'rb') as f:data=Restricted(f).load()
    notebook=json.loads(paths['notebook'].read_text(encoding='utf-8'))
    source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    source='\n'.join(l for l in source.splitlines() if not l.lstrip().startswith(('%','!')))
    # NumPy 1.18 allowed implicit ragged object-array arithmetic. NumPy 2 rejects
    # it; explicit per-flow arithmetic retains the official numerical operations.
    for split in ['train','validation','test']:
        source=source.replace(f'normalized_{split}_data = {split}_data / train_max',f'normalized_{split}_data = [np.asarray(v) / train_max for v in {split}_data]')
        source=source.replace(f'normalized_{split}_data = {split}_data + ({split}_min < 0) * np.abs({split}_min)',f'normalized_{split}_data = [np.asarray(v) + ({split}_min < 0) * np.abs({split}_min) for v in {split}_data]')
    wanted={'Normalization','flow_to_time_serie','prepare_data','prepare_attack_data_InterPktTime'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:raise RuntimeError('Official notebook helper definitions changed.')
    env={'np':np,'FLOW_TIME_SERIES_LEN':1000,'MAX_NUM_PKT_IN_FLOW_TIME_SERIES':100,'OVER_SAMPLING':0,'INPUT_ITEMS':['Dir+IPT']}
    # Only these reviewed helpers run: no notebook shell/install/training cells.
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_ftsc_helpers','exec'),env)
    prepared=env['prepare_data'](data,[len(data.name[data.traget_classification_index])],3,100)
    train_x,train_y,train_inf,val_x,val_y,val_inf,test_x,test_y,test_inf,norm=prepared
    clean_path=args.data_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv'
    adv_path=args.data_dir/'AdvBurstTrainedBurst_size_3_Step_2000.csv'
    clean=pd.read_csv(clean_path);adv=pd.read_csv(adv_path)
    if len(clean)!=len(test_x) or list(clean.columns)!=list(adv.columns) or len(adv)!=len(clean):
        raise ValueError('CSV schema/row count differs from the official prepared test set.')
    labels=np.argmax(test_y,axis=1)
    clean_error=np.max(np.abs(test_x[:,:100]-clean.iloc[:,:-1].to_numpy()),axis=1)
    attack_x,mask=env['prepare_attack_data_InterPktTime'](test_x,test_inf,{'AdvBurstOverHead':3},norm,set_name='test')
    uap=np.load(paths['uap'],allow_pickle=False)
    rebuilt=attack_x.copy()
    for i,cls in enumerate(labels):
        indices=np.flatnonzero(mask[i]==0)
        rebuilt[i,indices]=uap[cls,:len(indices),0]
    attack_error=np.max(np.abs(rebuilt[:,:100]-adv.iloc[:,:-1].to_numpy()),axis=1)
    report=dict(source_repository='https://github.com/amsadeghzadeh/AdversarialNetworkTraffic',
        source_sha256={k:sha(v) for k,v in paths.items()},csv_sha256={'clean':sha(clean_path),'level3':sha(adv_path)},
        raw_split_sizes={s:len(getattr(data,s+'_data')) for s in ['train','validation','test']},
        prepared_train_rows_before_oversampling=len(train_x),prepared_validation_rows=len(val_x),prepared_test_rows=len(test_x),
        clean_labels_match=bool(np.array_equal(labels,clean.iloc[:,-1])),level3_labels_match=bool(np.array_equal(labels,adv.iloc[:,-1])),
        clean_max_abs_error=float(clean_error.max()),clean_matching_rows_atol_1e_6=int((clean_error<=1e-6).sum()),
        level3_matching_rows_atol_1e_6=int((attack_error<=1e-6).sum()),level3_max_abs_error=float(attack_error.max()),
        discrepant_source_rows=np.flatnonzero((clean_error>1e-6)|(attack_error>1e-6)).tolist(),
        official_normalization={k:float(v) for k,v in vars(norm).items()},
        no_eligible_burst_rows=int((np.sum(mask==0,axis=1)==0).sum()),
        scope='Reconstruction with authors saved UAP, not re-optimization or packet replay.',
        compatibility='Explicit per-flow arithmetic for modern NumPy; original nan_to_num/clipping retained; no oversampling needed for test reconstruction.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
