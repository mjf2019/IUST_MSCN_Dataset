"""Export official FTSC-PS clean splits and all saved AdvBurst intensities.
Requires numpy/pandas and amcal_burst_official_verify.py in the same directory.
No optimization or GPU required; original authors' saved per-class UAPs are used.
"""
import argparse, ast, json
from pathlib import Path
import numpy as np
import pandas as pd
from amcal_burst_official_verify import Restricted, download, sha
UAPS = {'AdvBurstTrainedBurst_size_1_Step_2000.npy': '1Dmc2A99t8IKtD-5fG1yAdpVW4tpvzMRa', 'AdvBurstTrainedBurst_size_3_Step_2000.npy': '1Djp2DPGQrcdJ9_YfbgVYNZV6aCrUhhSe', 'AdvBurstTrainedBurst_size_5_Step_2000.npy': '1DXWSgKlazRHBZmqwi38CiAX3QCxQyVGZ', 'AdvBurstTrainedBurst_size_7_Step_2000.npy': '1Docs5gQuFx-U4BVU90wT7RhIfbxbxaBL', 'AdvBurstTrainedBurst_size_10_Step_2000.npy': '1DPY1n-LnIFm2uIwlv7mtOjrV7ZWCayCF', 'AdvBurstTrainedBurst_size_12_Step_2000.npy': '1DUGlNkjuMGc5mHTGv0wXEgdxo4BhFKpR', 'AdvBurstTrainedBurst_size_15_Step_2000.npy': '1D7AuTtn2wYwR3YE-HFBcCYLuW5h0Qfpx', 'AdvBurstTrainedBurst_size_17_Step_2000.npy': '1DZCAMmJe2QvTXi-1iQH30Zb_ZH5I6pu8', 'AdvBurstTrainedBurst_size_20_Step_2000.npy': '1CualUgnEa0rdKPP2T-ed_8VXMivsoi8f'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory must be empty to avoid replacing data.')
    a.cache.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    dataset=a.cache/'FTSC_Dataset.pickle';nb=a.cache/'FTSC.ipynb'
    download('https://drive.google.com/uc?export=download&id=1CVB0w-WvY6MIyG7TTNa4IL9cW90FdJyE',dataset,True)
    download('https://raw.githubusercontent.com/amsadeghzadeh/AdversarialNetworkTraffic/master/FTSC.ipynb',nb)
    with dataset.open('rb') as f:data=Restricted(f).load()
    source='\n'.join(''.join(c['source']) for c in json.loads(nb.read_text())['cells'] if c['cell_type']=='code')
    source='\n'.join(l for l in source.splitlines() if not l.lstrip().startswith(('%','!')))
    for split in ('train','validation','test'):
        replacements={
          f'({split}_data - mean) / std':f'[ (np.asarray(v) - mean) / std for v in {split}_data ]',
          f'{split}_data / train_max':f'[np.asarray(v) / train_max for v in {split}_data]',
          f'{split}_data + ({split}_min < 0) * np.abs({split}_min)':f'[np.asarray(v) + ({split}_min < 0) * np.abs({split}_min) for v in {split}_data]'}
        for old,new in replacements.items():source=source.replace(old,new)
    wanted={'Normalization','flow_to_time_serie','prepare_data','prepare_attack_data_PktSize'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:raise RuntimeError('Official helper definitions changed')
    env=dict(np=np,FLOW_TIME_SERIES_LEN=1000,MAX_NUM_PKT_IN_FLOW_TIME_SERIES=100,OVER_SAMPLING=0,INPUT_ITEMS=['Dir+PktSize'])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_ps_helpers','exec'),env)
    tx,ty,ti,vx,vy,vi,x,y,info,norm=env['prepare_data'](data,[len(data.name[data.traget_classification_index])],3,100)
    labels=np.argmax(y,axis=1);files={};levels={}
    header=norm.direct_PktSizeNormalization([np.array([54])],set_name='validation')[0][0]
    domain=(0.,float(np.max(tx)-header),float(np.min(tx)),float(np.max(tx)))
    def save(path,x,y):
        frame=pd.DataFrame(x[:,:100],columns=[str(i) for i in range(1,101)])
        frame['label']=np.argmax(y,axis=1);frame.to_csv(path,index=False)
        files[str(path.relative_to(a.output))]=dict(rows=len(frame),sha256=sha(path))
    for split,sx,sy in [('train',tx,ty),('validation',vx,vy),('test',x,y)]:
        save(a.output/f'{split}_dataset.csv',sx,sy)
    target=a.output/'AdvBurst_FTSC-PS';target.mkdir(exist_ok=True)
    save(target/'AdvBurstTrainedBurst_size_0_Step_2000.csv',x,y)
    for level in (1,3,5,7,10,12,15,17,20):
        name=f'AdvBurstTrainedBurst_size_{level}_Step_2000.npy';path=a.cache/('PS_'+name)
        download('https://drive.google.com/uc?export=download&id='+UAPS[name],path,True)
        uap=np.load(path,allow_pickle=False)
        if uap.shape!=(len(data.name[data.traget_classification_index]),1000,1):raise ValueError('Unexpected UAP shape')
        attacked,mask=env['prepare_attack_data_PktSize'](x,info,{'AdvBurstOverHead':level},norm,set_name='test')
        for i,cls in enumerate(labels):
            idx=np.flatnonzero(mask[i]==0);values=np.clip(uap[cls,:len(idx),0],min(domain[:2]),max(domain[:2]))
            # Official add_AdvBurst_to_FTSC_seq: UAP + sign(UAP)*header.
            attacked[i,idx]=values+np.sign(values)*attacked[i,idx]
        attacked=np.clip(attacked,domain[2],domain[3])
        if not np.isfinite(attacked).all():raise ValueError('Non-finite features')
        save(target/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv',attacked,y)
        levels[str(level)]={'uap_sha256':sha(path),'no_eligible_burst_rows':int((np.sum(mask==0,axis=1)==0).sum())}
        print('Exported PS level',level,flush=True)
    report=dict(representation='FTSC-PS',source='https://github.com/amsadeghzadeh/AdversarialNetworkTraffic',dataset_sha256=sha(dataset),notebook_sha256=sha(nb),classes=list(data.name[data.traget_classification_index]),normalization={k:float(v) for k,v in vars(norm).items()},files=files,levels=levels,scope='Official saved UAP reconstruction; no re-optimization. Original normalization retained, including split-dependent shifts. Train split is not oversampled. PS row counts can differ from IAT after short-flow filtering.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['rows'] for k,v in files.items()},indent=2))
if __name__=='__main__':main()
