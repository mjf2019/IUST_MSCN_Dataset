"""AdvBurst Algorithm 3 reproduction and CSV integrity diagnostics.

Audit proves only insertion/truncation consistency, not physical feasibility or
correct generation. Generate requires explicit signed-log representation and
packet metadata; never guesses packet directions/lengths from scaled features.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def insertion_candidates(clean, attacked, count, atol=1e-6):
    n = len(clean)
    if count < 1 or count > n:
        raise ValueError('Dummy count must be within sequence length.')
    # Append inside the visible window, shift original suffix, truncate to n.
    return [p for p in range(n-count+1)
            if np.allclose(attacked[:p], clean[:p], atol=atol, rtol=0)
            and np.allclose(attacked[p+count:], clean[p:n-count], atol=atol, rtol=0)]


def audit(args):
    root = args.data_dir
    clean_path = root / 'AdvBurstTrainedBurst_size_0_Step_2000.csv'
    clean = pd.read_csv(clean_path)
    x = clean.iloc[:, :-1].to_numpy(dtype=float)
    results = []
    for count in args.levels:
        path = root / f'AdvBurstTrainedBurst_size_{count}_Step_2000.csv'
        frame = pd.read_csv(path)
        row = dict(level=count, sha256=sha(path), rows=len(frame),
                   schema_matches=list(frame.columns)==list(clean.columns),
                   positional_correspondence='hypothesis tested by preserved feature prefix/suffix, not source provenance')
        if not row['schema_matches'] or len(frame) != len(clean):
            row['status']='shape/schema mismatch'; results.append(row); continue
        y = frame.iloc[:, :-1].to_numpy(dtype=float)
        row['labels_match'] = bool(np.array_equal(clean.iloc[:, -1], frame.iloc[:, -1]))
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            row['status']='nonfinite values';results.append(row);continue
        records=[]
        for i,(a,b) in enumerate(zip(x,y)):
            candidates=insertion_candidates(a,b,count,args.atol)
            changed=not np.allclose(a,b,atol=args.atol,rtol=0)
            p=candidates[0] if len(candidates)==1 else None
            records.append(dict(source_row=i,label=str(clean.iloc[i,-1]),changed=changed,
                candidate_count=len(candidates),insertion_index=p,
                inserted_values=None if p is None else b[p:p+count].tolist()))
        changed=[r for r in records if r['changed']]
        unique=[r for r in changed if r['candidate_count']==1]
        row.update(changed_rows=len(changed), unchanged_rows=len(records)-len(changed),
            insertion_consistent_changed=sum(r['candidate_count']>0 for r in changed),
            unique_insertion_changed=len(unique),
            inconsistent_changed=sum(r['candidate_count']==0 for r in changed),
            insertion_index_counts=dict(Counter(str(r['insertion_index']) for r in unique)),
            status='structural audit only; no automatic certification')
        classes={}
        for label in sorted(set(r['label'] for r in unique)):
            rs=[r for r in unique if r['label']==label]
            vectors=np.asarray([r['inserted_values'] for r in rs])
            classes[label]=dict(rows=len(rs),shared_uap_within_tolerance=bool(np.all(np.ptp(vectors,axis=0)<=args.atol)),
                coordinate_ranges=np.ptp(vectors,axis=0).tolist(), example_uap=vectors[0].tolist())
        row['per_class_uap']=classes
        pd.DataFrame(records).to_json(args.output/f'level{count}_rows.json',orient='records',indent=2)
        results.append(row)
        print(f"PER {count}: labels={row['labels_match']}; changed={len(changed)}; insertion consistent={row['insertion_consistent_changed']}; inconsistent={row['inconsistent_changed']}; positions={row['insertion_index_counts']}")
    report=dict(clean_sha256=sha(clean_path),clean_rows=len(clean),features=x.shape[1],levels=results,
        limitations=['Structural insertion equality does not establish capture/session provenance.',
        'Direction, IAT bounds and units need original normalization and packet metadata.',
        'Shared per-class UAP is necessary for this formulation but does not prove gradient optimization.',
        'No dependence on AMCAL accuracy; no existing data changed.'])
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))


def generate(args):
    # Torch imports are lazy: audit requires only numpy/pandas.
    import torch
    import torch.nn.functional as F
    import joblib
    import amcal_three_action_protocol as core
    if args.representation != 'signed-log':
        raise ValueError('Only explicitly confirmed paper signed-log IAT is supported.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifact=joblib.load(args.checkpoint_output/'Models/protocol_preprocessing.joblib')
    fit=pd.read_csv(args.fit_csv); target=pd.read_csv(args.target_csv)
    expected=artifact['feature_names']+[artifact['label_name']]
    if list(fit.columns)!=expected or list(target.columns)!=expected:
        raise ValueError('CSV schema differs from classifier preprocessing.')
    # Enforce the exact representation: directions are never inferred from transformed signs.
    xf=fit.iloc[:,:-1].to_numpy(dtype=np.float32);xt=target.iloc[:,:-1].to_numpy(dtype=np.float32)
    if not np.isfinite(xf).all() or not np.isfinite(xt).all() or max(np.abs(xf).max(),np.abs(xt).max())>2.0001:
        raise ValueError('Values incompatible with declared paper signed-log representation; supply original data/normalization.')
    if set(core.feature_groups(fit)) & set(core.feature_groups(target)):
        raise ValueError('Fit and target contain exact-feature overlap; supply disjoint source rows.')
    def metadata(path,n):
        m=pd.read_csv(path)
        required=['source_row','packet_count','burst_end','direction']
        if any(k not in m for k in required) or len(m)!=n:
            raise ValueError('Metadata needs source_row,packet_count,burst_end,direction for every CSV row.')
        for k in required:
            v=pd.to_numeric(m[k],errors='raise').to_numpy()
            if not np.isfinite(v).all() or not np.equal(v,np.floor(v)).all():
                raise ValueError('Metadata values must be finite integers.')
        if m.source_row.duplicated().any() or not m.direction.isin([-1,1]).all():
            raise ValueError('Invalid source IDs or packet directions.')
        if ((m.packet_count<1)|(m.packet_count>xf.shape[1])|(m.burst_end<1)|(m.burst_end>m.packet_count)).any():
            raise ValueError('burst_end is an exclusive packet index within packet_count.')
        return m
    mf=metadata(args.fit_metadata,len(fit));mt=metadata(args.target_metadata,len(target))
    if set(mf.source_row)&set(mt.source_row):
        raise ValueError('Fit and target metadata must use globally disjoint original flow IDs.')
    labels=artifact['label_encoder']; yf=labels.transform(fit.iloc[:,-1]);yt=labels.transform(target.iloc[:,-1])
    checkpoint=args.checkpoint_output/'Models/initial_cnn_model.pth'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['preprocessing_sha256']!=core.file_sha256(args.checkpoint_output/'Models/protocol_preprocessing.joblib'):
        raise ValueError('Classifier preprocessing hash mismatch.')
    cnn=core.CNN1D(xf.shape[1],len(labels.classes_)).to(device)
    cnn.load_state_dict(ck['model_state_dict']);cnn.eval()
    for param in cnn.parameters():param.requires_grad_(False)
    scale=torch.tensor(artifact['scaler'].scale_,dtype=torch.float32,device=device)
    mean=torch.tensor(artifact['scaler'].mean_,dtype=torch.float32,device=device)
    def logits(x):return cnn(((x-mean)/scale).unsqueeze(1))
    # Eq 21 in microseconds, IATmax=180 s in the paper experiments.
    def normalized(seconds):return 2*np.log(seconds*1e6+1)/np.log(args.iat_max_seconds*1e6)
    low,high=normalized(.001),normalized(.1)
    rng=np.random.default_rng(args.seed);torch.manual_seed(args.seed)
    fit_tensor=torch.tensor(xf,device=device)
    def insert(x,meta,uap):
        out=[];n=x.shape[1]
        for row,(_,m) in zip(x,meta.iterrows()):
            p=int(m.burst_end);length=int(m.packet_count)
            # Preserve observed sequence, append signed dummy IATs, then fixed-window truncate/pad.
            seq=torch.cat((row[:p],uap*int(m.direction),row[p:length]))[:n]
            out.append(F.pad(seq,(0,n-len(seq))))
        return torch.stack(out)
    def score(x):
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(x),128):predictions.extend(logits(x[start:start+128]).argmax(1).cpu().tolist())
        return float(np.mean(np.asarray(predictions)==yt))
    clean_score=score(torch.tensor(xt,device=device));rows=[]
    for count in args.levels:
        out=xt.copy(); random_out=xt.copy();uaps={}
        for cls in np.unique(yt):
            fit_ids=np.flatnonzero(yf==cls);target_ids=np.flatnonzero(yt==cls)
            if not len(fit_ids):raise ValueError('Target class absent from attack fitting partition.')
            u=torch.tensor(rng.uniform(low,high,count),dtype=torch.float32,device=device,requires_grad=True)
            for step in range(args.iterations):
                ids=rng.choice(fit_ids,size=min(args.batch_size,len(fit_ids)),replace=False)
                adv=insert(fit_tensor[ids],mf.iloc[ids],u)
                loss=F.cross_entropy(logits(adv),torch.full((len(ids),),int(cls),device=device,dtype=torch.long))
                grad,=torch.autograd.grad(loss,u)
                # Algorithm 3: raw gradient ascent, not sign-gradient/Adam.
                u=(u+args.step_size*grad).clamp(low,high).detach().requires_grad_(True)
            uaps[str(cls)]=u.detach().cpu().tolist()
            t=torch.tensor(xt[target_ids],device=device)
            with torch.no_grad():
                out[target_ids]=insert(t,mt.iloc[target_ids],u).cpu().numpy()
                ru=torch.tensor(rng.uniform(low,high,count),dtype=torch.float32,device=device)
                random_out[target_ids]=insert(t,mt.iloc[target_ids],ru).cpu().numpy()
        for name,data in [('AdvBurst',out),('RandBurst_single',random_out)]:
            f=target.copy();f.iloc[:,:-1]=data;f.to_csv(args.output/f'{name}_{count}.csv',index=False)
        row=dict(dummy_packets=count,clean_accuracy=clean_score,adversarial_accuracy=score(torch.tensor(out,device=device)),random_single_accuracy=score(torch.tensor(random_out,device=device)),uaps=uaps)
        rows.append(row);print(json.dumps(row))
    mt[['source_row']].to_csv(args.output/'source_rows.csv',index=False)
    report=dict(protocol='Algorithm 3, class-conditional UAP; first destination-to-source burst supplied by metadata; signed-log normalized coordinates',
        fit_sha256=sha(args.fit_csv),target_sha256=sha(args.target_csv),classifier_sha256=sha(checkpoint),
        fit_metadata_sha256=sha(args.fit_metadata),target_metadata_sha256=sha(args.target_metadata),
        iterations=args.iterations,batch_size=args.batch_size,step_size=args.step_size,seed=args.seed,iat_max_seconds=args.iat_max_seconds,
        limitations=['Sequence-space implementation, not packet replay validation.','Uses supplied CNN, not the original authors architecture search.','Random control is one run, not the 50 repetitions in the paper.','Metadata direction/burst selection and normalization must be confirmed by source generation.'],results=rows)
    (args.output/'generation.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--data-dir',type=Path,required=True);a.add_argument('--atol',type=float,default=1e-6)
    g=sub.add_parser('generate')
    for name in ['fit-csv','target-csv','fit-metadata','target-metadata','checkpoint-output']:
        g.add_argument('--'+name,type=Path,required=True)
    g.add_argument('--representation',choices=['signed-log'],required=True)
    g.add_argument('--iat-max-seconds',type=float,required=True)
    g.add_argument('--iterations',type=int,default=2000);g.add_argument('--batch-size',type=int,default=64)
    g.add_argument('--step-size',type=float,default=.01);g.add_argument('--seed',type=int,default=42)
    for parser in [a,g]:
        parser.add_argument('--levels',type=int,nargs='+',default=[1,3,5,7,10,12,15,17,20]);parser.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if any(n<1 or n>100 for n in args.levels):p.error('Levels must be between 1 and 100.')
    if args.command=='generate' and (args.iat_max_seconds<=1e-6 or args.iterations<1 or args.batch_size<1 or args.step_size<=0):p.error('Invalid generation settings.')
    if args.output.exists() and any(args.output.iterdir()):p.error('Use a new empty output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    (audit if args.command=='audit' else generate)(args)

if __name__=='__main__':main()
