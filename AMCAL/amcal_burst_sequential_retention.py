"""Burst IAT sequential retention with a fixed label-free audit subset.
All adaptive state survives level transitions. Audit source IDs never occur
in adaptation streams at ANY level. Labels are charged per domain, even when
a source identity has appeared at another intensity. Observer labels never
enter the selector, Context gradients, replay, or reward references.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--burst-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--levels",nargs="+",type=int,default=[5,7,10,12,15,17,20])
    p.add_argument("--audit-fraction",type=float,default=.2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--query-schedule",choices=["policy","uniform"],default="policy")
    p.add_argument("--modes",nargs="+",choices=["learned","random"],default=["learned","random"])
    p.add_argument("--device",choices=["cpu","cuda","auto"],default="cuda")
    a=p.parse_args()
    if not .05<=a.audit_fraction<=.5:p.error("Audit fraction must be .05.. .5.")
    if len(a.levels)<2 or len(set(a.levels))!=len(a.levels):p.error("At least two distinct domains required.")
    if a.output.exists():p.error("Use a fresh output directory.")
    import joblib
    import numpy as np
    import torch
    import amcal_burst_structured_bank_protocol as bank
    core=bank.core
    core.device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else "cpu" if a.device=="auto" else a.device)
    models=a.source_run.resolve()/"Models"
    saved=torch.load(models/"structured_bank_ddqn.pth",map_location="cpu",weights_only=False)
    if saved["version"]!=1 or saved.get("config",{}).get("decoupled_query",False):
        p.error("Use the approved coupled v1 checkpoint.")
    for name,digest in saved["model_hashes"].items():
        if core.file_sha256(models/name)!=digest:p.error("Source model hash mismatch: "+name)
    artifact=joblib.load(models/"protocol_preprocessing.joblib")
    if core.file_sha256(a.burst_dir/"AdvBurstTrainedBurst_size_0_Step_2000.csv")!=artifact["source_sha256"]:
        p.error("Clean source differs from training provenance.")
    cfg=dict(saved["config"])
    cfg.update(seed=a.seed,classes=[str(c) for c in artifact["label_encoder"].classes_],
               context_lr=.0005,budget_fraction=.2,eval_epsilon=.05,
               query_schedule=a.query_schedule,update_policy="learned",bank_utility=False,
               budget_aware_policy=False,decoupled_query=False,context_alpha=cfg.get("context_alpha",.8))
    settings=SimpleNamespace(**cfg)
    core.SELECTION_COST=settings.selection_cost
    core.GAIN_WEIGHT=settings.gain_weight
    core.RETENTION_WEIGHT=settings.retention_weight
    core.REFERENCE_BATCH_SIZE,core.REFERENCE_HISTORY_SIZE=32,256
    domains={level:bank.read_domain(a.burst_dir,artifact,level,"test_ids") for level in a.levels}
    common=set.intersection(*(set(map(int,d[2])) for d in domains.values()))
    if len(common)<10:p.error("Insufficient shared source IDs for a fixed audit split.")
    ids=np.asarray(sorted(common))
    permutation=np.random.default_rng(a.seed+8121).permutation(ids)
    audit_ids=set(map(int,permutation[:max(1,int(len(ids)*a.audit_fraction))]))
    streams={};audits={}
    for level,(x,y,source_ids) in domains.items():
        audit=np.asarray([int(i) in audit_ids for i in source_ids])
        streams[level]=(x[~audit],np.asarray(y)[~audit],np.asarray(source_ids)[~audit])
        audits[level]=(x[audit],np.asarray(y)[audit])
        assert not (set(map(int,streams[level][2])) & audit_ids)
    clean=bank.read_domain(a.burst_dir,artifact,0,"val_ids")[:2]
    a.output.mkdir(parents=True)
    (a.output/"sequential_config.json").write_text(json.dumps({
        "source_run":str(a.source_run.resolve()),"seed":a.seed,"levels":a.levels,
        "audit_ids":sorted(audit_ids),"audit_fraction":a.audit_fraction,
        "context_lr":.0005,"epsilon":.05,"budget_fraction":.2,
        "query_schedule":a.query_schedule,"modes":a.modes,
        "protocol":"per-domain budget; source identity charged again per domain; all adaptive state retained",
        "audit_protocol":"fixed source IDs excluded from EVERY adaptation domain; observer-only labels",
        "training_config":cfg,"training_protocol":saved.get("training_protocol")},default=str,indent=2))
    observer_rows=[];stream_rows=[];summary=[]
    for mode in a.modes:
        core.set_seed(a.seed)
        cnn,contexts,initial,width=bank.models_from_source(models,artifact)
        agent=bank.Agent(width,core.device,a.seed,policy_state=saved["policy_state_dict"],gamma=settings.gamma)
        continuation={}
        history=bank.Reliability(len(settings.classes))
        matrix=[]
        def observe(stage,used,cap,clock):
            row=[]
            for level in [0,*a.levels]:
                x,y=clean if level==0 else audits[level]
                pred,_=bank.reference_observer(cnn,contexts,agent,x,y,history,used,cap,clock,mode,a.seed)
                acc=float(np.mean(pred==y))
                observer_rows.append(dict(mode=mode,stage=stage,after_level=0 if stage==0 else a.levels[stage-1],
                                          audit_level=level,n=len(y),accuracy=acc))
                if level!=0:row.append(acc)
            matrix.append(row)
        observe(0,0,int(.2*len(streams[a.levels[0]][1])),0)
        clean_initial=observer_rows[-len(a.levels)-1]["accuracy"]
        for stage,level in enumerate(a.levels,start=1):
            m,trace=bank.episode(settings,cnn,contexts,initial,agent,streams[level],mode=mode,
                                learning=mode=="learned",continuation=continuation)
            history=continuation["history"]
            m.update(level=level,sequential_stage=stage,audit_ids_excluded=True)
            prefix=a.output/f"{mode}_stage{stage}_level{level}"
            prefix.with_name(prefix.name+"_metrics.json").write_text(json.dumps(m,indent=2))
            bank.base.write_trace(prefix.with_name(prefix.name+"_trace.csv"),trace)
            stream_rows.append(dict(mode=mode,stage=stage,level=level,samples=m["samples"],
                                    online_accuracy=m["accuracy"],labels=m["labels"],cap=m["label_cap"],
                                    route_updates=m["route_updates"],td_updates=m["td_updates"]))
            observe(stage,m["labels"],m["label_cap"],m["samples"])
            print(f"Sequential {mode} | stage {stage}/{len(a.levels)} | level {level} | online {100*m['accuracy']:.2f}% | labels {m['labels']}/{m['label_cap']}",flush=True)
        r=np.asarray(matrix);k=len(a.levels)
        # R[i,j]: audit accuracy on domain j after i adaptation domains.
        forgetting=float(np.mean([max(r[i,j] for i in range(j+1,k+1))-r[k,j] for j in range(k-1)]))
        bwt=float(np.mean([r[k,j]-r[j+1,j] for j in range(k-1)]))
        clean_final=observer_rows[-k-1]["accuracy"]
        summary.append(dict(mode=mode,mean_final_audit_accuracy_pct=100*float(r[-1].mean()),
                            forgetting_pp=100*forgetting,backward_transfer_pp=100*bwt,
                            clean_initial_pct=100*clean_initial,clean_final_pct=100*clean_final,
                            clean_drop_pp=100*(clean_initial-clean_final),
                            total_labels=sum(row["labels"] for row in stream_rows if row["mode"]==mode)))
    write_csv(a.output/"combined_sequential_audit_matrix.csv",observer_rows)
    write_csv(a.output/"combined_sequential_stream.csv",stream_rows)
    write_csv(a.output/"combined_sequential_summary.csv",summary)
    print("\nSequential summary (forgetting/BWT/Drop in percentage points):",flush=True)
    for row in summary:
        print(f"{row['mode']:8s} | final audit {row['mean_final_audit_accuracy_pct']:.2f}% | forgetting {row['forgetting_pp']:+.2f} pp | BWT {row['backward_transfer_pp']:+.2f} pp | clean drop {row['clean_drop_pp']:+.2f} pp | labels {row['total_labels']}",flush=True)
    print("Saved:",a.output,flush=True)

if __name__=="__main__":main()
