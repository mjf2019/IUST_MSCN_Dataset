"""Short matched Context ablation: remove Transformer encoder only.
Both architectures reuse the same Frozen CNN and original Context's retained
weights, then run identical short supervised Context refinement and router
training. TRAIN gradients and validation-only checkpoint selection; no target
test rows used. Evaluate through this wrapper to reconstruct the architecture.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["train","evaluate"])
    p.add_argument("--architecture",choices=["full","no-transformer"],required=True)
    p.add_argument("--source-run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--burst-dir",type=Path,required=True)
    p.add_argument("--levels",nargs="+",type=int)
    p.add_argument("--context-pretrain-epochs",type=int,default=5)
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--patience",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["cpu","cuda","auto"],default="cuda")
    a=p.parse_args()
    if min(a.epochs,a.context_pretrain_epochs,a.patience)<1 or a.patience>5:
        p.error("Positive epochs and patience 1..5 required.")
    import joblib
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    import amcal_burst_structured_bank_protocol as bank
    core=bank.core
    OriginalContext=core.ContextAwareNetwork
    if a.architecture=="no-transformer":
        class NoTransformerContext(OriginalContext):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.transformer=nn.Identity()
        core.ContextAwareNetwork=NoTransformerContext
    core.device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else "cpu" if a.device=="auto" else a.device)
    core.set_seed(a.seed)
    output=a.output.resolve();burst=a.burst_dir.resolve()
    config_path=output/"architecture_ablation.json"
    if a.command=="train":
        if a.source_run is None:p.error("--source-run required for train.")
        if output.exists():p.error("Use a fresh output directory.")
        source=a.source_run.resolve()/"Models"
        for name in ["initial_cnn_model.pth","adversarial_best_context.pth","adversarial_best_dqn.pth","protocol_preprocessing.joblib"]:
            if not (source/name).is_file():p.error("Missing source file: "+name)
        levels=a.levels or [0,1,3]
        output.mkdir(parents=True)
        prepared=output/"PreparedSource";models=prepared/"Models"
        shutil.copytree(source,models)
        context_path=models/"adversarial_best_context.pth"
        checkpoint=torch.load(context_path,map_location="cpu",weights_only=False)
        if a.architecture=="no-transformer":
            checkpoint["context_model_state_dict"]={k:v for k,v in checkpoint["context_model_state_dict"].items()
                                                    if not k.startswith("transformer.")}
        checkpoint["context_architecture_ablation"]=a.architecture
        torch.save(checkpoint,context_path)
        artifact=joblib.load(models/"protocol_preprocessing.joblib")
        cnn,contexts,initial,_=bank.models_from_source(models,artifact)
        context=contexts[0]
        optimizer=torch.optim.Adam(context.parameters(),lr=.0001,weight_decay=1e-4)
        domains={(level,split):bank.read_domain(burst,artifact,level,split)
                 for level in levels for split in ["train_ids","val_ids"]}
        tx=torch.cat([domains[level,"train_ids"][0] for level in levels])
        ty=torch.tensor(np.concatenate([domains[level,"train_ids"][1] for level in levels]),device=core.device,dtype=torch.long)
        loader=DataLoader(TensorDataset(tx,ty),batch_size=64,shuffle=True)
        best=float("inf");stale=0;best_state=deepcopy(context.state_dict());log=[]
        for epoch in range(a.context_pretrain_epochs):
            context.train()
            for x,y in loader:
                optimizer.zero_grad()
                with torch.no_grad():frozen_loss=nn.functional.cross_entropy(cnn(x),y)
                loss=bank.context_objective(nn.functional.cross_entropy(cnn(context(x)),y),frozen_loss,.8)
                loss.backward();nn.utils.clip_grad_norm_(context.parameters(),1);optimizer.step()
            context.eval();scores=[]
            with torch.no_grad():
                for level in levels:
                    vx,vy,_=domains[level,"val_ids"];total=0.
                    for start in range(0,len(vy),64):
                        y=torch.tensor(vy[start:start+64],device=core.device,dtype=torch.long)
                        total+=nn.functional.cross_entropy(cnn(context(vx[start:start+64])),y,reduction="sum").item()
                    scores.append(total/len(vy))
            val=float(np.mean(scores));log.append({"epoch":epoch+1,"validation_ce":val})
            print(f"{a.architecture} Context refinement {epoch+1}/{a.context_pretrain_epochs} | validation CE {val:.4f}",flush=True)
            if val<best:best=val;stale=0;best_state=deepcopy(context.state_dict())
            else:stale+=1
            if stale>=a.patience:break
        checkpoint["context_model_state_dict"]=best_state
        torch.save(checkpoint,context_path)
        config={"architecture":a.architecture,"source_run":str(a.source_run.resolve()),
                "train_levels":levels,"context_pretrain_epochs":a.context_pretrain_epochs,
                "router_epochs":a.epochs,"patience":a.patience,"seed":a.seed,
                "refinement_log":log,"test_used":False}
        config_path.write_text(json.dumps(config,indent=2))
        argv=["train","--source-run",str(prepared),"--output",str(output/"Bank"),
              "--levels",*map(str,levels),"--epochs",str(a.epochs),"--patience",str(a.patience),
              "--router-warmup-episodes","1"]
    else:
        config=json.loads(config_path.read_text())
        if config["architecture"]!=a.architecture:p.error("Architecture differs from training.")
        argv=["evaluate","--output",str(output/"Bank"),"--levels",*map(str,a.levels or [5,7,10,12,15,17,20]),
              "--modes","learned","random"]
    argv+=["--burst-dir",str(burst),"--balanced-reward-reference",
           "--context-lr",".0005","--budget-fraction",".2","--query-schedule","policy",
           "--update-policy","learned","--gamma",".95","--selection-cost",".0001",
           "--gain-weight","1","--retention-weight","1","--train-epsilon",".2",
           "--eval-epsilon",".05","--seed",str(a.seed),"--device",a.device]
    old=sys.argv
    try:
        sys.argv=[str(Path(__file__)),*argv];bank.main()
    finally:
        sys.argv=old
        core.ContextAwareNetwork=OriginalContext

if __name__=="__main__":main()
