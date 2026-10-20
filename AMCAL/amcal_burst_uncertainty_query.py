"""Burst uncertainty-query ablation without retraining.
Replace ONLY query timing by current Frozen least confidence. Calibrate the
threshold at the source-validation (1-budget) quantile with no target access.
Retain learned RL prediction routing, update destinations and paid-only online
selector updates. This is not a wholly non-RL prediction baseline.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--burst-dir",type=Path,required=True)
    p.add_argument("--levels",nargs="+",type=int,default=[5,7,10,12,15,17,20])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--eval-epsilon",type=float,default=.05)
    p.add_argument("--budget-fraction",type=float,default=.2)
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="cuda")
    a=p.parse_args()
    if not 0<a.budget_fraction<1:p.error("Budget fraction must be between zero and one.")
    if not 0<=a.eval_epsilon<=1:p.error("Epsilon must be in [0,1].")
    source=a.source_run.resolve()/"Models"
    if not (source/"structured_bank_ddqn.pth").is_file():p.error("Trained structured-bank source required.")
    if a.output.exists():p.error("Use a fresh output directory.")
    import numpy as np
    import torch
    import amcal_burst_structured_bank_protocol as bank
    original_evaluate=bank.evaluate
    def uncertainty_evaluate(settings,models,artifact):
        saved=torch.load(models/"structured_bank_ddqn.pth",map_location="cpu",weights_only=False)
        if saved["version"]!=1 or saved.get("config",{}).get("decoupled_query",False):
            raise ValueError("Use the original coupled v1 bank checkpoint.")
        cnn,contexts,initial,width=bank.models_from_source(models,artifact)
        x,_,_=bank.read_domain(settings.burst_dir,artifact,0,"val_ids")
        z,_=bank.base_cache(cnn,x)
        uncertainty=(1-z.softmax(1).max(1).values).detach().cpu().numpy()
        settings.uncertainty_threshold=float(np.quantile(uncertainty,1-a.budget_fraction))
        print(f"Uncertainty-query control | SOURCE validation threshold {settings.uncertainty_threshold:.6f} | original RL routes/destinations retained",flush=True)
        metadata={"source_run":str(a.source_run.resolve()),"source_validation_rows":len(x),
                  "threshold":settings.uncertainty_threshold,"calibration_quantile":1-a.budget_fraction,
                  "score":"1 - max frozen softmax","test_labels_used_for_calibration":False,
                  "scope":"query timing only; original RL routing and update destinations",
                  "seed":a.seed}
        (settings.output/"uncertainty_control.json").write_text(json.dumps(metadata,indent=2))
        del cnn,contexts,initial,z,x
        return original_evaluate(settings,models,artifact)
    a.output.mkdir(parents=True)
    shutil.copytree(source,a.output/"Models")
    bank.evaluate=uncertainty_evaluate
    argv=[sys.argv[0],"evaluate","--burst-dir",str(a.burst_dir.resolve()),
          "--output",str(a.output.resolve()),"--levels",*map(str,a.levels),
          "--balanced-reward-reference","--context-lr",".0005","--budget-fraction",str(a.budget_fraction),
          "--query-schedule","policy","--update-policy","learned","--gamma",".95",
          "--selection-cost",".0001","--gain-weight","1","--retention-weight","1",
          "--eval-epsilon",str(a.eval_epsilon),"--modes","learned",
          "--seed",str(a.seed),"--device",a.device,"--test-scaler","training"]
    old=sys.argv
    try:
        sys.argv=argv;bank.main()
    finally:
        sys.argv=old;bank.evaluate=original_evaluate

if __name__=="__main__":main()
