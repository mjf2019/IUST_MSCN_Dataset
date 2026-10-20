"""Evaluation-only controls for the approved structured Burst bank.

Copies existing Models to a fresh run; never modifies source models.
frozen-selector: freezes both learned selector heads, but retains Context
updates and paid-history state. matched-routing: identical uniform labels
and fixed destinations for learned/random, isolating prediction routing.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--burst-dir", type=Path, required=True)
    p.add_argument("--variant", choices=["frozen-selector", "matched-routing"], required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[5, 7, 10, 12, 15, 17, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--context-lr", type=float, default=.0005)
    p.add_argument("--budget-fraction", type=float, default=.2)
    p.add_argument("--eval-epsilon", type=float, default=.05)
    a = p.parse_args()
    source = a.source_run.resolve() / "Models"
    required = ["structured_bank_ddqn.pth", "protocol_preprocessing.joblib"]
    if any(not (source / name).is_file() for name in required):
        p.error("A complete structured-bank source run is required.")
    output = a.output.resolve()
    if output.exists():
        p.error("Use a fresh output directory; existing results are never overwritten.")
    import amcal_burst_structured_bank_protocol as bank
    original_episode = bank.episode

    def controlled_episode(*args, **kwargs):
        if kwargs.get("training", False):
            raise RuntimeError("This control is evaluation-only.")
        if a.variant == "frozen-selector":
            kwargs["learning"] = False
        result = original_episode(*args, **kwargs)
        metrics = result[0]
        metrics["ablation_variant"] = a.variant
        metrics["ablation_source_run"] = str(a.source_run.resolve())
        if a.variant == "frozen-selector":
            if metrics["route_updates"] != 0 or metrics["td_updates"] != 0:
                raise RuntimeError("Frozen-selector control unexpectedly updated weights.")
        return result

    output.mkdir(parents=True)
    shutil.copytree(source, output / "Models")
    (output / "ablation_config.json").write_text(json.dumps(
        {**vars(a), "source_models_unchanged": True}, default=str, indent=2))
    bank.episode = controlled_episode
    argv = [sys.argv[0], "evaluate", "--output", str(output),
            "--burst-dir", str(a.burst_dir.resolve()), "--levels", *map(str, a.levels),
            "--balanced-reward-reference", "--context-lr", str(a.context_lr),
            "--budget-fraction", str(a.budget_fraction), "--eval-epsilon", str(a.eval_epsilon),
            "--query-schedule", "policy" if a.variant == "frozen-selector" else "uniform",
            "--update-policy", "learned" if a.variant == "frozen-selector" else "fixed",
            "--gamma", ".95", "--selection-cost", ".0001", "--gain-weight", "1",
            "--retention-weight", "1", "--modes", "learned", "random",
            "--seed", str(a.seed), "--device", a.device, "--test-scaler", "training"]
    old_argv = sys.argv
    try:
        sys.argv = argv
        bank.main()
    finally:
        sys.argv = old_argv
        bank.episode = original_episode


if __name__ == "__main__":
    main()
