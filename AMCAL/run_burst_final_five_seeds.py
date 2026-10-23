"""Complete five evaluation seeds for the FINAL Burst comparison, preserving old runs.

Online methods reuse fixed seed-42 offline checkpoints. Seeds alter stream order
and online randomness, not offline training. AF is separately calibrated/refitted
for each seed and is explicitly an offline few-shot reference.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

LEVELS = [5, 7, 10, 12, 15, 17, 20]
SEEDS = [42, 48, 54, 62, 72]
GROUPS = {
    "AMCAL": ["amcal"],
    "Baselines": ["cnn", "dfe", "arf", "calmid", "micfoal", "ral5"],
    "AF": ["af_cal"],
}
REPO = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_summary(m, seed, level, n, window=False):
    e = m.get("execution_config", {})
    return (e.get("seed") == seed and e.get("budget") == .2
            and not e.get("test_scaler") and not e.get("unseen_rows")
            and m.get("level") == level and m.get("samples") == n
            and m.get("label_cap") == int(.2 * n)
            and m.get("labels", n + 1) <= int(.2 * n)
            and m.get("training_provenance", {}).get("train_levels") == [0, 1, 3]
            and (m.get("ral_online_window", {}).get("size") == 5 if window
                 else not e.get("ral_online_update_window")))


def amcal_execution(e, seed):
    return (e.get("seed") == seed and e.get("budget_fraction") == .2
            and e.get("context_lr") == .00025 and not e.get("continual")
            and e.get("context_count") == 1 and e.get("action_count") == 3
            and not e.get("architecture_ablation")
            and e.get("query_budget_bonus", {}).get("strength") == 1
            and e.get("paid_replay", {}).get("mode") == "batch5"
            and e.get("config", {}).get("memory_ce_weight") == .1
            and e.get("training_protocol", {}).get("levels") == [0, 1, 3])


def candidate(paths, validate, trace_name):
    matches = []
    for path in paths:
        m = read(path)
        if validate(m):
            trace = path.with_name(trace_name(path))
            matches.append((path, trace, m))
    # Prefer a complete trace, then the most recently produced matching result.
    return max(matches, key=lambda v: (v[1].exists(), v[0].stat().st_mtime)) if matches else None


def locations(dataset):
    root = REPO / "AMCAL" / ("ANT_Burst_" + dataset + "_TEST")
    runs = root / "protocol_runs"
    main = runs / ("paironly_batch5_trainseed42" if dataset == "IAT"
                   else "light_ps_paironly_batch5_trainseed42") / "paironly_batch5"
    common = runs / ("baselines_levels1_3_seed42" if dataset == "IAT"
                     else "baselines_levels1_3_rf10_seed42")
    sources = {
        "cnn": common, "dfe": common,
        "arf": runs / "arf_levels1_3_shared_seed42" if dataset == "IAT" else common,
        "calmid": runs / "calmid_micfoal_levels1_3_seed42",
        "micfoal": runs / "calmid_micfoal_levels1_3_seed42",
    }
    return root, runs, main, sources


def find(dataset, method, seed, level):
    root, runs, main, sources = locations(dataset)
    n = 1073 if dataset == "IAT" else 886
    if method == "amcal":
        matches = []
        for ep in (main / "Results").glob("evaluation_*/execution.json"):
            e = read(ep)
            if amcal_execution(e, seed):
                path = ep.with_name("learned_level" + str(level) + "_metrics.json")
                trace = ep.with_name("learned_level" + str(level) + "_trace.csv")
                if path.exists():
                    m = read(path)
                    if m.get("samples") != n or m.get("labels", n + 1) > int(.2*n):
                        raise ValueError("Unexpected AMCAL stream/budget: " + str(path))
                    matches.append((path, trace, m))
        return max(matches, key=lambda v: (v[1].exists(), v[0].stat().st_mtime)) if matches else None
    if method == "af_cal":
        paths = runs.glob("af_author_burst_seed" + str(seed) + "_*/af_fewshot/level"
                          + str(level) + "_shot5/metrics.json")
        def valid(m):
            e = m.get("execution", {})
            p = e.get("parameters", {})
            return (e.get("seed") == seed and m.get("shots") == 5
                    and m.get("level") == level and m.get("samples") == n
                    and e.get("method") == "AF-author-structure-Burst-PyTorch-port"
                    and e.get("test_per_class") == 0 and m.get("target_calibration_labels") == 30
                    and p.get("iterations") == 1000 and p.get("tuning_epochs") == 30)
        return candidate(paths, valid, lambda p: "predictions.csv")
    folder = runs / ("ral_recent5_seed" + str(seed)) if method == "ral5" else sources[method]
    native = "ral" if method == "ral5" else method
    paths = (folder / "Results").glob(native + "_per" + str(level) + "_seed" + str(seed) + "_*_metrics.json")
    return candidate(paths, lambda m: valid_summary(m, seed, level, n, method == "ral5"),
                     lambda p: p.name.replace("_metrics.json", "_trace.csv"))


def invoke(args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True)


def complete(dataset, method, seed):
    if all((v := find(dataset, method, seed, level)) and v[1].exists() for level in LEVELS):
        print("Reusing complete:", dataset, method, seed, flush=True)
        return
    root, runs, main, sources = locations(dataset)
    burst = root / "Dataset_official_v1" / ("AdvBurst_FTSC-" + dataset)
    if method == "amcal":
        if not (main / "Models/final_enhancement.pth").exists():
            raise FileNotFoundError("Final trained checkpoint missing: " + str(main))
        invoke(["AMCAL/amcal_final_enhancement.py", "evaluate", "--output", main,
                "--burst-dir", burst, "--levels", *LEVELS, "--seed", seed,
                "--budget-fraction", ".2", "--context-lr", ".00025",
                "--context-count", "1", "--paid-replay-mode", "batch5",
                "--query-budget-bonus", "1", "--online-memory-ce-weight", ".1",
                "--evaluation-tag", "final_five_seeds_bonus1_batch5_lr0p00025",
                "--device", "cuda"])
    elif method == "af_cal":
        source = runs / "structured_bank3_policy_lr0p0005_seed42"
        out = runs / ("af_author_burst_seed" + str(seed) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        invoke(["AMCAL/amcal_af_paper_burst.py", "--source-run", source,
                "--burst-dir", burst, "--output", out / "af_fewshot",
                "--levels", *LEVELS, "--shots", "5", "--seed", seed, "--device", "cuda"])
    else:
        if method == "ral5":
            source = runs / ("ral_paper_rf10_prefix_levels1_3_seed42" if dataset == "IAT"
                             else "baselines_levels1_3_rf10_seed42") / "Models"
            folder = runs / ("ral_recent5_seed" + str(seed))
            (folder / "Models").mkdir(parents=True, exist_ok=True)
            for name in ["ral.joblib", "protocol_preprocessing.joblib"]:
                src, dst = source / name, folder / "Models" / name
                if not src.exists():
                    raise FileNotFoundError(src)
                if dst.exists() and sha(dst) != sha(src):
                    raise ValueError("Preserving mismatched checkpoint: " + str(dst))
                if not dst.exists():
                    shutil.copy2(src, dst)
            native = "ral"
        else:
            folder, native = sources[method], method
        if not (folder / "Models" / (native + ".joblib")).exists():
            raise FileNotFoundError("Baseline checkpoint missing: " + str(folder))
        args = ["AMCAL/amcal_burst_baselines.py", "evaluate", "--methods", native,
                "--output", folder, "--burst-dir", burst, "--levels", *LEVELS,
                "--seed", seed, "--budget-fraction", ".2",
                "--device", "cuda" if method in ("cnn", "dfe") else "cpu"]
        if method == "ral5":
            args += ["--ral-budget-mode", "prefix", "--ral-update-window", "5"]
        invoke(args)
    for level in LEVELS:
        v = find(dataset, method, seed, level)
        if not v or not v[1].exists():
            raise ValueError("Incomplete output after execution: " + str((dataset, method, seed, level)))


def derived(trace, original):
    cm = [[0] * 6 for _ in range(6)]
    n = 0
    with trace.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            y, p = int(row["truth"]), int(row["prediction"])
            if not 0 <= y < 6 or not 0 <= p < 6:
                raise ValueError("Unexpected class in " + str(trace))
            cm[y][p] += 1
            n += 1
    if n != original["samples"]:
        raise ValueError("Trace sample count mismatch: " + str(trace))
    ac = sum(cm[k][k] for k in range(6)) / n
    ps, rs, fs = [], [], []
    for k in range(6):
        tp = cm[k][k]
        support, predicted = sum(cm[k]), sum(row[k] for row in cm)
        p, r = tp / predicted if predicted else 0., tp / support if support else 0.
        ps.append(p); rs.append(r)
        fs.append(2*p*r/(p+r) if p+r else 0.)
    f1 = sum(fs) / 6
    if abs(ac-original["accuracy"]) > 1e-10 or abs(f1-original["macro_f1"]) > 1e-10:
        raise ValueError("Saved metrics disagree with trace: " + str(trace))
    return dict(accuracy=ac, macro_precision=sum(ps)/6, macro_recall=sum(rs)/6,
                macro_f1=f1, confusion_matrix=cm, trace_sha256=sha(trace))


def collect(dataset, methods, group):
    _, runs, _, _ = locations(dataset)
    bundle = runs / "paper_final_five_seeds"
    bundle.mkdir(parents=True, exist_ok=True)
    coverage = []
    for method in methods:
        for seed in SEEDS:
            for level in LEVELS:
                v = find(dataset, method, seed, level)
                entry = dict(dataset=dataset, method=method, seed=seed, level=level)
                if not v:
                    entry["status"] = "missing_metrics"
                else:
                    path, trace, m = v
                    entry.update(status="complete" if trace.exists() else "missing_trace",
                                 source=str(path.relative_to(REPO)).replace("\\", "/"),
                                 source_sha256=sha(path), original=m)
                    entry["metrics"] = derived(trace, m) if trace.exists() else dict(
                        accuracy=m["accuracy"], macro_f1=m["macro_f1"])
                    out = bundle / method / ("seed" + str(seed)) / ("level" + str(level) + ".json")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(entry, indent=2), encoding="utf-8")
                coverage.append({k:v for k,v in entry.items() if k not in ("original", "metrics")})
    manifest = dict(version=1, dataset=dataset, group=group, seeds=SEEDS, levels=LEVELS,
                    online_training_seed=42,
                    protocol="Online evaluation seeds with fixed offline checkpoints; AF independently fitted/calibrated per seed",
                    entries=coverage)
    (bundle / ("coverage_" + group + ".json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for method in methods:
        own = [r for r in coverage if r["method"] == method]
        print(dataset, method, "complete", sum(r["status"] == "complete" for r in own),
              "/35; missing traces", sum(r["status"] == "missing_trace" for r in own),
              "; missing metrics", sum(r["status"] == "missing_metrics" for r in own), flush=True)
    print("Push this small JSON-only evidence folder:", bundle, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["IAT", "PS"], required=True)
    p.add_argument("--group", choices=[*GROUPS, "All"], default="All")
    p.add_argument("--action", choices=["collect", "run"], default="collect")
    a = p.parse_args()
    methods = sum(GROUPS.values(), []) if a.group == "All" else GROUPS[a.group]
    if a.action == "run":
        for method in methods:
            for seed in SEEDS:
                complete(a.dataset, method, seed)
    collect(a.dataset, methods, a.group)


if __name__ == "__main__":
    main()
