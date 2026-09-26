# CPU-only inference resource benchmark

This benchmark isolates deployment inference from model training. Every method
is trained on the complete **Low** congestion level, saved locally, reloaded in
a fresh process, and evaluated on the same deterministic **Medium** sample.
Training may use CUDA; measured inference is CPU-only.

The directory `artifacts/` contains models and the prepared sample. Both
`artifacts/` and `results/` are intentionally excluded from Git.

## Fairness controls

- A single fixed Medium sample is shared by all methods.
- The sample contains 2,000 scored rows by default plus causal warm-up rows.
- Moving-window state is initialized only from preceding rows of the same
  sequence; future rows are never used.
- Model loading is measured separately and excluded from inference latency.
- Every method receives the same CPU-thread budget.
- CUDA is disabled in each measured child process.
- Streaming is the default deployment protocol: records arrive in capture
  order, batch size is exactly one, and one prediction is completed before the
  next scored record is submitted.
- Causal state is retained independently per sequence. Context-only warm-up
  rows initialize moving windows and are excluded from scored latency.
- One complete warm-up run precedes five measured repetitions by default.
- Streaming p50/p95/p99 are computed from individual end-to-end record
  latencies, not by dividing batch latency by the number of records.
- Accuracy is computed on the marked 2,000 rows. Throughput is reported both
  for all processed input rows and for scored rows.
- Original CDR-MLC does not compute oracle routes at inference.
- In streaming mode MF-CDR-MLC is reported in two execution modes using the
  same saved model:
  - `MF-Sequential`: expert and utility banks are evaluated serially.
  - `MF-Parallel`: three experts run concurrently, followed by three concurrent
    utility estimators. Dependency order and predictions are unchanged.
  Persistent worker threads are reused across arrivals. `MF-Pipelined` remains
  available only in the optional offline batch-throughput protocol because a
  full-batch pipeline does not represent single-arrival response latency.

AF intrinsically needs labeled target adaptation. Its default 1% Medium
calibration prefix is disjoint from the fixed Medium tail used for timing. The
calibration/training phase is never included in inference timing.

## 1. Prepare the fixed Medium sample

```powershell
python CDR_MLC/benchmarks/inference_resources/prepare_inference_sample.py --sample-size 2000 --max-context-window 50
```

## 2. Train and save all models

CUDA is used automatically when available:

```powershell
python CDR_MLC/benchmarks/inference_resources/train_models.py --methods rf original-cdr mf-cdr 1d-cnn ft-transformer graphsage scarf dfe af --device cuda --window 3 --congestion-window 50 --expert-trees 20 --utility-trees 10 --meta-trees 20 --rf-trees 110 --seed 42
```

To train on CPU, replace `--device cuda` with `--device cpu`. Individual models
can be retrained by passing only their method name to `--methods`.

## 3. Run CPU-only inference measurement

The following command starts a fresh process for each saved model:

```powershell
python CDR_MLC/benchmarks/inference_resources/run_all.py --mode streaming --cpu-threads 3 --warmup-runs 1 --repeats 5
```

The former offline batch-throughput protocol is still available explicitly:

```powershell
python CDR_MLC/benchmarks/inference_resources/run_all.py --mode batch --cpu-threads 3 --warmup-runs 3 --repeats 30
```

The combined result is written to:

```text
CDR_MLC/benchmarks/inference_resources/results/inference_resource_summary.csv
```

Per-method repetitions and stage measurements are stored below the corresponding
method directory. MF stage columns include TTFEF, router/preprocessing,
congestion context, expert bank, utility bank, meta-fusion, and end-to-end time.

## Reproducibility note

Run the benchmark while the machine is otherwise idle and keep the CPU power
plan, thread count, Python environment, and hardware unchanged. For the paper,
report the CPU model, RAM, OS, Python version, library versions, thread budget,
sample size, warm-up count, and repetition count.
