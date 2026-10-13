# GraphSAGE benchmark

Independent inductive GraphSAGE benchmark. Every flow is a node. A node receives
messages from its preceding neighbors within the same capture sequence. Train,
validation and test graphs are built separately, and test edges never cross a
partition boundary or point to future records.

```powershell
pip install -r CDR_MLC/benchmarks/GraphSAGE/requirements.txt
python CDR_MLC/benchmarks/GraphSAGE/graphsage_iust_mscn.py `
  --scenarios 1 2 3 4 5 6 7 `
  --seed 42 `
  --device cuda `
  --output CDR_MLC/outputs/graphsage_seed42
```

The default graph uses the three preceding records from the same sequence.
Inference time includes construction of the causal test graph.
