"""Console reporting only; metrics use percentages, forgetting uses points."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def print_level_result(name, result):
    online=result['online']
    print(f"{name} | level {result['level']} | online AC {online['AC']:.2f}% | "
          f"PR {online['PR']:.2f}% | RC {online['RC']:.2f}% | F1 {online['F1']:.2f}% | "
          f"labels {result['labels']}/{result['cap']} ({result['labels_percent']:.2f}%) | "
          f"original-domain forgetting {result['original_domain_forgetting']['AC']:+.2f} pp",
          flush=True)
    if result.get('target_after'):
        print(f"  Final target AC {result['target_after']['AC']:.2f}% | "
              f"source AC {result['source_before']['AC']:.2f}% -> {result['source_after']['AC']:.2f}%",
              flush=True)


def print_saved_summaries(output):
    print('\nSAVED TEST SUMMARY | arithmetic mean across levels | AC/PR/RC/F1 in percent',flush=True)
    print('Method         Budget Levels     AC     PR     RC     F1  FinalAC  SourceAC  Forget(pp) Labels(%)',flush=True)
    found=False
    for folder in ('Results','UniformResults'):
        path=Path(output)/folder/'summary.csv'
        if not path.exists():continue
        found=True
        with path.open(encoding='utf-8-sig',newline='') as source:
            groups=defaultdict(list)
            for row in csv.DictReader(source):
                groups[row['method'],float(row['budget_fraction'])].append(row)
        for (method,budget),rows in groups.items():
            average=lambda key:mean(float(row[key]) for row in rows)
            print(f"{method:<14} {budget*100:5.1f}% {len(rows):6d} "
                  + ' '.join(f'{average(key):6.2f}' for key in ('AC','PR','RC','F1'))
                  + f" {average('target_final_AC'):8.2f} {average('source_final_AC'):9.2f}"
                  + f" {average('forgetting_AC'):+11.2f} {average('labels_percent'):9.2f}",flush=True)
    if not found:print('No saved TEST summary found.',flush=True)
    print('FinalAC: same-stream target accuracy after adaptation; descriptive endpoint.',flush=True)
    print('SourceAC: final level-0 accuracy. Negative Forget(pp): improved level-0 accuracy.',flush=True)

