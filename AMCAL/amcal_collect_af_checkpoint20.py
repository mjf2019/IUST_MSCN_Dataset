"""Collect existing checkpoint-AF online and paired endpoint outputs; no fitting."""
import argparse
import json
from pathlib import Path
import amcal_final_multiseed as io
from burst_table_completion_utils import LEVELS, METRICS, table_row, aggregate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    args = p.parse_args()
    runs = io.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'protocol_runs'
    rows,inputs = [],{}
    seeds = [42,48,54,62,72]
    for seed in seeds:
        root = runs/f'AF_checkpoint20_online_seed{seed}_v2'
        done = io.read(root/'evaluation_complete.json')
        recipe = io.read(root/'recipe.json')
        if (done['cases']!=7 or
                done['summary_sha256']!=io.digest(root/'Results'/'summary_by_intensity.csv') or
                done['recipe_sha256']!=io.digest(root/'recipe.json') or
                recipe['seed']!=seed or recipe['train_levels']!=[0] or recipe['budget_fraction']!=.2):
            raise ValueError('Incomplete or changed AF checkpoint20 protocol.')
        inputs[str(seed)] = dict(recipe_sha256=io.digest(root/'recipe.json'),
                                 summary_sha256=done['summary_sha256'])
        for level in LEVELS:
            case = root/'Results'/'Burst'/f'level_{level}'
            complete = io.read(case/'completed.json')
            if complete['fingerprint']['recipe_sha256']!=io.digest(root/'recipe.json'):
                raise ValueError('AF case recipe changed.')
            for relative,sha in complete['hashes'].items():
                if io.digest(root/relative)!=sha:
                    raise ValueError('AF saved model or prediction file changed: '+relative)
            m = io.read(case/'metrics.json')
            row = table_row(args.dataset,'AF-checkpoint20-online',seed,level,
                {k:m['online_'+k] for k in METRICS},
                {k:m['same_domain_initial_'+k] for k in METRICS},
                {k:m['same_domain_final_'+k] for k in METRICS},
                {k:m['original_initial_'+k] for k in METRICS},
                {k:m['original_final_'+k] for k in METRICS},m['samples'],m['labels'])
            row.update(target_observer_rows=m['observer_samples'],
                       original_observer_rows=m['reference_samples'])
            rows.append(row)
    target = runs/'AF_checkpoint20_five_seed_tables_v1'
    io.write(target/'inputs.json',inputs)
    overall = aggregate(target/'Results',rows,seeds,'AF-checkpoint20-online')
    print('AF ALL SEEDS COMPLETE '+json.dumps(overall,sort_keys=True),flush=True)
    print('OUTPUT: '+str(target),flush=True)


if __name__=='__main__':
    main()
