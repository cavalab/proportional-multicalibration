import argparse
import json
from glob import glob
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Evaluate a method on a dataset.",
                                 add_help=False)
parser.add_argument('rdir', action='store', type=str)

args = parser.parse_args()

frames = []
for f in glob(args.rdir + '/*.json'):
    with open(f, 'r') as file:
        d = json.load(file)
        frames.append(d)
        df_results = pd.DataFrame.from_records(frames)
print(len(frames), 'records')

df_results = pd.DataFrame.from_records(frames)
# print(frames[0].keys())
print(df_results.groupby('algorithm')['random_state'].count())
print('stats:')
metrics = ['roc_auc', 'auprc', 'MC_loss', 'PMC_loss', 'DC_loss']
test_metrics = [m + '_test' for m in metrics]
print(df_results.groupby('algorithm')[test_metrics].mean().round(3))
