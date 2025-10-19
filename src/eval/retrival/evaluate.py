import pandas as pd
from pathlib import Path

from src.config.db import qdrant_db
from src.config.utils import load_ground_truth
from src.eval.retrival.metrics import evaluate

ground_truth = load_ground_truth()
results = {}

search_types = ['semantic', 'lexical', ['hybrid', 'DBSF'], ['hybrid', 'RRF']]

for i,s in enumerate(search_types):
  print(f'Evaluating {s} ({i+1}/{len(search_types)})')
  if isinstance(s, list):
    results[f'{s[0]}_{s[1]}'] = evaluate(ground_truth, lambda query: qdrant_db.search(query, type=s[0], fusion=s[1], limit=10))
  else:
    results[s] = evaluate(ground_truth, lambda query: qdrant_db.search(query, type=s, limit=10))

results_df = pd.DataFrame(results)
BASE_DIR = Path(__file__).resolve().parent
results_df.to_csv( BASE_DIR / 'Eval_results.csv')

print(results_df)