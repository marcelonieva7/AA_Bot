import json
import os

from tqdm import tqdm

from src.eval import prompts
from src.LLM.main import chat
from src.config.utils import load_FAQS
from src.config.paths import PROCESSED_DATA_DIR

faqs = load_FAQS()
GROUD_TRUTH_IDX_PATH = PROCESSED_DATA_DIR / 'Ground_Truth_IDX.json'
results = {}

if os.path.exists(GROUD_TRUTH_IDX_PATH):
    with open(GROUD_TRUTH_IDX_PATH, 'r') as file:
        results = json.load(file)
else:
	with open(GROUD_TRUTH_IDX_PATH, 'w') as file:
		json.dump(results, file)

def generate_questions(doc, retries=5):
	try:
		response = chat(system=prompts.generate_ground_truth(doc))
		questions = json.loads(response)
		results[doc_id] = questions["questions"]
		with open(GROUD_TRUTH_IDX_PATH, 'w') as file:
			json.dump(results, file)
	except Exception as e:
		print('error generating questions for doc ', doc_id, 'retrying...')
		if retries > 0:
			retries -= 1
			generate_questions(doc, retries)
		else:
			print(e)
			raise Exception("error generating questions, max retries reached for doc ", doc_id)
	
for doc in tqdm(faqs):
	doc_id = doc['id']
	if doc_id in results:
		continue
	generate_questions(doc)

final_results = []

for id, questions in results.items():
  for question in questions:
    final_results.append({
      'id': id,
      'question': question
    })

GROUD_TRUTH_PATH = PROCESSED_DATA_DIR / 'Ground_Truth.json'

with open(GROUD_TRUTH_PATH, 'w') as file:
  json.dump(final_results, file)
