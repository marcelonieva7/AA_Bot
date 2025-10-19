import json
import os
import pathlib

from tqdm import tqdm

from src.config.utils import load_ground_truth, load_FAQS
from src.LLM.main import Models
from src.RAG.main import rag
from src.RAG.prompts import system_prompt_v1

faqs_idx = load_FAQS(idx=True)
ground_truth = load_ground_truth()

def generate_answers(system_prompt, ground_truth, model):
	print(f'Generating answers for model: {model.name}')
	answers = {}
	BASE_DIR = pathlib.Path(__file__).parent.resolve()
	FILE_PATH = BASE_DIR / f'answers_{model.name}.json'

	if os.path.exists(f'{FILE_PATH}'):
	 	with open(f'{FILE_PATH}', 'r') as file:
				answers = json.load(file)
				

	def get_answer(query, retries=5):  
		try:
			answer_llm = rag(query=query, system_prompt=system_prompt, model=model.value)
			return answer_llm
		except Exception:
			retries -= 1
			if retries > 0:
				return get_answer(query, retries)
			else:
				print("ERROR: Max retries reached")

	for i, rec in tqdm(enumerate(ground_truth), total=len(ground_truth)):
		if str(i) in answers:
			continue

    
		answer_llm = get_answer(rec['question'])
		doc_id = rec['id']
		original_doc = faqs_idx[doc_id]
		answer_orig = original_doc['answer']

		answers[str(i)] = {
			'answer_llm': answer_llm,
			'answer_orig': answer_orig,
			'document': doc_id,
			'question': rec['question'],
		}

	with open(f'{FILE_PATH}', 'w') as file:
			json.dump(answers, file)
  
for m in Models:
    generate_answers(system_prompt=system_prompt_v1, ground_truth=ground_truth, model=m)