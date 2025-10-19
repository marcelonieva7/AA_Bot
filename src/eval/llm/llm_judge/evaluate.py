import json
import pathlib
import os

from tqdm import tqdm

from src.LLM.main import chat
from src.eval.llm.llm_judge.prompts import prompt1_template, prompt2_template
from src.config.paths import ANSWERS_GPT_PATH, ANSWERS_LLAMA_PATH, ANSWERS_KIMI_PATH

with open(ANSWERS_GPT_PATH, 'r') as f:
    answers_gpt = json.load(f)

with open(ANSWERS_LLAMA_PATH, 'r') as f:
    answers_llama = json.load(f)

with open(ANSWERS_KIMI_PATH, 'r') as f:
    answers_kimi = json.load(f)


MODELS = [
    {'name': 'GPT', 'answers': answers_gpt},
    {'name': 'LLAMA', 'answers': answers_llama},
    {'name': 'KIMI', 'answers': answers_kimi}
]

PROMPTS = [
    {'name': 'prompt1', 'template': prompt1_template},
    {'name': 'prompt2', 'template': prompt2_template}
]

BASE_DIR = pathlib.Path(__file__).resolve().parent


def get_evaluation(doc, prompt_template, id, retries=5):
    try:
        response = chat(system=prompt_template.format(**doc))
        evaluation = json.loads(response)
        return evaluation
        
    except Exception as e:
        print('error generating questions for doc ', id, 'retrying...')
        if retries > 0:
            retries -= 1
            get_evaluation(doc, prompt_template, id, retries)
        else:
            print(e)
            print("error getting eval, max retries reached for doc ", id)

for m in MODELS:
    for p in PROMPTS:
        results = []
        file_name = f'{m["name"]}_{p["name"]}.json'
        if file_name in os.listdir(BASE_DIR):
            print(f'Skipping {m["name"]} with {p["name"]}')
            continue
        
        print(f'Evaluating {m["name"]} with {p["name"]}')
        for answer in tqdm(list(m["answers"].values())):
            results.append(get_evaluation(answer, p["template"], answer["document"]))

        with open(BASE_DIR / file_name, 'w') as f:
            json.dump(results, f)




    
    