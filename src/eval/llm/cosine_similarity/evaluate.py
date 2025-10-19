import json
import pathlib
import os

import pandas as pd
from fastembed import TextEmbedding
from tqdm import tqdm

from src.config.paths import ANSWERS_GPT_PATH, ANSWERS_LLAMA_PATH, ANSWERS_KIMI_PATH

DENSE_MODEL = 'jinaai/jina-embeddings-v2-base-es'

embedding_model = TextEmbedding(DENSE_MODEL)

embeddings_org_cache = {}

with open(ANSWERS_GPT_PATH, 'r') as f:
    answers_gpt = json.load(f)

with open(ANSWERS_LLAMA_PATH, 'r') as f:
    answers_llama = json.load(f)

with open(ANSWERS_KIMI_PATH, 'r') as f:
    answers_kimi = json.load(f)

def cosine(doc):
    def embed(txt):
        return list(embedding_model.embed([txt]))[0]

    embedding_answer_org = None

    if doc['document'] in embeddings_org_cache:
        embedding_answer_org = embeddings_org_cache[doc['document']]
    else:
        embedding_answer_org = embed(doc['answer_orig'])
        embeddings_org_cache[doc['document']] = embedding_answer_org
        
    embedding_answer_llm = embed(doc['answer_llm'])

    return (embedding_answer_org.dot(embedding_answer_llm))


models_answers = [
    {'answers': answers_gpt, 'name': 'gpt_20b'},
    {'answers': answers_llama, 'name': 'llama4_scout'},
    {'answers': answers_kimi, 'name': 'kimi_k2'}
]

BASE_DIR = pathlib.Path(__file__).resolve().parent

# create and save results
for m in models_answers:
    FILE = f'similarity_{m["name"]}.csv'
    if FILE in os.listdir(BASE_DIR):
        df = pd.read_csv(BASE_DIR / FILE)
        print(f'\nResults Similarity for {m['name']}')
        print(df.describe())
        continue

    similarity = []

    for doc in tqdm(m['answers'].values()):
        sim = cosine(doc)
        similarity.append(sim)

    df = pd.DataFrame(similarity)
    print(m['name'])
    print(df.describe())
    
    df.to_csv(BASE_DIR / FILE, index=False)