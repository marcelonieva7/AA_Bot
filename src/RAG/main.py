from src.config.db import qdrant_db
from src.LLM.main import chat
from src.RAG.prompts import system_prompt_v1

def rag(query, model='meta/llama-4-scout-17b-16e-instruct'):
    retrival = qdrant_db.search(query, limit=10, type='hybrid', fusion='RRF')
    answer = chat(system=system_prompt_v1(retrival), user=query, model=model)

    return answer