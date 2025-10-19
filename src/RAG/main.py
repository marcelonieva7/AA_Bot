from src.config.db import qdrant_db
from src.LLM.main import chat
from src.RAG.prompts import build_user_prompt, system_prompt_v1

def rag(query, model='openai/gpt-oss-20b'):
    retrival = qdrant_db.search(query, limit=10, type='hybrid', fusion='DBSF')
    user = build_user_prompt(query, retrival)
    answer = chat(system=system_prompt_v1, user=user, model=model)

    return answer