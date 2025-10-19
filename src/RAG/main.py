from src.config.db import qdrant_db
from src.LLM.main import chat
from src.RAG.prompts import build_user_prompt

def rag(query, system_prompt, model='openai/gpt-oss-20b'):
    retrival = qdrant_db.search(query, limit=10, type='hybrid', fusion='DBSF')
    user = build_user_prompt(query, retrival)
    answer = chat(system=system_prompt, user=user, model=model)

    return answer