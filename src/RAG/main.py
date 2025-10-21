from opentelemetry.trace import Status, StatusCode

from src.LLM.main import chat
from src.monitoring.tracing import tracer
from src.RAG.prompts import system_prompt_v1

def rag(query, model='meta/llama-4-scout-17b-16e-instruct'):
    from src.config.db import qdrant_db

    with tracer.start_as_current_span("rag.pipeline") as span:
        span.set_attribute("rag.query_length", len(query))
        span.set_attribute("rag.query_preview", query[:500])
        try:
            with tracer.start_as_current_span("rag_search") as db_span:
                limit=10
                search_type='hybrid'
                fusion='RRF'
                retrival = qdrant_db.search(query, limit=limit, type=search_type, fusion=fusion)

                db_span.set_attribute("search_limit", limit)
                db_span.set_attribute("search_type", search_type)
                db_span.set_attribute("search_fusion_alg", fusion)

                db_span.set_attribute("retrieved_documents_count", len(retrival))

                with tracer.start_as_current_span("rag_search_results") as results_span:
                    docs_preview = []
                    for i, doc in enumerate(retrival, start=1):
                        if doc:
                            docs_preview.append({
                                "ranking": i,
                                "answer": doc["answer"][:200],
                                "question": doc["question"][:200],
                                "id": doc["id"],
                                "topic": doc["topic"],
                                "source": doc["source"]
                            })
                    results_span.set_attribute("documents", str(docs_preview))

            answer = chat(system=system_prompt_v1(retrival), user=query, model=model)
            span.set_status(Status(StatusCode.OK))

            return answer
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise