from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FINAL_DATA_DIR = BASE_DIR / 'data' / 'final'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

GROUND_TRUTH_IDX_PATH = FINAL_DATA_DIR / 'Ground_Truth_IDX.json'
GROUND_TRUTH_PATH = FINAL_DATA_DIR / 'Ground_Truth.json'
FAQS_PATH = FINAL_DATA_DIR / "FAQS.json"
FAQS_IDX_PATH = FINAL_DATA_DIR / "FAQS_IDX.json"
ANSWERS_GPT_PATH = FINAL_DATA_DIR / "answers_gpt_20b.json"
ANSWERS_KIMI_PATH = FINAL_DATA_DIR / "answers_kimi_k2.json"
ANSWERS_LLAMA_PATH = FINAL_DATA_DIR / "answers_llama4_scout.json"