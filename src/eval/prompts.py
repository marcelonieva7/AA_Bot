from typing import TypedDict

class Question(TypedDict):
  question: str
  answer: str
  source: str

class Ground_Truth(TypedDict):
  questions: list[str]

def generate_ground_truth(doc: Question) -> Ground_Truth:
  return f"""
You are tasked with generating evaluation data for a FAQ system about Alcolicos Anonimos (Alcoholics Anonymous) AA.
Your role is to emulate a real person searching for information about AA.

Instructions:
- Formulate 5 different questions that a person might ask.
- The questions should be entirely in spanish.
- The questions must be relevant to the provided FAQ record.
- Each question must be complete, natural, and not too short (avoid one-word or overly generic questions).
- Do not copy long fragments from the record. Use paraphrasing when possible.
- Avoid repeating the exact wording of the provided question/answer.
- Make sure the questions can reasonably be answered by the FAQ record.

FAQ record:
source: {doc["source"]}
question: {doc["question"]}
answer: {doc["answer"]}

Output format:
Return only valid JSON (no code blocks, no explanations):

{{
  "questions": [
    "question1",
    "question2",
    "question3",
    "question4",
    "question5"
  ]
}}
""".strip()