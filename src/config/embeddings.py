import json
import requests
from typing import TypedDict, List

from src.config.envs import settings

URL = settings.EMBEDDINGS_URL

class Sparse(TypedDict):
    values: List[float]
    indices: List[float]

class Res(TypedDict):
    sparse: Sparse
    dense: List[float]

def get_embeddings(query: str) -> Res:
    if not URL:
        raise Exception('NO EMBEDDINGS URL')
    response = requests.post(URL, json={"text": query})

    return json.loads(response.text)