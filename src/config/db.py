from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from typing import Literal
import os
import logging

# ⚠️ Configurar ANTES de imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "4"

from src.config.embeddings import get_embeddings
from src.config.envs import settings

logger = logging.getLogger(__name__)

Mode = Literal["local"] | Literal["online"]

class Qdrant_DB():
    def __init__(self, qdrant_url, qdrant_api_key, collection, dense_model, sparse_model_name):
        self.collection = collection
        self.dense_model = dense_model
        self.sparse_model_name = sparse_model_name
        self._sparse_model = None  # Lazy loading
        
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30
        )
    
    @property
    def sparse_model(self):
        """Lazy loading del modelo sparse"""
        if self._sparse_model is None:
            logger.info(f"🔄 Loading sparse model: {self.sparse_model_name}")
            self._sparse_model = SparseTextEmbedding(
                model_name=self.sparse_model_name,
                threads=1  # ⚠️ Crucial para serverless
            )
            logger.info("✅ Sparse model loaded")
        return self._sparse_model

    def create_collection(self):
        # if collection already exists, delete it
        for c in self.client.get_collections().collections:
            if self.collection == c.name:
                self.client.delete_collection(collection_name=self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_model['dimensions'],
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )
        logger.info(f'✅ Collection created: {self.collection}')

    def create_points(self, documents):
        points = []
        id = 0

        for doc in documents:
            text = doc['question'] + ' ' + doc['answer']
            sparse_embeddings = list(self.sparse_model.embed(text))[0]
            point = models.PointStruct(
                id=id,
                vector={
                    'dense': models.Document(text=text, model=self.dense_model['name']),
                    'sparse': models.SparseVector(
                        indices=sparse_embeddings.indices.tolist(),
                        values=sparse_embeddings.values.tolist(),
                    )
                },
                payload={
                    "answer": doc['answer'],
                    "question": doc['question'],
                    "source": doc['source'],
                    "topic": doc['topic'],
                    "id": doc['id']
                }
            )
            points.append(point)
            id += 1
        
        return points

    def upsert_points(self, points, batch_size=10):
        for i in tqdm(range(0, len(points), batch_size)):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=True
            )

    def _search_builder(self, args):
        results = self.client.query_points(
            collection_name=self.collection,
            with_payload=True,
            **args,
        )
        return [r.payload for r in results.points]

    def _build_semantic_args(self, query, limit, embeddings=None):
        local_embeddings = models.Document(
            text=query,
            model=self.dense_model['name']
        )        
        args = {
            'limit': limit,
            'query': embeddings['dense'] if embeddings else local_embeddings,
            'using': "dense"
        }
        return args

    def _build_lexical_args(self, query, limit, embeddings=None):
        local_embeddings = list(self.sparse_model.embed(query))[0]
        args = {
            'limit': limit,
            'query': models.SparseVector(
                indices=embeddings['sparse']['indices'] if embeddings else local_embeddings.indices.tolist(),
                values=embeddings['sparse']['values'] if embeddings else local_embeddings.values.tolist(),
            ),
            'using': "sparse"
        }
        return args

    def _build_hybrid_args(self, query, limit, fusion_type, embeddings_mode):
        fusion = None
        match fusion_type:
            case 'DBSF':
                fusion = models.Fusion.DBSF
            case 'RRF':
                fusion = models.Fusion.RRF
            case _:
                raise Exception(f'invalid fusion type {fusion_type}')

        embeddings = None if embeddings_mode == 'local' else get_embeddings(query)

        semantic = self._build_semantic_args(query, limit=limit*2, embeddings=embeddings)
        lexical = self._build_lexical_args(query, limit=limit*2, embeddings=embeddings)
        args = {
            'prefetch': [
                models.Prefetch(**semantic),
                models.Prefetch(**lexical)
            ],
            'query': models.FusionQuery(fusion=fusion),
            'limit': limit
        }
        return args

    def search(self, query_txt, limit=5, type='semantic', fusion='DBSF', embeddings_mode: Mode="local"):
        args = {}
        match type:
            case 'semantic':
                args = self._build_semantic_args(query_txt, limit, embeddings_mode)
            case 'lexical':
                args = self._build_lexical_args(query_txt, limit, embeddings_mode)
            case 'hybrid':
                args = self._build_hybrid_args(query_txt, limit, fusion, embeddings_mode)
            case _:
                raise Exception(f'invalid search type {type}')

        results = self._search_builder(args)    
        return results


# Configuración
COLLECTION_NAME = "FAQ_AA"
DENSE_MODEL = {
    'name': 'jinaai/jina-embeddings-v2-base-es',
    'dimensions': 768
}
SPARSE_MODEL_NAME = 'Qdrant/bm25'

# ✅ Instancia única (lazy loading interno)
qdrant_db = Qdrant_DB(
    qdrant_api_key=settings.QDRANT_API_KEY,
    qdrant_url=settings.QDRANT_URL,
    collection=COLLECTION_NAME,
    dense_model=DENSE_MODEL,
    sparse_model_name=SPARSE_MODEL_NAME  # Solo el nombre
)