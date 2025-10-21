from fastembed import TextEmbedding, SparseTextEmbedding

def preload_models():
    # Modelo denso
    dense_model = TextEmbedding('jinaai/jina-embeddings-v2-base-es')
    dense_model.embed(["warmup"])
    print("✅ Dense model cached successfully!")

    # Modelo disperso (sparse)
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    sparse_model.embed(["warmup"])
    print("✅ Sparse model cached successfully!")

if __name__ == "__main__":
    preload_models()
