from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def cosine_similarity(left: str, right: str) -> float:
    model = get_embedding_model()
    vectors = model.encode([left, right], normalize_embeddings=True)
    score = float(np.dot(vectors[0], vectors[1]))
    return round(max(0.0, min(1.0, score)), 4)
