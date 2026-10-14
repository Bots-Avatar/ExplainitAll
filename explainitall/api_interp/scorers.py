from typing import List, Sequence

import numpy as np


class SemanticScorer:
    """
    Семантическая близость ответов по косинусу эмбеддингов.
    embedder: любой объект с методом encode(list[str]) -> массив (n, d), например SentenceTransformer
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self.embedder.encode(list(texts)), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def similarities(self, reference: str, texts: Sequence[str]) -> List[float]:
        """Близость каждого текста к эталону в [0, 1]; отрицательный косинус считается нулевой близостью"""
        if not texts:
            return []
        vectors = self._encode([reference, *texts])
        sims = vectors[1:] @ vectors[0]
        return [float(s) for s in np.clip(sims, 0.0, 1.0)]

    def distances(self, reference: str, texts: Sequence[str]) -> List[float]:
        """Насколько каждый текст отличается от эталона: 1 - близость"""
        return [1.0 - s for s in self.similarities(reference, texts)]
