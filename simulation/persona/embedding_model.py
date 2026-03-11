import hashlib
import re

import numpy as np


class _HashEmbeddingBackend:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, text: str, convert_to_numpy=True, show_progress_bar=False):
        del convert_to_numpy, show_progress_bar

        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            tokens = [text.lower()]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vec[index] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class EmbeddingModel:
    def __init__(self, device, backend: str = "auto") -> None:
        self.device = device
        self.backend = "hash"
        self.model = _HashEmbeddingBackend()

        if backend in {"auto", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                if backend == "sentence_transformers":
                    raise
            else:
                self.model = SentenceTransformer(
                    "mixedbread-ai/mxbai-embed-large-v1", device=device
                )
                self.backend = "sentence_transformers"

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.squeeze()

    def embed_retrieve(self, text: str) -> np.ndarray:
        return self.embed(
            f"Represent this sentence for searching relevant passages: {text}"
        )
