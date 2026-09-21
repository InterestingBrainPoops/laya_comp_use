"""Order-independent candidate retrieval: sentence-embedding similarity between the goal
and each element label. Replaces Laya-over-chunks for the first cut.

Why: chunked Laya passes keep an element only if it is top-k inside whichever 20 elements
it happened to be grouped with, so recall depended on shuffle order (measured: the same
case passed or failed across seeds). Embedding similarity scores every element against the
goal independently, in ~10-60ms for 100 labels, and catches synonyms Laya's chunk ranking
lost ("get rid of this file" -> Delete, "make it louder" -> Volume).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

import numpy as np

from laya_agent.config import Config
from laya_agent.models import UIElement

EmbedFn = Callable[[list[str]], np.ndarray]  # texts -> (n, d) unit-normalised rows


class Retriever(Protocol):
    def rank(self, goal: str, elements: list[UIElement]) -> list[tuple[UIElement, float]]:
        """Elements best-first with cosine similarity to the goal."""
        ...


class EmbeddingRetriever:
    def __init__(self, cfg: Config, embed: EmbedFn | None = None) -> None:
        self.cfg = cfg
        self._embed = embed
        self._lock = threading.Lock()
        self.load_ms: float | None = None

    def rank(self, goal: str, elements: list[UIElement]) -> list[tuple[UIElement, float]]:
        if not elements:
            return []
        vecs = self.embed([goal] + [e.label() for e in elements])
        sims = vecs[1:] @ vecs[0]
        order = np.argsort(-sims, kind="stable")
        return [(elements[i], float(sims[i])) for i in order]

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._embed is None:
            self.preload()
        return self._embed(texts)

    def preload(self, log: Callable[[str], None] = lambda s: None) -> None:
        with self._lock:
            if self._embed is not None:
                return
            t0 = time.perf_counter()
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(self.cfg.retriever_model, device=device)
            model.encode(["warmup"] * 8, normalize_embeddings=True)

            def embed(texts: list[str]) -> np.ndarray:
                return model.encode(texts, normalize_embeddings=True, batch_size=128, convert_to_numpy=True)

            self._embed = embed
            self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
            log(f"retriever {self.cfg.retriever_model.split('/')[-1]} ready on {device} in {self.load_ms / 1000:.1f}s")

    @property
    def ready(self) -> bool:
        return self._embed is not None


def make_retriever(cfg: Config) -> Retriever | None:
    if cfg.shortlist == "retriever":
        return EmbeddingRetriever(cfg)
    return None
