"""BM25 + BGE dense retrieval + RRF fusion + Cross-Encoder reranking."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    BM25_CANDIDATES,
    BM25_FILE,
    CHUNKS_PICKLE_FILE,
    CROSS_ENCODER_MODEL_NAME,
    DENSE_CANDIDATES,
    EMBEDDING_MODEL_NAME,
    FAISS_FILE,
    FINAL_TOP_K,
    HYBRID_CANDIDATES_FOR_RERANKING,
    QUERY_PREFIX,
    RRF_K,
)
from indexing import tokenize_for_bm25


class ResearchRetriever:
    def __init__(self) -> None:
        for path in (BM25_FILE, FAISS_FILE, CHUNKS_PICKLE_FILE):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing index artifact: {path}. Run `python indexing.py` first."
                )

        with BM25_FILE.open("rb") as handle:
            self.bm25 = pickle.load(handle)
        with CHUNKS_PICKLE_FILE.open("rb") as handle:
            self.chunks: list[dict[str, Any]] = pickle.load(handle)
        self.faiss_index = faiss.read_index(str(FAISS_FILE))

        if self.faiss_index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS vector count does not match the chunk metadata count. "
                "Rebuild the indexes using python indexing.py."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, device=self.device
        )
        self.cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME, device=self.device
        )

        self.chunk_id_to_record = {
            str(item["chunk_id"]): item for item in self.chunks
        }

    @staticmethod
    def _pages_text(record: dict[str, Any]) -> str:
        pages = record.get("pages")
        if isinstance(pages, list) and pages:
            return ", ".join(str(int(p)) if isinstance(p, (int, float)) else str(p) for p in pages)
        start = record.get("start_page")
        end = record.get("end_page")
        if start is not None and end is not None:
            try:
                if int(start) == int(end):
                    return str(int(start))
                return f"{int(start)}–{int(end)}"
            except (TypeError, ValueError):
                return f"{start}–{end}"
        return "Not recorded"

    def _format_result(self, record: dict[str, Any], score: float, method: str) -> dict[str, Any]:
        result = dict(record)
        result["retrieval_score"] = float(score)
        result["retrieval_method"] = method
        result["display_pages"] = self._pages_text(record)
        return result

    def retrieve_bm25(self, query: str, top_k: int = BM25_CANDIDATES) -> list[dict[str, Any]]:
        tokens = tokenize_for_bm25(query)
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        if scores.size == 0:
            return []
        indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        return [
            self._format_result(self.chunks[int(i)], float(scores[int(i)]), "BM25")
            for i in indices
            if np.isfinite(scores[int(i)])
        ]

    def retrieve_dense(self, query: str, top_k: int = DENSE_CANDIDATES) -> list[dict[str, Any]]:
        query_text = QUERY_PREFIX + query
        vector = self.embedding_model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.faiss_index.search(
            vector, min(top_k, len(self.chunks))
        )
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                self._format_result(self.chunks[int(idx)], float(score), "SBERT")
            )
        return results

    def retrieve_hybrid(self, query: str, candidate_k: int = HYBRID_CANDIDATES_FOR_RERANKING) -> list[dict[str, Any]]:
        bm25_results = self.retrieve_bm25(query, top_k=BM25_CANDIDATES)
        dense_results = self.retrieve_dense(query, top_k=DENSE_CANDIDATES)

        # Reciprocal Rank Fusion; ranks are one-based.
        rrf_scores: dict[str, float] = {}
        for result_list in (bm25_results, dense_results):
            for rank, result in enumerate(result_list, start=1):
                chunk_id = str(result["chunk_id"])
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        ordered = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        fused: list[dict[str, Any]] = []
        for chunk_id, score in ordered[:candidate_k]:
            record = self.chunk_id_to_record[chunk_id]
            fused.append(self._format_result(record, score, "RRF"))
        return fused

    def retrieve_hybrid_reranked(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict[str, Any]]:
        candidates = self.retrieve_hybrid(
            query, candidate_k=HYBRID_CANDIDATES_FOR_RERANKING
        )
        if not candidates:
            return []

        pairs = [(query, str(item.get("text", ""))) for item in candidates]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        reranked = sorted(
            candidates, key=lambda item: item["rerank_score"], reverse=True
        )
        return reranked[:top_k]
