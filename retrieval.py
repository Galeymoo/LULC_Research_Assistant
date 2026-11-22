
"""BM25 + BGE dense retrieval + RRF fusion + Cross-Encoder reranking."""

from __future__ import annotations

import logging
import pickle
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


# ==================================================
# LOGGING
# ==================================================

logger = logging.getLogger(__name__)


# ==================================================
# RETRIEVAL SETTINGS
# ==================================================

# Keep None until a threshold has been calibrated
# using labelled relevant and irrelevant questions.
#
# Cross-Encoder scores are not probabilities.
# Do not choose an arbitrary threshold.

RERANK_SCORE_THRESHOLD: float | None = None

# Keep at most one passage per article in final results.
MAX_PASSAGES_PER_ARTICLE = 1


class ResearchRetriever:
    """Retrieve and rerank academic research passages."""

    def __init__(self) -> None:

        # ------------------------------------------
        # CHECK INDEX FILES
        # ------------------------------------------

        for path in (
            BM25_FILE,
            FAISS_FILE,
            CHUNKS_PICKLE_FILE,
        ):

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing index artifact: {path}. "
                    "Run `python indexing.py` first."
                )

        # ------------------------------------------
        # LOAD BM25 INDEX
        # ------------------------------------------

        with BM25_FILE.open("rb") as handle:
            self.bm25 = pickle.load(handle)

        # ------------------------------------------
        # LOAD CHUNKS
        # ------------------------------------------

        with CHUNKS_PICKLE_FILE.open("rb") as handle:

            self.chunks: list[dict[str, Any]] = (
                pickle.load(handle)
            )

        # ------------------------------------------
        # LOAD FAISS INDEX
        # ------------------------------------------

        self.faiss_index = faiss.read_index(
            str(FAISS_FILE)
        )

        if self.faiss_index.ntotal != len(self.chunks):

            raise ValueError(
                "FAISS vector count does not match "
                "the chunk metadata count. "
                "Rebuild the indexes using python indexing.py."
            )

        # ------------------------------------------
        # SELECT DEVICE
        # ------------------------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ------------------------------------------
        # LOAD EMBEDDING MODEL
        # ------------------------------------------

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=self.device,
        )

        # ------------------------------------------
        # LOAD CROSS-ENCODER
        # ------------------------------------------

        self.cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME,
            device=self.device,
        )

        # ------------------------------------------
        # CHUNK LOOKUP
        # ------------------------------------------

        self.chunk_id_to_record = {
            str(item["chunk_id"]): item
            for item in self.chunks
        }

        logger.info(
            "Research retriever loaded | "
            "chunks=%s | device=%s",
            len(self.chunks),
            self.device,
        )

    # ==================================================
    # PAGE FORMATTING
    # ==================================================

    @staticmethod
    def _pages_text(record: dict[str, Any]) -> str:

        pages = record.get("pages")

        if isinstance(pages, list) and pages:

            return ", ".join(
                str(int(page))
                if isinstance(page, (int, float))
                else str(page)
                for page in pages
            )

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

    # ==================================================
    # ARTICLE IDENTIFICATION
    # ==================================================

    @staticmethod
    def _article_key(
        passage: dict[str, Any]
    ) -> str:
        """
        Return an article identifier for deduplication.

        Prefer article_id. If unavailable, use chunk_id
        so unrelated chunks are not grouped together.
        """

        article_id = str(
            passage.get("article_id", "")
        ).strip().upper()

        if article_id:
            return article_id

        chunk_id = passage.get("chunk_id")

        if chunk_id is not None:
            return f"CHUNK_{chunk_id}"

        # Last-resort fallback. This avoids treating every
        # missing-ID passage as the same article.
        return f"UNKNOWN_{id(passage)}"

    # ==================================================
    # FORMAT RETRIEVAL RESULT
    # ==================================================

    def _format_result(
        self,
        record: dict[str, Any],
        score: float,
        method: str,
    ) -> dict[str, Any]:

        result = dict(record)

        result["retrieval_score"] = float(score)
        result["retrieval_method"] = method
        result["display_pages"] = self._pages_text(record)

        return result

    # ==================================================
    # BM25 RETRIEVAL
    # ==================================================

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = BM25_CANDIDATES,
    ) -> list[dict[str, Any]]:

        tokens = tokenize_for_bm25(query)

        scores = np.asarray(
            self.bm25.get_scores(tokens),
            dtype=np.float32,
        )

        if scores.size == 0:
            return []

        indices = np.argsort(scores)[::-1][
            :min(top_k, len(scores))
        ]

        results = []

        for index in indices:

            score = float(scores[int(index)])

            if not np.isfinite(score):
                continue

            results.append(
                self._format_result(
                    self.chunks[int(index)],
                    score,
                    "BM25",
                )
            )

        return results

    # ==================================================
    # DENSE RETRIEVAL
    # ==================================================

    def retrieve_dense(
        self,
        query: str,
        top_k: int = DENSE_CANDIDATES,
    ) -> list[dict[str, Any]]:

        query_text = QUERY_PREFIX + query

        vector = self.embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(
            vector,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            if not np.isfinite(score):
                continue

            results.append(
                self._format_result(
                    self.chunks[int(index)],
                    float(score),
                    "SBERT",
                )
            )

        return results

    # ==================================================
    # HYBRID RETRIEVAL WITH RRF
    # ==================================================

    def retrieve_hybrid(
        self,
        query: str,
        candidate_k: int = HYBRID_CANDIDATES_FOR_RERANKING,
    ) -> list[dict[str, Any]]:

        bm25_results = self.retrieve_bm25(
            query,
            top_k=BM25_CANDIDATES,
        )

        dense_results = self.retrieve_dense(
            query,
            top_k=DENSE_CANDIDATES,
        )

        # ------------------------------------------
        # RECIPROCAL RANK FUSION
        # ------------------------------------------

        rrf_scores: dict[str, float] = {}

        for result_list in (
            bm25_results,
            dense_results,
        ):

            for rank, result in enumerate(
                result_list,
                start=1,
            ):

                chunk_id = str(result["chunk_id"])

                rrf_scores[chunk_id] = (
                    rrf_scores.get(chunk_id, 0.0)
                    + 1.0 / (RRF_K + rank)
                )

        # ------------------------------------------
        # SORT FUSED RESULTS
        # ------------------------------------------

        ordered = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        fused = []

        for chunk_id, score in ordered[:candidate_k]:

            record = self.chunk_id_to_record.get(
                chunk_id
            )

            if record is None:

                logger.warning(
                    "Chunk ID missing from lookup: %s",
                    chunk_id,
                )

                continue

            fused.append(
                self._format_result(
                    record,
                    score,
                    "RRF",
                )
            )

        return fused

    # ==================================================
    # REMOVE DUPLICATE ARTICLES
    # ==================================================

    @staticmethod
    def _unique_articles(
        passages: list[dict[str, Any]],
        max_per_article: int = MAX_PASSAGES_PER_ARTICLE,
    ) -> list[dict[str, Any]]:
        """
        Retain the highest-ranked passage per article.

        Input must already be sorted by Cross-Encoder score.
        """

        results = []
        article_counts: dict[str, int] = {}

        for passage in passages:

            article_key = ResearchRetriever._article_key(
                passage
            )

            current_count = article_counts.get(
                article_key,
                0,
            )

            if current_count >= max_per_article:
                continue

            results.append(passage)

            article_counts[article_key] = (
                current_count + 1
            )

        return results

    # ==================================================
    # CROSS-ENCODER SCORING
    # ==================================================

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and sort hybrid candidates."""

        if not candidates:
            return []

        pairs = [
            (
                query,
                str(item.get("text", "")),
            )
            for item in candidates
        ]

        scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if len(scores) != len(candidates):

            raise ValueError(
                "Cross-Encoder returned an unexpected "
                "number of scores."
            )

        reranked = []

        for item, score in zip(candidates, scores):

            if not np.isfinite(score):
                continue

            result = dict(item)

            result["rerank_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked

    # ==================================================
    # RETRIEVAL SCORE DIAGNOSTICS
    # ==================================================

    def inspect_reranking_scores(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Inspect Cross-Encoder scores for a question.

        Use this to compare relevant and irrelevant queries
        before selecting a relevance threshold.
        """

        candidates = self.retrieve_hybrid(
            question,
            candidate_k=HYBRID_CANDIDATES_FOR_RERANKING,
        )

        if not candidates:

            print("No retrieval candidates found.")

            return []

        reranked = self._rerank_candidates(
            question,
            candidates,
        )

        results = []

        for item in reranked[:top_k]:

            result = {
                "article_id": item.get(
                    "article_id",
                    "Unknown",
                ),
                "title": item.get(
                    "title",
                    "Unknown",
                ),
                "rerank_score": item["rerank_score"],
                "retrieval_score": item.get(
                    "retrieval_score"
                ),
                "text": str(
                    item.get("text", "")
                )[:350],
            }

            results.append(result)

            print("\n" + "=" * 65)
            print("Article:", result["article_id"])
            print("Title:", result["title"])
            print(
                "Cross-Encoder score:",
                round(result["rerank_score"], 4),
            )
            print(
                "RRF score:",
                round(result["retrieval_score"], 6),
            )
            print("Passage:", result["text"])

        if reranked:

            all_scores = [
                item["rerank_score"]
                for item in reranked
            ]

            print("\n" + "=" * 65)
            print("RETRIEVAL SCORE SUMMARY")
            print("Candidates:", len(reranked))
            print("Highest score:", round(max(all_scores), 4))
            print("Lowest score:", round(min(all_scores), 4))
            print("Mean score:", round(float(np.mean(all_scores)), 4))

        return results

    # ==================================================
    # HYBRID RETRIEVAL + CROSS-ENCODER RERANKING
    # ==================================================

    def retrieve_hybrid_reranked(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Retrieve hybrid candidates, rerank them,
        remove duplicate articles, and return top results.
        """

        if not query or not query.strip():
            return []

        # ------------------------------------------
        # GET HYBRID CANDIDATES
        # ------------------------------------------

        candidates = self.retrieve_hybrid(
            query,
            candidate_k=HYBRID_CANDIDATES_FOR_RERANKING,
        )

        if not candidates:
            return []

        # ------------------------------------------
        # CROSS-ENCODER RERANKING
        # ------------------------------------------

        reranked = self._rerank_candidates(
            query,
            candidates,
        )

        # ------------------------------------------
        # OPTIONAL RELEVANCE THRESHOLD
        # ------------------------------------------

        if RERANK_SCORE_THRESHOLD is not None:

            reranked = [
                item
                for item in reranked
                if item["rerank_score"]
                >= RERANK_SCORE_THRESHOLD
            ]

        # ------------------------------------------
        # REMOVE DUPLICATE ARTICLES
        # ------------------------------------------

        unique_results = self._unique_articles(
            reranked,
        )

        # ------------------------------------------
        # LOG RETRIEVAL SUMMARY
        # ------------------------------------------

        logger.info(
            "Retrieval completed | "
            "question_length=%s | "
            "hybrid_candidates=%s | "
            "reranked_candidates=%s | "
            "unique_articles=%s | "
            "returned=%s",
            len(query),
            len(candidates),
            len(reranked),
            len(unique_results),
            min(top_k, len(unique_results)),
        )

        # ------------------------------------------
        # RETURN FINAL RESULTS
        # ------------------------------------------

        return unique_results[:top_k]
