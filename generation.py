
"""Evidence-grounded answer generation through OpenRouter."""

import logging
import os
import re
from pathlib import Path

import requests
import streamlit as st

from config import (
    LLM_MODEL_NAME,
    OPENROUTER_BASE_URL,
    MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
)


# ==================================================
# LOGGING
# ==================================================

logger = logging.getLogger(__name__)


# ==================================================
# OPENROUTER API KEY
# ==================================================

def get_openrouter_key():
    """Load the OpenRouter API key securely."""

    key = os.getenv("OPENROUTER_API_KEY")

    if not key:
        try:
            key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            key = None

    if not key:
        try:
            import tomllib

            secret_path = (
                Path(__file__).resolve().parent
                / ".streamlit"
                / "secrets.toml"
            )

            if secret_path.exists():
                with secret_path.open("rb") as file:
                    secrets = tomllib.load(file)

                key = secrets.get("OPENROUTER_API_KEY")

        except Exception:
            key = None

    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Configure it in Streamlit secrets."
        )

    return key


# ==================================================
# ANSWER COMPLETENESS CHECK
# ==================================================

def looks_incomplete(answer: str) -> bool:
    """
    Detect common unfinished answers.

    This is a heuristic, not a guarantee that an answer
    is factually complete or correctly supported.
    """

    if not answer or not answer.strip():
        return True

    text = answer.strip()

    # Normal sentence-ending punctuation.
    if text[-1] in ".!?。！？":
        return False

    # Common unfinished punctuation.
    if text.endswith((",", ";", ":", "—", "-")):
        return True

    # Common dangling words.
    dangling_endings = {
        "and",
        "or",
        "but",
        "with",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "from",
        "because",
        "which",
        "that",
        "including",
        "such as",
        "as well as",
    }

    final_words = re.findall(
        r"[A-Za-z]+(?:[-'][A-Za-z]+)?",
        text.lower()
    )

    if final_words:

        if final_words[-1] in dangling_endings:
            return True

        if len(final_words) >= 2:
            last_two = " ".join(final_words[-2:])

            if last_two in dangling_endings:
                return True

    return False


# ==================================================
# ANSWER GENERATOR
# ==================================================

class EvidenceAnswerGenerator:
    """Generate research answers using retrieved evidence."""

    def __init__(self):

        self.api_key = get_openrouter_key()

        self.api_url = (
            f"{OPENROUTER_BASE_URL}/chat/completions"
        )

        self.model = LLM_MODEL_NAME

    # ==================================================
    # FORMAT EVIDENCE
    # ==================================================

    @staticmethod
    def format_evidence(retrieved_passages):
        """Format retrieved research passages."""

        evidence_blocks = []

        for i, passage in enumerate(
            retrieved_passages,
            start=1
        ):

            evidence_blocks.append(
                f"[EVIDENCE {i}]\n"
                f"Article ID: "
                f"{passage.get('article_id', 'Unknown')}\n"
                f"Title: "
                f"{passage.get('title', 'Unknown')}\n"
                f"Year: "
                f"{passage.get('year', 'Unknown')}\n"
                f"Pages: "
                f"{passage.get('display_pages', 'Unknown')}\n"
                f"Passage: "
                f"{passage.get('text', '')}"
            )

        return "\n\n".join(evidence_blocks)

    # ==================================================
    # SEND OPENROUTER REQUEST
    # ==================================================

    def _send_request(self, messages, max_tokens):
        """Send a generation request to OpenRouter."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": (
                "DAT401 Evidence-Grounded Research Assistant"
            ),
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": LLM_TEMPERATURE,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=(20, 180),
        )

        if response.status_code != 200:

            logger.error(
                "OpenRouter HTTP error: status=%s",
                response.status_code
            )

            raise RuntimeError(
                f"OpenRouter API error "
                f"{response.status_code}: "
                f"{response.text[:1000]}"
            )

        result = response.json()

        choices = result.get("choices", [])

        if not choices:
            raise RuntimeError(
                "OpenRouter returned no answer choices."
            )

        choice = choices[0]

        message = choice.get("message", {})

        answer = message.get("content")

        # ------------------------------------------
        # HANDLE TEXT BLOCK RESPONSES
        # ------------------------------------------

        if isinstance(answer, list):

            answer = "\n".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        if not isinstance(answer, str):
            answer = ""

        answer = answer.strip()

        finish_reason = choice.get(
            "finish_reason",
            "unknown"
        )

        selected_model = result.get(
            "model",
            self.model
        )

        usage = result.get("usage", {})

        # ------------------------------------------
        # DIAGNOSTIC LOGGING
        # ------------------------------------------

        logger.info(
            "OpenRouter generation completed | "
            "model=%s | finish_reason=%s | "
            "max_tokens=%s | answer_characters=%s | "
            "usage=%s",
            selected_model,
            finish_reason,
            max_tokens,
            len(answer),
            usage,
        )

        return (
            answer,
            finish_reason,
            selected_model,
            usage,
        )

    # ==================================================
    # BUILD PROMPTS
    # ==================================================

    def _build_messages(
        self,
        question,
        evidence,
        concise=False
    ):
        """Build evidence-grounded system and user prompts."""

        system_prompt = """
You are an academic research assistant specialising in
remote sensing, satellite imagery, and Land Use and
Land Cover (LULC) research.

Your responsibility is to provide accurate,
evidence-grounded academic answers using ONLY the
research evidence supplied by the retrieval system.

STRICT EVIDENCE RULES:

1. Answer the exact question asked by the user.

2. Use only the supplied research evidence.

3. Never invent facts, findings, statistics, dates,
   article IDs, publication years, or page numbers.

4. Cite factual claims using the exact article ID
   and page number supplied in the evidence.

5. Use this citation format:
   [A15, p. 23]

6. Only cite article IDs and pages that appear in
   the supplied evidence.

7. Never create citations simply to demonstrate
   that retrieved articles were considered.

8. Do not use unrelated papers as supporting evidence.

9. A related topic does not automatically mean that
   the evidence answers the question.

10. Verify that the supplied evidence directly supports
    the specific information requested.

STRICT ABSTENTION RULES:

11. If the question is outside the scope of remote
    sensing, satellite imagery, or LULC research,
    briefly state that the question is outside the
    scope of the research collection.

12. Do not discuss unrelated retrieved papers when
    the question is outside the research scope.

13. If the question is within the research scope but
    the supplied evidence does not answer it,
    clearly state that there is insufficient evidence.

14. Do not substitute older statistics, approximate
    figures, projections, or estimates for a specific
    statistic or year requested by the user.

15. If only partial or related information is available,
    do not present it as a complete answer.

16. Do not assume that a paper's mention of a topic
    proves the specific claim asked by the user.

17. Keep straightforward abstention responses to
    one or two sentences.

ANSWER QUALITY RULES:

18. Write clearly and academically.

19. For questions involving multiple studies,
    compare the findings using the supplied evidence.

20. Explain similarities, differences, advantages,
    limitations, and research context when supported.

21. Ensure that every sentence is complete.

22. Do not end the answer halfway through a sentence.

23. Do not repeat the same finding unnecessarily.

24. Do not make unsupported generalisations.

25. If the evidence is sufficient, provide a useful,
    complete answer with appropriate citations.

26. If the evidence is insufficient, prefer an honest
    abstention over an unsupported answer.

Remember:
Retrieved passages are not automatically relevant.
Use only passages that genuinely support the answer.
"""

        if concise:

            length_instruction = """
Write a concise answer of approximately 100–150 words.

Prioritise the most important findings.

Do not introduce new claims or unsupported information.

Use the exact evidence and citation details provided.

Make sure the final sentence is complete.
"""

        else:

            length_instruction = """
Write approximately 150–250 words when the evidence
allows.

Organise the answer logically.

Use relevant citations throughout the answer.

Avoid unnecessary repetition.

Make sure the final sentence is complete.
"""

        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED RESEARCH EVIDENCE:\n"
            f"{evidence}\n\n"
            f"{length_instruction}\n"
            "Answer using only the supplied research evidence."
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    def generate_answer(
        self,
        question,
        retrieved_passages,
        use_evidence=True,
    ):
        """Generate a complete evidence-grounded answer."""

        if not use_evidence:

            raise ValueError(
                "Evidence-grounded generation is required."
            )

        if not question or not question.strip():

            raise ValueError(
                "Please enter a research question."
            )

        if not retrieved_passages:

            return (
                "Insufficient evidence: no relevant research "
                "passages were retrieved. Please try a question "
                "related to remote sensing, satellite imagery, "
                "or Land Use and Land Cover."
            )

        evidence = self.format_evidence(
            retrieved_passages
        )

        messages = self._build_messages(
            question,
            evidence,
            concise=False
        )

        try:

            # ======================================
            # FIRST GENERATION ATTEMPT
            # ======================================

            (
                answer,
                finish_reason,
                selected_model,
                usage,
            ) = self._send_request(
                messages,
                MAX_NEW_TOKENS
            )

            if not answer:

                logger.warning(
                    "OpenRouter returned an empty answer."
                )

            # ======================================
            # CHECK ANSWER COMPLETENESS
            # ======================================

            needs_retry = (
                finish_reason == "length"
                or looks_incomplete(answer)
                or not answer
            )

            # ======================================
            # SECOND GENERATION ATTEMPT
            # ======================================

            if needs_retry:

                logger.warning(
                    "Incomplete generation detected. "
                    "Attempting one concise regeneration. "
                    "First finish_reason=%s | "
                    "answer_characters=%s | model=%s",
                    finish_reason,
                    len(answer),
                    selected_model,
                )

                retry_messages = self._build_messages(
                    question,
                    evidence,
                    concise=True
                )

                retry_max_tokens = max(
                    1500,
                    MAX_NEW_TOKENS
                )

                (
                    retry_answer,
                    retry_finish_reason,
                    retry_model,
                    retry_usage,
                ) = self._send_request(
                    retry_messages,
                    retry_max_tokens
                )

                # ----------------------------------
                # ACCEPT COMPLETE RETRY
                # ----------------------------------

                if (
                    retry_answer
                    and retry_finish_reason != "length"
                    and not looks_incomplete(retry_answer)
                ):

                    logger.info(
                        "Concise regeneration succeeded. "
                        "Model=%s | finish_reason=%s",
                        retry_model,
                        retry_finish_reason,
                    )

                    return retry_answer

                # ----------------------------------
                # RETRY ALSO INCOMPLETE
                # ----------------------------------

                logger.warning(
                    "Retry may also be incomplete. "
                    "finish_reason=%s | "
                    "answer_characters=%s | "
                    "model=%s | usage=%s",
                    retry_finish_reason,
                    len(retry_answer),
                    retry_model,
                    retry_usage,
                )

                if retry_answer:

                    return (
                        retry_answer
                        + "\n\n"
                        + "**Generation warning:** "
                        + "The language model may have "
                        + "stopped before completing its "
                        + "answer. Please verify the final "
                        + "sentence and citations."
                    )

                raise RuntimeError(
                    "The language model did not return "
                    "a usable answer after the retry. "
                    f"Finish reason: {retry_finish_reason}. "
                    f"Model: {retry_model}. "
                    "Please try again."
                )

            # ======================================
            # NORMAL COMPLETE ANSWER
            # ======================================

            if answer:

                logger.info(
                    "Answer generation completed normally. "
                    "Model=%s | finish_reason=%s",
                    selected_model,
                    finish_reason,
                )

                return answer

            raise RuntimeError(
                "The language model returned an empty answer."
            )

        # ==========================================
        # TIMEOUT HANDLING
        # ==========================================

        except requests.Timeout as exc:

            logger.exception(
                "OpenRouter request timed out."
            )

            raise RuntimeError(
                "OpenRouter request timed out. "
                "Please try again."
            ) from exc

        # ==========================================
        # CONNECTION ERROR HANDLING
        # ==========================================

        except requests.RequestException as exc:

            logger.exception(
                "Could not connect to OpenRouter."
            )

            raise RuntimeError(
                f"Could not connect to OpenRouter: {exc}"
            ) from exc
