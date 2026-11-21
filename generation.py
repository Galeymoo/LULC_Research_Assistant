
"""Evidence-grounded answer generation through OpenRouter."""

import os
from pathlib import Path

import requests
import streamlit as st

from config import (
    LLM_MODEL_NAME,
    OPENROUTER_BASE_URL,
    MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
)


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


class EvidenceAnswerGenerator:
    """Generate research answers using retrieved evidence."""

    def __init__(self):
        self.api_key = get_openrouter_key()
        self.api_url = (
            f"{OPENROUTER_BASE_URL}/chat/completions"
        )
        self.model = LLM_MODEL_NAME

    @staticmethod
    def format_evidence(retrieved_passages):
        """Format retrieved research passages."""

        evidence_blocks = []

        for i, passage in enumerate(
            retrieved_passages, start=1
        ):
            evidence_blocks.append(
                f"[EVIDENCE {i}]\n"
                f"Article ID: {passage.get('article_id', 'Unknown')}\n"
                f"Title: {passage.get('title', 'Unknown')}\n"
                f"Year: {passage.get('year', 'Unknown')}\n"
                f"Pages: {passage.get('display_pages', 'Unknown')}\n"
                f"Passage: {passage.get('text', '')}"
            )

        return "\n\n".join(evidence_blocks)

    def _send_request(self, messages, max_tokens):
        """Send a generation request to OpenRouter."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "DAT401 Evidence-Grounded Research Assistant",
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
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: "
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

        # Handle providers that return content as blocks.
        if isinstance(answer, list):
            answer = "\n".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        if not isinstance(answer, str):
            answer = ""

        return (
            answer.strip(),
            choice.get("finish_reason", "unknown"),
            result.get("model", self.model),
        )

    def generate_answer(
        self,
        question,
        retrieved_passages,
        use_evidence=True,
    ):
        """Generate an answer grounded in retrieved evidence."""

        if not use_evidence:
            raise ValueError(
                "Evidence-grounded generation is required."
            )

        if not retrieved_passages:
            return (
                "Insufficient evidence: no relevant research "
                "passages were retrieved."
            )

        evidence = self.format_evidence(
            retrieved_passages
        )

        system_prompt = """
You are an academic research assistant.

Answer the user's question using ONLY the supplied research evidence.

Rules:
1. Do not invent facts, findings, article IDs, or page numbers.
2. Cite factual claims using the exact article ID and page number.
3. Use this citation format: [A15, p. 23].
4. Only cite article IDs and pages included in the supplied evidence.
5. If the evidence is insufficient, clearly say so.
6. Write approximately 150–250 words when the evidence allows.
7. Be direct, clear, and academically appropriate.
8. Complete your answer. Do not leave sentences unfinished.
"""

        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED RESEARCH EVIDENCE:\n{evidence}\n\n"
            "Answer the question using the evidence above. "
            "Include relevant article and page citations."
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            # First attempt
            answer, finish_reason, selected_model = (
                self._send_request(
                    messages,
                    MAX_NEW_TOKENS,
                )
            )

            if answer:
                return answer

            # Retry if the model did not return final text.
            if finish_reason == "length":

                retry_prompt = (
                    f"Question: {question}\n\n"
                    f"Evidence:\n{evidence}\n\n"
                    "Give a concise answer in approximately "
                    "100–150 words. Use only the evidence. "
                    "Include exact article IDs and page citations. "
                    "Finish the answer completely."
                )

                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an academic research assistant. "
                            "Use only the supplied evidence. "
                            "Do not invent citations or findings."
                        ),
                    },
                    {
                        "role": "user",
                        "content": retry_prompt,
                    },
                ]

                answer, finish_reason, selected_model = (
                    self._send_request(
                        retry_messages,
                        1500,
                    )
                )

                if answer:
                    return answer

            raise RuntimeError(
                "The language model did not return final text "
                "after the retry. "
                f"Finish reason: {finish_reason}. "
                f"Selected model: {selected_model}. "
                "The free model may have reached its output limit "
                "or may be temporarily unavailable. "
                "Please try again."
            )

        except requests.Timeout as exc:
            raise RuntimeError(
                "OpenRouter request timed out. Please try again."
            ) from exc

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not connect to OpenRouter: {exc}"
            ) from exc
