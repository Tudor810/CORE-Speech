"""Async local-LLM wrapper for sign-token -> natural sentence translation.

We talk to a local `Ollama <https://ollama.com>`_ server over HTTP
(default ``http://localhost:11434``) using its streaming
``/api/chat`` endpoint. Tokens are yielded as soon as they arrive so
the downstream TTS can start synthesizing before the full sentence is
generated, just like the previous Gemini client.

GPU usage is automatic: Ollama detects CUDA / Metal / ROCm at launch
and offloads as many model layers as the device can hold. The
``OLLAMA_NUM_GPU`` env var (mapped to ``options.num_gpu`` here) lets
you override that — ``-1`` (the default) means "use as many GPU layers
as fit", ``0`` forces CPU-only.

If Ollama is not reachable, the configured model isn't pulled, or the
HTTP request fails, we fall back to a tiny rule-based stub so the rest
of the pipeline still works end-to-end.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable

import httpx

from app.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You convert sign-language gloss tokens into a natural, human-sounding English sentence.\n"
    "\n"
    "ABSOLUTE RULES (no exceptions):\n"
    "1. Every content word in the output MUST come from the input tokens. Keep them in the same order whenever possible.\n"
    "2. You are FORBIDDEN from introducing any new content words. That means NO new nouns, verbs, adjectives, adverbs, "
    "numbers, names, places, or any word that adds meaning, detail, emphasis, or context that wasn't in the tokens.\n"
    "3. The ONLY words you may add are closed-class connector / function words, strictly from this list:\n"
    "   - Articles: a, an, the\n"
    "   - Pronouns: I, you, he, she, it, we, they, me, him, her, us, them, my, your, his, our, their, this, that, these, those\n"
    "   - Prepositions: to, of, in, on, at, for, with, by, from, about, into, over, under\n"
    "   - Conjunctions: and, or, but, so, if, because, that\n"
    "   - Auxiliaries / copulas: is, am, are, was, were, be, been, being, do, does, did, have, has, had, will, would, can, could, should, may, might\n"
    "   - Negation: not, n't\n"
    "4. You MAY inflect the given words for tense, number, or agreement (e.g. 'go' -> 'went', 'child' -> 'children'), "
    "but you MUST NOT replace them with synonyms or related words.\n"
    "5. Do NOT add adjectives, adverbs, intensifiers, interjections, greetings, fillers, or descriptive phrases — even if "
    "the tone preset suggests excitement, sadness, etc. The tone preset only affects punctuation (e.g. '!' vs '.'), NOT word choice.\n"
    "6. Output ONLY the final sentence. No quotes, no preamble, no explanations, no emojis.\n"
    "\n"
    "Examples:\n"
    "Sign tokens: i want water\n"
    "Sentence: I want the water.\n"
    "\n"
    "Sign tokens: you go store\n"
    "Sentence: You go to the store.\n"
    "\n"
    "Sign tokens: mother cook food\n"
    "Sentence: The mother is cooking the food.\n"
    "\n"
    "Sign tokens: help me please\n"
    "Sentence: Help me, please.\n"
)

def _build_user_prompt(tokens: Iterable[str], emotion: str, intensity: float) -> str:
    glosses = " ".join(tokens)
    return (
        f"Sign tokens: {glosses}\n"
        f"Tone preset: {emotion} (intensity={intensity:.2f})\n"
        f"Sentence:"
    )

async def _fallback_stream(tokens: Iterable[str]) -> AsyncIterator[str]:
    """Offline stub: just join tokens into a basic sentence."""
    text = " ".join(tokens).strip()
    if text:
        yield text.upper() + text[1:] + "."
    else:
        yield ""

def _build_options() -> dict:
    """Build the Ollama ``options`` block from settings.

    ``num_gpu`` is only forwarded when explicitly configured (>= 0); the
    default ``-1`` lets Ollama auto-pick the largest GPU offload that
    fits, which is what we want for "use the GPU if it exists".

    For the gloss->sentence task we cap the temperature so the model
    stops inventing extra content words. The configured temperature is
    still used as the upper bound, but never exceeds 0.2.
    """
    options: dict = {
        "temperature": min(settings.ollama_temperature, 0.2),
        "top_p": 0.8,
        "repeat_penalty": 1.1,
        "num_predict": settings.ollama_max_tokens,
    }
    if settings.ollama_num_gpu >= 0:
        options["num_gpu"] = settings.ollama_num_gpu
    return options

async def translate_signs(
    tokens: Iterable[str],
    emotion: str = "neutral",
    intensity: float = 0.5,
) -> AsyncIterator[str]:
    """Stream natural-language tokens from the local LLM (Ollama)."""
    user_prompt = _build_user_prompt(tokens, emotion, intensity)
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": True,
        "keep_alive": settings.ollama_keep_alive,
        "options": _build_options(),
    }
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

    timeout = httpx.Timeout(settings.ollama_request_timeout, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    log.error(
                        "Ollama HTTP %s: %s (model=%s)",
                        response.status_code, body[:300], settings.ollama_model,
                    )
                    async for chunk in _fallback_stream(tokens):
                        yield chunk
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if err := event.get("error"):
                        log.error("Ollama stream error: %s", err)
                        async for chunk in _fallback_stream(tokens):
                            yield chunk
                        return
                    text = (event.get("message") or {}).get("content")
                    if text:
                        yield text
                    if event.get("done"):
                        break
    except (httpx.HTTPError, OSError) as exc:
        log.error("Local LLM (Ollama @ %s) request failed: %s", url, exc)
        async for chunk in _fallback_stream(tokens):
            yield chunk