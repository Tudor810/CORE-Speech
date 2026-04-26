"""Lazy stand-in for ``melo.text.cleaner``.

Upstream MeloTTS imports every language module in ``melo.text.cleaner`` at
import-time, which drags in optional stacks (MeCab for JP, jamo for KR, etc.)
even when you're only using English.

We register this module into ``sys.modules['melo.text.cleaner']`` before
importing ``melo.api`` so only the requested language module is imported.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

from melo.text import cleaned_text_to_sequence

_LANG_TO_MODULE = {
    "ZH": "chinese",
    "JP": "japanese",
    "EN": "english",
    "ZH_MIX_EN": "chinese_mix",
    "KR": "korean",
    "FR": "french",
    "SP": "spanish",
    "ES": "spanish",
}

_cache: dict[str, Any] = {}


def _get_module(language: str):
    mod_name = _LANG_TO_MODULE.get(language)
    if not mod_name:
        raise KeyError(f"Unknown Melo language code {language!r}")
    if language not in _cache:
        _cache[language] = importlib.import_module(f"melo.text.{mod_name}")
    return _cache[language]


def clean_text(text, language):
    language_module = _get_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = _get_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)

