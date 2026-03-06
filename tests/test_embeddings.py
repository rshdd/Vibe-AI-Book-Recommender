"""
Tests for embeddings.py - covers embed_text validation.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import characters

from utils.embeddings import embed_text


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_embed_text_raises_on_empty_string():
    """embed_text raises ValueError for empty string."""
    with pytest.raises(ValueError):
        embed_text("", api_key="fake")


def test_embed_text_raises_on_whitespace_only():
    """embed_text raises ValueError for whitespace-only strings."""
    for ws in ["   ", "\t", "\n", "  \t\n  "]:
        with pytest.raises(ValueError):
            embed_text(ws, api_key="fake")


# ---------------------------------------------------------------------------
# Property 3: Empty and whitespace-only prompts are rejected
# Feature: ai-book-recommender, Property 3: whitespace prompts rejected
# ---------------------------------------------------------------------------

_WHITESPACE_CHARS = " \t\n\r\x0b\x0c\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"

@settings(max_examples=100)
@given(
    text=st.one_of(
        st.just(""),
        st.text(
            alphabet=st.sampled_from(list(_WHITESPACE_CHARS)),
            min_size=1,
            max_size=50,
        ),
    )
)
def test_property3_whitespace_prompts_rejected(text):
    """
    Property 3: Empty and whitespace-only prompts are rejected.
    Validates: Requirements 2.4
    """
    # Feature: ai-book-recommender, Property 3: whitespace prompts rejected
    with pytest.raises(ValueError):
        embed_text(text, api_key="fake")
