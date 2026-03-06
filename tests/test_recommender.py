"""
Tests for recommender.py - covers core retrieval functionality.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.recommender import (
    cosine_similarity_batch,
    compute_personalized_query,
    get_recommendations,
    load_books,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_load_books_missing_file():
    """load_books raises FileNotFoundError with the filepath in the message."""
    missing = "data/nonexistent_file.pkl"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_books(missing)
    assert missing in str(exc_info.value)


def test_cosine_similarity_batch_known_values():
    """Identical vectors should have similarity 1.0."""
    v = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scores = cosine_similarity_batch(v, matrix)
    assert abs(scores[0] - 1.0) < 1e-6
    assert abs(scores[1] - 0.0) < 1e-6


def test_compute_personalized_query_no_history():
    """With empty liked/disliked, result should be normalized prompt."""
    prompt = np.array([3.0, 4.0])
    result = compute_personalized_query(prompt, [], [])
    expected = prompt / np.linalg.norm(prompt)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_compute_personalized_query_unit_length():
    """Output should always have unit L2 norm."""
    prompt = np.array([1.0, 2.0, 3.0])
    liked = [np.array([0.5, 0.5, 0.5])]
    disliked = [np.array([0.1, 0.1, 0.1])]
    result = compute_personalized_query(prompt, liked, disliked)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6


def test_get_recommendations_returns_top_n():
    """get_recommendations should return at most top_n results."""
    dim = 4
    n_books = 20
    embeddings = np.random.rand(n_books, dim).tolist()
    books_df = pd.DataFrame({
        "title": [f"Book {i}" for i in range(n_books)],
        "authors": ["Author"] * n_books,
        "categories": ["Fiction"] * n_books,
        "description": ["desc"] * n_books,
        "embedding": embeddings,
    })
    query = np.random.rand(dim)
    profile = {"liked_embeddings": [], "disliked_embeddings": []}
    results = get_recommendations(query, books_df, profile, top_n=5)
    assert len(results) == 5


def test_get_recommendations_sorted_descending():
    """Results must be sorted by similarity_score descending."""
    dim = 4
    n_books = 15
    embeddings = np.random.rand(n_books, dim).tolist()
    books_df = pd.DataFrame({
        "title": [f"Book {i}" for i in range(n_books)],
        "authors": ["Author"] * n_books,
        "categories": ["Fiction"] * n_books,
        "description": ["desc"] * n_books,
        "embedding": embeddings,
    })
    query = np.random.rand(dim)
    profile = {"liked_embeddings": [], "disliked_embeddings": []}
    results = get_recommendations(query, books_df, profile, top_n=10)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_get_recommendations_includes_similarity_score():
    """Each result dict must include similarity_score."""
    dim = 4
    embeddings = np.random.rand(5, dim).tolist()
    books_df = pd.DataFrame({
        "title": [f"Book {i}" for i in range(5)],
        "authors": ["Author"] * 5,
        "categories": ["Fiction"] * 5,
        "description": ["desc"] * 5,
        "embedding": embeddings,
    })
    query = np.random.rand(dim)
    profile = {"liked_embeddings": [], "disliked_embeddings": []}
    results = get_recommendations(query, books_df, profile, top_n=3)
    for r in results:
        assert "similarity_score" in r


# ---------------------------------------------------------------------------
# Property 2: Cosine similarity scores are bounded and results are ranked
# Feature: ai-book-recommender, Property 2: scores bounded, results ranked
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    n_books=st.integers(min_value=1, max_value=50),
    dim=st.integers(min_value=2, max_value=16),
    top_n=st.integers(min_value=1, max_value=20),
)
def test_property2_scores_bounded_and_ranked(n_books, dim, top_n):
    """
    Property 2: Cosine similarity scores are bounded and results are ranked.
    Validates: Requirements 2.2, 2.3, 2.5, 10.3
    """
    # Feature: ai-book-recommender, Property 2: scores bounded, results ranked
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_books, dim)).tolist()
    books_df = pd.DataFrame({
        "title": [f"Book {i}" for i in range(n_books)],
        "authors": ["Author"] * n_books,
        "categories": ["Fiction"] * n_books,
        "description": ["desc"] * n_books,
        "embedding": embeddings,
    })
    query = rng.standard_normal(dim)
    profile = {"liked_embeddings": [], "disliked_embeddings": []}
    results = get_recommendations(query, books_df, profile, top_n=top_n)

    # Length <= top_n
    assert len(results) <= top_n

    scores = [r["similarity_score"] for r in results]

    # Scores in [-1.0, 1.0]
    for s in scores:
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6

    # Sorted descending
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Property 9: Normalized query has unit length
# Feature: ai-book-recommender, Property 9: normalized query unit length
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    values=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    )
)
def test_property9_normalized_query_unit_length(values):
    """
    Property 9: Normalized query has unit length.
    Validates: Requirements 4.4
    """
    # Feature: ai-book-recommender, Property 9: normalized query unit length
    prompt = np.array(values)
    # Skip zero vectors (undefined normalization)
    if np.linalg.norm(prompt) < 1e-10:
        return
    result = compute_personalized_query(prompt, [], [])
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Unit test for predefined categories list (Task 10.1)
# ---------------------------------------------------------------------------

def test_explore_categories_non_empty_and_sufficient():
    """
    Assert the EXPLORE_CATEGORIES list is non-empty and contains at least 4 entries.
    Requirements: 8.2
    """
    import importlib, sys
    # Import the constant directly from app module source without running Streamlit
    import ast, pathlib

    source = pathlib.Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    categories = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXPLORE_CATEGORIES":
                    categories = ast.literal_eval(node.value)

    assert categories is not None, "EXPLORE_CATEGORIES not found in app.py"
    assert len(categories) >= 4, f"Expected at least 4 categories, got {len(categories)}"
    assert all(isinstance(c, str) and c.strip() for c in categories), "All categories must be non-empty strings"
