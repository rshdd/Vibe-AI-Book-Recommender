"""
Tests for user_profile.py - covers profile management functionality.
"""
import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.user_profile import (
    add_disliked,
    add_liked,
    get_genre_distribution,
    get_or_create_profile,
    load_profiles,
    reset_profile,
    save_profiles,
)

REQUIRED_KEYS = {"liked_books", "disliked_books", "liked_embeddings", "disliked_embeddings"}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_load_profiles_missing_file():
    """load_profiles returns empty dict when file does not exist."""
    result = load_profiles("nonexistent_path_xyz.pkl")
    assert result == {}


def test_get_or_create_profile_new_user():
    """New username returns a profile with all four keys set to empty lists."""
    profiles = {}
    profile = get_or_create_profile(profiles, "alice")
    assert set(profile.keys()) == REQUIRED_KEYS
    for key in REQUIRED_KEYS:
        assert isinstance(profile[key], list)
        assert profile[key] == []


def test_get_or_create_profile_existing_user():
    """Existing username returns the same profile without overwriting."""
    profiles = {}
    p1 = get_or_create_profile(profiles, "bob")
    p1["liked_books"].append({"title": "Dune"})
    p2 = get_or_create_profile(profiles, "bob")
    assert len(p2["liked_books"]) == 1


def test_add_liked_basic():
    """add_liked adds a book and its embedding to the profile."""
    book = {"title": "Dune", "authors": "Herbert", "categories": "Sci-Fi"}
    profiles = add_liked({}, "alice", book, [0.1, 0.2])
    assert len(profiles["alice"]["liked_books"]) == 1
    assert len(profiles["alice"]["liked_embeddings"]) == 1


def test_add_liked_idempotent():
    """Calling add_liked twice with the same book adds it only once."""
    book = {"title": "Dune", "authors": "Herbert", "categories": "Sci-Fi"}
    profiles = add_liked({}, "alice", book, [0.1, 0.2])
    profiles = add_liked(profiles, "alice", book, [0.1, 0.2])
    assert len(profiles["alice"]["liked_books"]) == 1
    assert len(profiles["alice"]["liked_embeddings"]) == 1


def test_add_disliked_idempotent():
    """Calling add_disliked twice with the same book adds it only once."""
    book = {"title": "Foundation", "authors": "Asimov", "categories": "Sci-Fi"}
    profiles = add_disliked({}, "bob", book, [0.3])
    profiles = add_disliked(profiles, "bob", book, [0.3])
    assert len(profiles["bob"]["disliked_books"]) == 1
    assert len(profiles["bob"]["disliked_embeddings"]) == 1


def test_reset_profile_clears_all():
    """reset_profile clears all four lists."""
    book = {"title": "Dune", "categories": "Sci-Fi"}
    profiles = add_liked({}, "carol", book, [0.1])
    profiles = add_disliked(profiles, "carol", {"title": "Other"}, [0.2])
    profiles = reset_profile(profiles, "carol")
    for key in REQUIRED_KEYS:
        assert profiles["carol"][key] == []


def test_save_load_round_trip():
    """Saving and loading profiles produces an equal dict."""
    book = {"title": "Dune", "authors": "Herbert", "categories": "Sci-Fi"}
    profiles = add_liked({}, "dave", book, [0.5])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        path = f.name
    try:
        save_profiles(profiles, path)
        loaded = load_profiles(path)
        assert loaded == profiles
    finally:
        os.unlink(path)


def test_get_genre_distribution_counts():
    """get_genre_distribution returns correct category counts."""
    profile = {
        "liked_books": [
            {"title": "A", "categories": "Sci-Fi"},
            {"title": "B", "categories": "Fantasy"},
            {"title": "C", "categories": "Sci-Fi"},
        ]
    }
    dist = get_genre_distribution(profile)
    assert dist == {"Sci-Fi": 2, "Fantasy": 1}


def test_get_genre_distribution_empty():
    """get_genre_distribution returns empty dict for empty liked_books."""
    profile = {"liked_books": []}
    assert get_genre_distribution(profile) == {}


# ---------------------------------------------------------------------------
# Property 4: User profile round-trip persistence
# Feature: ai-book-recommender, Property 4: profile round-trip persistence
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    username=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    titles=st.lists(
        st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
        min_size=0,
        max_size=5,
        unique=True,
    ),
)
def test_property4_profile_round_trip_persistence(username, titles):
    """
    Property 4: User profile round-trip persistence.
    Validates: Requirements 3.1, 3.4, 3.5
    """
    # Feature: ai-book-recommender, Property 4: profile round-trip persistence
    profiles = {}
    for title in titles:
        book = {"title": title, "categories": "Fiction"}
        profiles = add_liked(profiles, username, book, [0.1])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        path = f.name
    try:
        save_profiles(profiles, path)
        loaded = load_profiles(path)
        assert loaded == profiles
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Property 5: User profile structural invariant
# Feature: ai-book-recommender, Property 5: profile structural invariant
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    username=st.text(min_size=1, max_size=30),
)
def test_property5_profile_structural_invariant(username):
    """
    Property 5: User profile structural invariant.
    Validates: Requirements 3.2, 3.3
    """
    # Feature: ai-book-recommender, Property 5: profile structural invariant
    profiles = {}
    profile = get_or_create_profile(profiles, username)
    assert set(profile.keys()) == REQUIRED_KEYS
    for key in REQUIRED_KEYS:
        assert isinstance(profile[key], list)


# ---------------------------------------------------------------------------
# Property 6: Like and dislike operations are idempotent
# Feature: ai-book-recommender, Property 6: like/dislike idempotence
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    repeat=st.integers(min_value=1, max_value=10),
    title=st.text(min_size=1, max_size=30),
)
def test_property6_like_dislike_idempotence(repeat, title):
    """
    Property 6: Like and dislike operations are idempotent.
    Validates: Requirements 3.7, 3.8
    """
    # Feature: ai-book-recommender, Property 6: like/dislike idempotence
    book = {"title": title, "categories": "Fiction"}
    profiles = {}
    for _ in range(repeat):
        profiles = add_liked(profiles, "user", book, [0.1])
    assert len(profiles["user"]["liked_books"]) == 1

    profiles2 = {}
    for _ in range(repeat):
        profiles2 = add_disliked(profiles2, "user", book, [0.1])
    assert len(profiles2["user"]["disliked_books"]) == 1


# ---------------------------------------------------------------------------
# Property 7: Reset produces an empty profile
# Feature: ai-book-recommender, Property 7: reset produces empty profile
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    titles=st.lists(
        st.text(min_size=1, max_size=20),
        min_size=0,
        max_size=10,
        unique=True,
    ),
)
def test_property7_reset_produces_empty_profile(titles):
    """
    Property 7: Reset produces an empty profile.
    Validates: Requirements 3.6
    """
    # Feature: ai-book-recommender, Property 7: reset produces empty profile
    profiles = {}
    for title in titles:
        book = {"title": title, "categories": "Fiction"}
        profiles = add_liked(profiles, "user", book, [0.1])

    profiles = reset_profile(profiles, "user")
    for key in REQUIRED_KEYS:
        assert profiles["user"][key] == []


# ---------------------------------------------------------------------------
# Property 10: Genre distribution counts are accurate
# Feature: ai-book-recommender, Property 10: genre distribution accuracy
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    books=st.lists(
        st.fixed_dictionaries({
            "title": st.text(min_size=1, max_size=20),
            "categories": st.sampled_from(["Sci-Fi", "Fantasy", "Mystery", "Romance", "Thriller"]),
        }),
        min_size=0,
        max_size=20,
    )
)
def test_property10_genre_distribution_accuracy(books):
    """
    Property 10: Genre distribution counts are accurate.
    Validates: Requirements 9.2
    """
    # Feature: ai-book-recommender, Property 10: genre distribution accuracy
    profile = {"liked_books": books}
    dist = get_genre_distribution(profile)

    # Each count must match manual count
    for category, count in dist.items():
        expected = sum(1 for b in books if b.get("categories") == category)
        assert count == expected

    # Sum of all counts equals total liked books
    assert sum(dist.values()) == len(books)
