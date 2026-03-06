"""
User profile management for the AI Book Recommender.
Handles persistence, profile creation, and like/dislike operations.
"""
import pickle
from collections import Counter


def load_profiles(filepath: str) -> dict:
    """Loads users.pkl. Returns empty dict if file doesn't exist."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_profiles(profiles: dict, filepath: str) -> None:
    """Persists profiles dict to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(profiles, f)


def get_or_create_profile(profiles: dict, username: str) -> dict:
    """Returns existing profile or creates a new empty one."""
    if username not in profiles:
        profiles[username] = {
            "liked_books": [],
            "disliked_books": [],
            "liked_embeddings": [],
            "disliked_embeddings": [],
        }
    return profiles[username]


def add_liked(profiles: dict, username: str, book: dict, embedding) -> dict:
    """Adds book to liked_books idempotently (matched by title). Returns updated profiles."""
    profile = get_or_create_profile(profiles, username)
    if not any(b.get("title") == book.get("title") for b in profile["liked_books"]):
        profile["liked_books"].append(book)
        profile["liked_embeddings"].append(embedding)
    return profiles


def add_disliked(profiles: dict, username: str, book: dict, embedding) -> dict:
    """Adds book to disliked_books idempotently (matched by title). Returns updated profiles."""
    profile = get_or_create_profile(profiles, username)
    if not any(b.get("title") == book.get("title") for b in profile["disliked_books"]):
        profile["disliked_books"].append(book)
        profile["disliked_embeddings"].append(embedding)
    return profiles


def reset_profile(profiles: dict, username: str) -> dict:
    """Clears all liked/disliked data for the user. Returns updated profiles."""
    profile = get_or_create_profile(profiles, username)
    profile["liked_books"] = []
    profile["disliked_books"] = []
    profile["liked_embeddings"] = []
    profile["disliked_embeddings"] = []
    return profiles


def get_genre_distribution(profile: dict) -> dict:
    """Counts category occurrences across liked_books. Returns dict mapping category -> count."""
    categories = [book.get("categories", "") for book in profile.get("liked_books", [])]
    return dict(Counter(categories))
