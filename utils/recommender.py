import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["title", "authors", "categories", "description", "embedding"]


def load_books(filepath: str = "data/books_with_embeddings.pkl") -> pd.DataFrame:
    """
    Loads books_with_embeddings.pkl from the given filepath.
    Raises FileNotFoundError with descriptive message if missing.
    Validates required fields and excludes malformed records.
    Returns a pandas DataFrame.
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Book data file not found at '{filepath}'. "
            "Please ensure 'books_with_embeddings.pkl' exists at the specified path."
        )

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    # Validate required fields
    before = len(df)
    mask = df[REQUIRED_FIELDS].notnull().all(axis=1)
    malformed = before - mask.sum()
    if malformed > 0:
        logger.warning(f"Excluded {malformed} malformed record(s) missing required fields.")
    df = df[mask].reset_index(drop=True)

    return df


def cosine_similarity_batch(query: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a 1-D query vector and a 2-D embeddings matrix.
    Returns a 1-D array of similarity scores in [-1.0, 1.0].
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10
    normalized = embeddings_matrix / norms
    return normalized @ query_norm


def compute_personalized_query(
    prompt_embedding: np.ndarray,
    liked_embeddings: list,
    disliked_embeddings: list,
    alpha: float = 0.3,
    beta: float = 0.2,
) -> np.ndarray:
    """
    Computes: prompt + alpha * mean(liked) - beta * mean(disliked), normalized to unit length.
    Falls back to normalized prompt if both lists are empty.
    """
    query = np.array(prompt_embedding, dtype=float)

    if liked_embeddings:
        query = query + alpha * np.mean(liked_embeddings, axis=0)

    if disliked_embeddings:
        query = query - beta * np.mean(disliked_embeddings, axis=0)

    norm = np.linalg.norm(query)
    if norm < 1e-10:
        return query
    return query / norm


def get_recommendations(
    query_embedding: np.ndarray,
    books_df: pd.DataFrame,
    profile: dict,
    top_n: int = 10,
    alpha: float = 0.3,
    beta: float = 0.2,
) -> list:
    """
    Orchestrates personalized query + cosine similarity + top-N ranking.
    Returns list of dicts with book metadata + similarity_score, sorted descending.
    """
    liked_embeddings = profile.get("liked_embeddings", [])
    disliked_embeddings = profile.get("disliked_embeddings", [])

    personalized = compute_personalized_query(
        query_embedding, liked_embeddings, disliked_embeddings, alpha, beta
    )

    embeddings_matrix = np.array(books_df["embedding"].tolist())
    scores = cosine_similarity_batch(personalized, embeddings_matrix)

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        row = books_df.iloc[idx]
        results.append({
            "title": row["title"],
            "authors": row["authors"],
            "categories": row["categories"],
            "description": row["description"],
            "similarity_score": float(scores[idx]),
        })

    return results
