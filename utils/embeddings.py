from openai import OpenAI


def embed_text(text: str, api_key: str) -> list[float]:
    """
    Calls OpenAI embeddings API (model: text-embedding-ada-002) and returns a vector.
    Raises ValueError if text is empty or whitespace-only.
    Raises RuntimeError if API call fails.
    """
    if not text or not text.strip():
        raise ValueError("Vibe prompt must not be empty or whitespace-only.")

    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"OpenAI embeddings API call failed: {e}") from e


def generate_explanation(
    vibe_prompt: str,
    book: dict,
    liked_books: list[dict],
    api_key: str,
    max_words: int = 100,
) -> str:
    """
    Calls OpenAI chat completions to explain why a book was recommended.
    Returns "Explanation unavailable." on any API failure.
    """
    title = book.get("title", "Unknown")
    authors = book.get("authors", "Unknown")
    categories = book.get("categories", "Unknown")
    description = book.get("description", "")

    if liked_books:
        liked_titles = ", ".join(b.get("title", "") for b in liked_books[:5])
        user_context = (
            f"The user has previously liked these books: {liked_titles}. "
            "Explain the recommendation in the format: "
            "'Recommended because it shares themes with books you liked such as X and Y.'"
        )
    else:
        user_context = (
            "The user has no liked books yet. "
            "Explain the recommendation based solely on the vibe prompt and book metadata."
        )

    prompt = (
        f"You are a book recommendation assistant. "
        f"The user is looking for: \"{vibe_prompt}\".\n\n"
        f"Book: \"{title}\" by {authors}\n"
        f"Categories: {categories}\n"
        f"Description: {description[:300]}\n\n"
        f"{user_context}\n\n"
        f"Limit your explanation to a maximum of {max_words} words."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Explanation unavailable."
