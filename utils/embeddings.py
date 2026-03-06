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
            model="text-embedding-3-small",
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
    max_words: int = 120,
    disliked_books: list[dict] = None,
) -> str:
    """
    Calls OpenAI chat completions to explain why a book was recommended.
    Returns "Explanation unavailable." on any API failure.
    """
    title = book.get("title", "Unknown")
    authors = book.get("authors", "Unknown")
    categories = book.get("categories", "Unknown")
    description = book.get("description", "")

    liked_titles = ", ".join(f'"{b.get("title", "")}"' for b in (liked_books or [])[:5])
    disliked_titles = ", ".join(f'"{b.get("title", "")}"' for b in (disliked_books or [])[:5])

    if liked_titles and disliked_titles:
        user_context = (
            f"The user liked: {liked_titles}. "
            f"The user disliked: {disliked_titles}. "
            f"Explain why \"{title}\" fits what they like and differs from what they disliked, "
            f"drawing connections only to \"{title}\" itself."
        )
    elif liked_titles:
        user_context = (
            f"The user liked: {liked_titles}. "
            f"Explain why \"{title}\" matches their taste by drawing thematic connections to \"{title}\" itself."
        )
    elif disliked_titles:
        user_context = (
            f"The user disliked: {disliked_titles}. "
            f"Explain why \"{title}\" is a better fit and how it differs from what they disliked."
        )
    else:
        user_context = (
            f"The user has no rating history yet. "
            f"Explain why \"{title}\" matches their vibe based solely on its description and categories."
        )

    prompt = (
        f"You are a book recommendation assistant. "
        f"Explain ONLY why the specific book below was recommended. "
        f"Do NOT mention, suggest, or reference any other book titles.\n\n"
        f"Book being explained: \"{title}\" by {authors}\n"
        f"Categories: {categories}\n"
        f"Description: {description[:300]}\n\n"
        f"The user's vibe/search: \"{vibe_prompt}\"\n\n"
        f"{user_context}\n\n"
        f"Your explanation must be about \"{title}\" only. "
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
