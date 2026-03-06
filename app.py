"""
AI Book Recommender - Streamlit App
"""
import os
import streamlit as st

from utils.recommender import load_books, get_recommendations
from utils.user_profile import (
    load_profiles,
    save_profiles,
    get_or_create_profile,
    add_liked,
    add_disliked,
    reset_profile,
    get_genre_distribution,
)
from utils.embeddings import embed_text, generate_explanation

PROFILES_PATH = "users.pkl"
BOOKS_PATH = "data/books_with_embeddings.pkl"

st.set_page_config(page_title="AI Book Recommender", page_icon="📚", layout="wide")


# ── Environment check ────────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    st.error(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it before running the app."
    )
    st.stop()


# ── Load books (cached in session_state) ─────────────────────────────────────
if "books_df" not in st.session_state:
    try:
        st.session_state["books_df"] = load_books(BOOKS_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

books_df = st.session_state["books_df"]


# ── Load / init profiles ──────────────────────────────────────────────────────
if "profiles" not in st.session_state:
    st.session_state["profiles"] = load_profiles(PROFILES_PATH)


def persist_profiles():
    save_profiles(st.session_state["profiles"], PROFILES_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Book Recommender")
    st.divider()

    username = st.text_input("Username", placeholder="Enter your username…", key="username_input")

    if username:
        profile = get_or_create_profile(st.session_state["profiles"], username)

        # Reset Profile button
        if st.button("🔄 Reset Profile", use_container_width=True):
            st.session_state["profiles"] = reset_profile(st.session_state["profiles"], username)
            persist_profiles()
            st.success("Profile reset.")
            st.rerun()

        st.divider()

        # Liked books history
        liked = profile.get("liked_books", [])
        st.subheader(f"👍 Liked ({len(liked)})")
        if liked:
            for b in liked:
                st.caption(f"• {b.get('title', 'Unknown')}")
        else:
            st.caption("No liked books yet.")

        # Disliked books history
        disliked = profile.get("disliked_books", [])
        st.subheader(f"👎 Disliked ({len(disliked)})")
        if disliked:
            for b in disliked:
                st.caption(f"• {b.get('title', 'Unknown')}")
        else:
            st.caption("No disliked books yet.")
    else:
        st.info("Enter a username to track your preferences.")


# ── Predefined vibe categories (Requirement 8.2) ─────────────────────────────
EXPLORE_CATEGORIES = [
    "Dark Fantasy",
    "Cozy Mystery",
    "Space Opera",
    "Historical Romance",
    "Hard Science Fiction",
    "Literary Fiction",
    "Thriller",
    "Self-Help",
]


# ── Helper: render recommendation cards ──────────────────────────────────────
def render_recommendation_cards(results, username, profile, vibe, card_key_prefix="home"):
    """Render a list of recommendation cards with Like/Dislike and explanation."""
    liked_titles = {b.get("title") for b in profile.get("liked_books", [])}
    disliked_titles = {b.get("title") for b in profile.get("disliked_books", [])}

    for i, book in enumerate(results):
        title = book.get("title", "Unknown")
        authors = book.get("authors", "Unknown")
        categories = book.get("categories", "Unknown")
        description = book.get("description", "")
        score = book.get("similarity_score", 0.0)

        desc_display = description[:200] + "…" if len(description) > 200 else description

        is_liked = title in liked_titles
        is_disliked = title in disliked_titles

        if is_liked:
            card_icon = "🟢"
            state_label = " ✅ Liked"
        elif is_disliked:
            card_icon = "🔴"
            state_label = " ❌ Disliked"
        else:
            card_icon = "📘"
            state_label = ""

        with st.container(border=True):
            col_info, col_actions = st.columns([4, 1])

            with col_info:
                st.markdown(f"**{card_icon} {title}**{state_label}")
                st.caption(f"✍️ {authors}  |  🏷️ {categories}  |  🎯 Score: {score:.3f}")
                st.write(desc_display)

            with col_actions:
                like_label = "👍 Liked" if is_liked else "👍 Like"
                if st.button(like_label, key=f"{card_key_prefix}_like_{i}_{title}", disabled=is_liked):
                    match = books_df[books_df["title"] == title]
                    embedding = match.iloc[0]["embedding"] if not match.empty else []
                    st.session_state["profiles"] = add_liked(
                        st.session_state["profiles"], username, book, embedding
                    )
                    persist_profiles()
                    st.rerun()

                dislike_label = "👎 Disliked" if is_disliked else "👎 Dislike"
                if st.button(dislike_label, key=f"{card_key_prefix}_dislike_{i}_{title}", disabled=is_disliked):
                    match = books_df[books_df["title"] == title]
                    embedding = match.iloc[0]["embedding"] if not match.empty else []
                    st.session_state["profiles"] = add_disliked(
                        st.session_state["profiles"], username, book, embedding
                    )
                    persist_profiles()
                    st.rerun()

        with st.expander("💡 Why this book?"):
            explanation_key = f"{card_key_prefix}_explanation_{i}_{title}"
            if explanation_key not in st.session_state:
                if st.button("Generate explanation", key=f"{card_key_prefix}_explain_btn_{i}_{title}"):
                    with st.spinner("Generating explanation…"):
                        liked_books = profile.get("liked_books", [])
                        explanation = generate_explanation(vibe, book, liked_books, api_key)
                        st.session_state[explanation_key] = explanation
                    st.rerun()
            else:
                st.write(st.session_state[explanation_key])


# ── Page tabs ─────────────────────────────────────────────────────────────────
tab_home, tab_explore, tab_profile = st.tabs(["🔍 Home", "🧭 Explore", "👤 My Profile"])


# ── Home tab ──────────────────────────────────────────────────────────────────
with tab_home:
    st.title("🔍 Find Your Next Book")
    st.write("Describe the kind of book you're in the mood for and we'll find the best matches.")

    vibe_prompt = st.text_area(
        "What's your vibe?",
        placeholder="e.g. dark fantasy with political intrigue and morally grey characters…",
        height=100,
        key="home_vibe_prompt",
    )

    no_username = not username
    if no_username:
        st.caption("⚠️ Enter a username in the sidebar to enable recommendations.")

    get_recs_clicked = st.button(
        "Get Recommendations",
        disabled=no_username,
        type="primary",
        key="home_get_recs",
    )

    if get_recs_clicked:
        if not vibe_prompt or not vibe_prompt.strip():
            st.warning("Please enter a vibe prompt before searching.")
        else:
            with st.spinner("Finding books for you…"):
                try:
                    query_embedding = embed_text(vibe_prompt, api_key)
                    profile = get_or_create_profile(st.session_state["profiles"], username)
                    results = get_recommendations(query_embedding, books_df, profile)
                    st.session_state["last_results"] = results
                    st.session_state["last_vibe"] = vibe_prompt
                except ValueError as e:
                    st.warning(str(e))
                except RuntimeError as e:
                    st.error(str(e))

    results = st.session_state.get("last_results", [])
    if results and username:
        profile = get_or_create_profile(st.session_state["profiles"], username)
        st.divider()
        st.subheader("📖 Recommendations")
        render_recommendation_cards(
            results, username, profile,
            vibe=st.session_state.get("last_vibe", ""),
            card_key_prefix="home",
        )


# ── Explore tab ───────────────────────────────────────────────────────────────
with tab_explore:
    st.title("🧭 Explore by Vibe")
    st.write("Pick a vibe category to discover books — personalization applied if you have a profile.")

    # Render category buttons in a grid (4 per row)
    cols_per_row = 4
    rows = [EXPLORE_CATEGORIES[i:i + cols_per_row] for i in range(0, len(EXPLORE_CATEGORIES), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, category in zip(cols, row):
            with col:
                if st.button(category, key=f"explore_cat_{category}", use_container_width=True):
                    st.session_state["explore_selected_category"] = category
                    st.session_state.pop("explore_results", None)

    selected_category = st.session_state.get("explore_selected_category")

    if selected_category:
        # Run recommendations if not already cached for this category
        if "explore_results" not in st.session_state:
            with st.spinner(f"Finding '{selected_category}' books…"):
                try:
                    query_embedding = embed_text(selected_category, api_key)
                    profile = get_or_create_profile(st.session_state["profiles"], username) if username else {"liked_books": [], "disliked_books": [], "liked_embeddings": [], "disliked_embeddings": []}
                    results = get_recommendations(query_embedding, books_df, profile)
                    st.session_state["explore_results"] = results
                    st.session_state["explore_profile_snapshot"] = profile
                except RuntimeError as e:
                    st.error(str(e))

        explore_results = st.session_state.get("explore_results", [])
        if explore_results:
            st.divider()
            st.subheader(f"📖 {selected_category} — Top Picks")
            profile = st.session_state.get("explore_profile_snapshot", {})
            if username:
                profile = get_or_create_profile(st.session_state["profiles"], username)
            render_recommendation_cards(
                explore_results, username, profile,
                vibe=selected_category,
                card_key_prefix=f"explore_{selected_category.replace(' ', '_')}",
            )


# ── My Profile tab ────────────────────────────────────────────────────────────
with tab_profile:
    st.title("👤 My Profile")

    if not username:
        st.info("Enter a username in the sidebar to view your profile.")
    else:
        profile = get_or_create_profile(st.session_state["profiles"], username)
        liked_books = profile.get("liked_books", [])
        disliked_books = profile.get("disliked_books", [])

        # ── Genre distribution chart (Requirement 9.2) ────────────────────────
        st.subheader("📊 Genre Distribution")
        genre_dist = get_genre_distribution(profile)
        if genre_dist:
            st.bar_chart(genre_dist)
        else:
            st.caption("No genre data yet — like some books to see your taste profile.")

        st.divider()

        # ── Liked books list (Requirement 9.3) ────────────────────────────────
        st.subheader(f"👍 Liked Books ({len(liked_books)})")
        if liked_books:
            for book in liked_books:
                st.write(f"**{book.get('title', 'Unknown')}** — {book.get('authors', 'Unknown')}")
        else:
            st.info("You haven't liked any books yet. Start rating books on the Home or Explore page.")

        st.divider()

        # ── Disliked books list (Requirement 9.3) ─────────────────────────────
        st.subheader(f"👎 Disliked Books ({len(disliked_books)})")
        if disliked_books:
            for book in disliked_books:
                st.write(f"**{book.get('title', 'Unknown')}** — {book.get('authors', 'Unknown')}")
        else:
            st.info("You haven't disliked any books yet.")
