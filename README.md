# AI Book Recommender 📚

A personalized book recommendation app powered by OpenAI embeddings and semantic search.

## Features

- Search for books by describing your vibe in natural language
- Personalized recommendations based on your liked/disliked history
- Explore curated genre categories
- AI-generated explanations for why each book was recommended
- Per-user profile with genre distribution chart

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the book data

Place `books_with_embeddings.pkl` inside a `data/` folder at the project root.  
This file is not included in the repo due to its size.

### 4. Run the app

```bash
streamlit run app.py
```

```bash
Enjoy the deployed version here: https://vibe-ai-book-recommender-prla.streamlit.app/
```

## Project Structure

```
├── app.py                  # Streamlit app entry point
├── utils/
│   ├── embeddings.py       # OpenAI embedding + explanation calls
│   ├── recommender.py      # Cosine similarity & personalized ranking
│   └── user_profile.py     # Profile persistence (like/dislike)
├── tests/                  # Unit + property-based tests
├── data/                   # Book data (not tracked in git)
└── requirements.txt
```

## Notes

- `users.pkl` (user profiles) is created at runtime and is not tracked in git.
- The notebook `Embedding_Books.ipynb` was used to generate `books_with_embeddings.pkl`.
