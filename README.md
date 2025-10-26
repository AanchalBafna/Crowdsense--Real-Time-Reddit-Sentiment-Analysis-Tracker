# CrowdSense - Reddit sentiment prototype

This workspace contains a simple backend script to fetch Reddit posts and analyze sentiment using VADER.

Setup (Windows):

1. python -m venv venv
2. .\venv\Scripts\activate
3. pip install praw pandas vaderSentiment wordcloud matplotlib streamlit

Run quick test:

python analyzer.py

Notes:
- `config.py` holds your Reddit credentials (CLIENT_ID, CLIENT_SECRET, USER_AGENT). It is intentionally gitignored.
- Day 2: create `app.py` and import functions from `analyzer.py` to build a Streamlit UI.

Day 2 notes (Streamlit):

- Create `app.py` and use `st.text_input()` to accept a subreddit and `st.button()` to trigger analysis.
- Import `fetch_posts` and `posts_to_dataframe` from `analyzer.py` and display with `st.write(df)`.
- For visualizations: use `matplotlib` for pie charts and `wordcloud.WordCloud` to generate an image and display with `st.image()`.

