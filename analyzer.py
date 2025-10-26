import os
import logging
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import json
import hashlib
import streamlit as st # <-- NEW: Import Streamlit

# Simple file-based cache for fetch_posts
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL = 600  # seconds (10 minutes)


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _make_cache_key(query: str, limit: int) -> str:
    h = hashlib.sha1(f"{query}|{limit}".encode("utf-8")).hexdigest()
    return h


def get_cached_posts(query: str, limit: int):
    key = _make_cache_key(query, limit)
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - payload.get("ts", 0) > CACHE_TTL:
            return None
        return payload.get("posts", [])
    except Exception:
        return None


def set_cached_posts(query: str, limit: int, posts):
    key = _make_cache_key(query, limit)
    path = _cache_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "posts": posts}, f, ensure_ascii=False)
    except Exception:
        pass

# Try to import credentials from config.py if present; allow environment variable overrides.
try:
    from config import CLIENT_ID as CONFIG_CLIENT_ID, CLIENT_SECRET as CONFIG_CLIENT_SECRET, USER_AGENT as CONFIG_USER_AGENT
except Exception:
    CONFIG_CLIENT_ID = CONFIG_CLIENT_SECRET = CONFIG_USER_AGENT = None

logging.basicConfig(level=logging.INFO)


@st.cache_resource # <-- NEW: Use caching to prevent re-initializing PRAW client on every run
def create_reddit_client():
    """Create and return a PRAW Reddit instance.

    Credential precedence:
    1. Streamlit secrets (for deployed app)
    2. Environment variables REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    3. config.py values (if present)
    """
    # ----------------------------------------------------------------------
    # NEW CODE BLOCK: Retrieve credentials from st.secrets first (for deployed app)
    if "reddit" in st.secrets:
        try:
            client_id = st.secrets["reddit"]["CLIENT_ID"]
            client_secret = st.secrets["reddit"]["CLIENT_SECRET"]
            user_agent = st.secrets["reddit"]["USER_AGENT"]
        except KeyError:
             # Fall through to the next check if the keys are named differently
             client_id = client_secret = user_agent = None
    else:
        client_id = client_secret = user_agent = None
    # ----------------------------------------------------------------------


    # FALLBACK/LOCAL DEV: Use existing logic (Environment variables or config.py)
    # The existing code now acts as a fallback to the st.secrets logic above
    client_id = client_id or os.environ.get("REDDIT_CLIENT_ID") or CONFIG_CLIENT_ID
    client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET") or CONFIG_CLIENT_SECRET
    user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT") or CONFIG_USER_AGENT or "CrowdSense v0.1"


    if not client_id or not client_secret:
        logging.error("Reddit credentials not found. Set environment variables or update config.py.")
        raise RuntimeError("Missing Reddit credentials") # <--- Your original error line!

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    return reddit


def fetch_posts(subreddit_name, limit=1000):
    """Fetch the hot posts from a subreddit and return a list of dicts with title, score, num_comments.

    On error, returns an empty list and logs the exception.
    """
    # Try cache first
    cached = get_cached_posts(subreddit_name, limit)
    if cached is not None:
        logging.info("Using cached posts for query '%s' (limit=%s)", subreddit_name, limit)
        return cached

    try:
        reddit = create_reddit_client()
        posts = []

        # First, try to treat the input as a subreddit name
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=limit):
                posts.append({
                    "id": getattr(post, 'id', None),
                    "title": getattr(post, 'title', ""),
                    "selftext": getattr(post, 'selftext', ""),
                    "score": getattr(post, 'score', 0),
                    "num_comments": getattr(post, 'num_comments', 0),
                    "created_utc": getattr(post, 'created_utc', None),
                    "author": getattr(getattr(post, 'author', None), 'name', None),
                    "subreddit": getattr(getattr(post, 'subreddit', None), 'display_name', None),
                    "url": getattr(post, 'url', None),
                })

        except Exception:
            # If subreddit access fails (e.g., invalid name), we'll fall back to a search below
            posts = []

        # If we found no posts from the subreddit attempt, treat the input as a search query on r/all
        if not posts:
            logging.info("No subreddit results for '%s' — falling back to search on r/all", subreddit_name)
            try:
                for post in reddit.subreddit('all').search(subreddit_name, limit=limit, sort='relevance'):
                    posts.append({
                        "id": getattr(post, 'id', None),
                        "title": getattr(post, 'title', ""),
                        "selftext": getattr(post, 'selftext', ""),
                        "score": getattr(post, 'score', 0),
                        "num_comments": getattr(post, 'num_comments', 0),
                        "created_utc": getattr(post, 'created_utc', None),
                        "author": getattr(getattr(post, 'author', None), 'name', None),
                        "subreddit": getattr(getattr(post, 'subreddit', None), 'display_name', None),
                        "url": getattr(post, 'url', None),
                    })
            except Exception as e:
                logging.exception("Search fallback failed: %s", e)

        # Save to cache
        try:
            set_cached_posts(subreddit_name, limit, posts)
        except Exception:
            pass
        return posts
    except Exception as e:
        logging.exception("Failed to fetch posts from Reddit: %s", e)
        return []


def analyze_sentiment(text, analyzer=None):
    """Return 'Positive'/'Negative'/'Neutral' based on VADER compound score."""
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"
    vs = analyzer.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    if compound > 0.05:
        return "Positive"
    if compound < -0.05:
        return "Negative"
    return "Neutral"


def posts_to_dataframe(posts):
    """Convert list of post dicts to a Pandas DataFrame and add sentiment column."""
    analyzer = SentimentIntensityAnalyzer()
    df = pd.DataFrame(posts)
    if df.empty:
        return df
    # Compute compound score and sentiment label using VADER on title + selftext
    def _compound(text):
        vs = analyzer.polarity_scores(text)
        return vs.get("compound", 0.0)

    combined = df["title"].fillna("") + " \n" + df.get("selftext", "").fillna("")
    df["compound"] = combined.apply(_compound)

    def _label(c):
        if c > 0.05:
            return "Positive"
        if c < -0.05:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["compound"].apply(_label)

    # Convert created_utc to datetime if available
    if "created_utc" in df.columns:
        try:
            df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        except Exception:
            df["created_dt"] = None
    return df


def export_posts_to_csv(posts, path):
    """Write posts (list of dicts) to CSV at path."""
    df = pd.DataFrame(posts)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    # Quick test: fetch from r.python and print DataFrame
    posts = fetch_posts("python", limit=1000)
    df = posts_to_dataframe(posts)
    print(df)
