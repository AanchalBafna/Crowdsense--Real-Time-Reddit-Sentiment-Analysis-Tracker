import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from wordcloud import STOPWORDS
from io import BytesIO
from analyzer import fetch_posts, posts_to_dataframe
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
import warnings


st.set_page_config(page_title="CrowdSense — Real-Time Reddit Sentiment Tracker", layout="wide")
st.markdown("# CrowdSense — Real-Time Reddit Sentiment Tracker")

# --- Custom “Image Palette” Styling ---
# Adjusted for Ivory White background and new color scheme
st.markdown(f"""
<style>
/* Overall layout - Set to IVORY WHITE: #FAF8F5 */
.block-container {{
    max-width: 1400px;
    padding: 3.5rem 2rem 1rem 2rem; /* extra top padding so header isn't overlapped */
    background-color: #FAF8F5; /* IVORY WHITE */
    color: #4C4C4C;
    font-family: 'Inter', sans-serif;
}}

/* Headings - Use the deep red for contrast */
h1, h2, h3 {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #9C4C4C; /* Deep Red/Maroon */
    letter-spacing: 0.5px;
}}
/* Ensure main title is fully visible and prominent */
h1 {{
    font-size: 34px;
    line-height: 1.1;
    margin-top: 0;
    margin-bottom: 0.35rem;
}}

/* KPI cards */
.kpi {{
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #EAD7CE;
    background: #FFF;
    text-align: center;
    color: #4C4C4C;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
}}
/* KPI values */
.kpi .value {{
    font-size: 22px;
    display: block;
    margin-top: 6px;
    font-weight: 700;
    color: #9C4C4C; /* Deep Red/Maroon */
}}

/* Buttons - Use the Muted Rose/Coral as Primary */
.stButton>button {{
    background-color: #D99499 !important; /* Muted Rose/Coral */
    color: #FFF !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}}

/* Sidebar - Use the Soft Pink/Beige for a light background */
.stSidebar {{
    background-color: #F9E9E6; /* Lightest Pink/Beige */
    color: #4C4C4C;
}}
/* Sidebar Headings */
.stSidebar h3 {{
    color: #9C4C4C; /* Deep Red/Maroon */
}}

/* Multiselect selected tags (pill) styling - maroon background */
.stMultiSelect div[role="listbox"] .css-12w0qpk {{ background: #8B2F2F; color: #fff; border-radius:6px; padding:2px 6px; }}
.stMultiSelect .css-14xtw13 {{ background: #8B2F2F; color: #fff; border-radius:6px; }}

/* Slider and input range styling - maroon accent */
input[type="range"]::-webkit-slider-runnable-track {{ background: #EAD7CE; height:6px; border-radius:6px; }}
input[type="range"]::-webkit-slider-thumb {{ background:#8B2F2F; border-radius:50%; width:16px; height:16px; margin-top:-5px; }}
input[type="range"]::-moz-range-track {{ background: #EAD7CE; height:6px; border-radius:6px; }}
input[type="range"]::-moz-range-progress {{ background:#8B2F2F; height:6px; border-radius:6px; }}

/* Chart container cards - Keep white for clean look */
.chart-card {{
    background: #FFF;
    border-radius: 12px;
    border: 1px solid #EAD7CE;
    padding: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}}

/* Chart alignment */
.stPlotlyChart {{
    padding: 8px;
}}

/* Wordcloud centering */
.wc-center {{
    display: flex;
    justify-content: center;
}}

/* Table styling */
table {{
    border: 1px solid #EAD7CE;
    border-radius: 6px;
}}
</style>
""", unsafe_allow_html=True)

# --- NEW Image-Based Color Palette ---
PALETTE = {
    'primary': '#8B2F2F',    # deep maroon (main accent)
    'accent': '#C96F6F',     # softer maroon
    'positive': '#2D3138',   # ivory (light)
    'neutral': '#FAF8F5',    # pale pink / beige
    'negative': '#E8C8CD',   # dark slate (for contrast)
}

SENTIMENT_COLOR_MAP = {
    'Positive': PALETTE['positive'],
    'Neutral': PALETTE['neutral'],
    'Negative': PALETTE['negative']
}
DEFAULT_DISCRETE = [PALETTE['primary'], PALETTE['accent'], PALETTE['positive'], PALETTE['neutral'], PALETTE['negative']]
DEFAULT_CONTINUOUS = 'Sunset'


# Helper to render Plotly charts with a consistent light template and responsive width
def render_plotly(fig, key=None, height=None, config=None):
    try:
        if fig is None:
            return
        # Apply a consistent light template and tidy margins
        try:
            # Using 'plotly_white' template for light background, and set font to dark for contrast
            fig.update_layout(template='plotly_white', margin=dict(t=10, b=10, l=10, r=10), font=dict(color='#4C4C4C')) 
        except Exception:
            pass
        if height is not None:
            try:
                fig.update_layout(height=height)
            except Exception:
                pass
        st.plotly_chart(fig, use_container_width=True, key=key, config=(config or plot_config))
    except Exception as e:
        st.warning(f"Could not render a chart: {e}")

# Input area: subreddit and Analyze button under the intro
sub_col, _ = st.columns([3, 1])
with sub_col:
    subreddit = st.text_input("Subreddit / Topic", help="Enter a subreddit name (e.g., 'learnpython') or a keyword/topic to search for.")
    analyze = st.button("Analyze")

# Posts to analyze: numeric input only (1..1000)
posts_limit = st.sidebar.number_input('Posts to analyze', min_value=1, max_value=1000, value=1000, step=1)

# Prepare unique keys for plotly charts to avoid duplicate auto-generated IDs
if 'run_id' not in st.session_state:
    st.session_state['run_id'] = 0
st.session_state['run_id'] += 1
plot_config = {"displayModeBar": False}


def make_pie_chart_plotly(counts):
    labels = counts.index.tolist()
    values = counts.values.tolist()
    # Use the first four colors from DEFAULT_DISCRETE for a pleasant palette
    palette_for_pie = DEFAULT_DISCRETE[:4]
    # Map labels to palette positions; fallback to SENTIMENT_COLOR_MAP where available
    colors = []
    for i, l in enumerate(labels):
        if l in SENTIMENT_COLOR_MAP:
            colors.append(SENTIMENT_COLOR_MAP[l])
        else:
            colors.append(palette_for_pie[i % len(palette_for_pie)])
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5,
                                 marker=dict(colors=colors))])
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
    return fig


def sentiment_histogram_plotly(df):
    if 'compound' not in df.columns:
        return None
    fig = px.histogram(df, x='compound', nbins=50, color='sentiment', color_discrete_map=SENTIMENT_COLOR_MAP)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), xaxis_title='VADER compound score')
    return fig


def correlation_heatmap_plotly(df):
    # include title_len if present
    if 'title' in df.columns and 'title_len' not in df.columns:
        try:
            df['title_len'] = df['title'].astype(str).apply(len)
        except Exception:
            pass
    cols = [c for c in ['score', 'num_comments', 'compound', 'title_len'] if c in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu', zmin=-1, zmax=1, text_auto=True)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10))
    return fig


def comments_vs_upvotes_scatter(df):
    if 'score' not in df.columns or 'num_comments' not in df.columns:
        return None
    fig = px.scatter(df, x='score', y='num_comments', color='sentiment' if 'sentiment' in df.columns else None,
                     size='num_comments', hover_data=['title'], color_discrete_map=SENTIMENT_COLOR_MAP)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10))
    return fig


def top_authors_bar(df, n=10):
    if 'author' not in df.columns:
        return None
    grp = df.groupby('author').agg(posts=('id','count'), upvotes=('score','sum')).reset_index()
    top = grp.sort_values('posts', ascending=False).head(n)
    fig = px.bar(top, x='posts', y='author', orientation='h', color='upvotes', color_continuous_scale=DEFAULT_CONTINUOUS)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10))
    return fig


def title_length_scatter(df):
    if 'title' not in df.columns:
        return None
    df['title_len'] = df['title'].astype(str).apply(len)
    fig = px.scatter(df, x='title_len', y='score' if 'score' in df.columns else None, size='num_comments' if 'num_comments' in df.columns else None,
                     hover_data=['title'], color='sentiment' if 'sentiment' in df.columns else None,
                     color_discrete_map=SENTIMENT_COLOR_MAP)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), xaxis_title='Title length (chars)')
    return fig


def sentiment_bubble(df):
    # x: compound, y: score, size: num_comments
    if 'compound' not in df.columns:
        return None
    fig = px.scatter(df, x='compound', y='score' if 'score' in df.columns else 0, size='num_comments' if 'num_comments' in df.columns else None,
                     color='sentiment' if 'sentiment' in df.columns else None, hover_data=['title'], color_discrete_map=SENTIMENT_COLOR_MAP)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10))
    return fig


def keyword_grouped_bar(df, keywords):
    if 'title' not in df.columns or 'compound' not in df.columns:
        return None
    data = []
    for kw in keywords:
        mask = df['title'].str.contains(kw, case=False, na=False)
        if mask.sum() == 0:
            avg = None
        else:
            avg = df.loc[mask, 'compound'].mean()
        data.append({'keyword': kw, 'avg_compound': avg if avg is not None else 0, 'count': int(mask.sum())})
    kdf = pd.DataFrame(data)
    fig = px.bar(kdf, x='keyword', y='avg_compound', color='count', color_continuous_scale=DEFAULT_CONTINUOUS)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), yaxis_title='Avg VADER compound')
    return fig


def make_wordcloud_image(text):
    # Use the Ivory White background color
    wc = WordCloud(width=800, height=400, background_color="#FAF8F5", colormap="magma").generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return buf


def plot_avg_score_over_time(df, date_col='created_dt', value_col='score'):
    """Return a Plotly line for daily average of value_col (e.g., score) over date_col."""
    if date_col not in df.columns or value_col not in df.columns:
        return None
    try:
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
        daily = tmp.set_index(date_col).resample('D')[value_col].mean().reset_index()
        if daily.empty:
            return None
        fig = px.line(daily, x=date_col, y=value_col, markers=True, color_discrete_sequence=[PALETTE['primary']])
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10))
        return fig
    except Exception:
        return None


def plot_score_hist_plotly(df):
    fig = px.histogram(df, x="score", nbins=20, color_discrete_sequence=[PALETTE['primary']]) 
    fig.update_layout(bargap=0.1, margin=dict(t=20, b=20, l=20, r=20))
    return fig


def plot_sentiment_over_time_plotly(df):
    if "created_dt" not in df.columns or df["created_dt"].isnull().all():
        return None
    ts = df.set_index("created_dt").resample('D').sentiment.apply(lambda s: (s == 'Positive').sum())
    ts = ts.reset_index()
    ts.columns = ['date', 'positive_count']
    fig = px.line(ts, x='date', y='positive_count', markers=True, color_discrete_sequence=[PALETTE['accent']]) 
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    return fig


def top_words_bar_plotly(df, n=10):
    # prepare NLTK resources
    try:
        nltk.data.find('corpora/wordnet')
    except Exception:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except Exception:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        nltk.download('stopwords')

    lemmatizer = WordNetLemmatizer()
    text = " ".join(df['title'].astype(str).tolist()).lower()
    tokens = [w.strip(".,!?()[]{}\"'`") for w in text.split()]
    sw = set(STOPWORDS) | set(nltk_stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) > 3 and t not in sw]
    c = Counter(tokens)
    items = c.most_common(n)
    if not items:
        return None
    words, counts = zip(*items)
    # Horizontal bar with continuous color to show frequency; use a consistent sequential colormap
    fig = px.bar(x=list(counts), y=list(words), orientation='h', color=list(counts), color_continuous_scale=DEFAULT_CONTINUOUS)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title='Frequency', yaxis={'automargin': True})
    return fig


# Persist fetched dataframe across reruns so sidebar filters don't force the user to re-click Analyze
if 'last_subreddit' not in st.session_state:
    st.session_state['last_subreddit'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None

if st.session_state['last_subreddit'] != subreddit:
    # Clear cached df when the user changes the subreddit input
    st.session_state['df'] = None
    st.session_state['last_subreddit'] = subreddit

# Main fetch / use cached logic
if analyze or st.session_state.get('df') is not None:
    df = None
    # If user clicked Analyze, fetch new posts. If not but a cached df exists, use it.
    if analyze:
        with st.spinner(f"Fetching posts for '{subreddit}'..."):
            posts = fetch_posts(subreddit, limit=posts_limit)
            if not posts:
                st.error("No posts fetched. Check subreddit name or Reddit credentials.")
            else:
                df = posts_to_dataframe(posts)
                st.session_state['df'] = df
    else:
        # use cached dataframe from previous Analyze
        df = st.session_state.get('df')

    if df is not None:
        # --- Interactive filters ---
        st.sidebar.markdown(f"<h3 style='color:{PALETTE['primary']}; margin-bottom:0.25rem'>Filters</h3>", unsafe_allow_html=True)
        max_score = int(df['score'].max()) if 'score' in df.columns and not df['score'].isnull().all() else 100
        min_upvotes, max_upvotes = st.sidebar.slider('Upvotes (score) range', 0, max_score, (0, max_score))
        max_comments = int(df['num_comments'].max()) if 'num_comments' in df.columns and not df['num_comments'].isnull().all() else 50
        min_comments, max_comments = st.sidebar.slider('Comments range', 0, max_comments, (0, max_comments))
        sentiments = ['Positive','Neutral','Negative'] if 'sentiment' in df.columns else []
        selected_sentiments = st.sidebar.multiselect('Sentiments to include', sentiments, default=sentiments)
        keywords_input = st.sidebar.text_input('Keywords (comma separated)', value='')
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]

        # apply filters
        filt = pd.Series([True]*len(df))
        if 'score' in df.columns:
            filt = filt & (df['score'] >= min_upvotes) & (df['score'] <= max_upvotes)
        if 'num_comments' in df.columns:
            filt = filt & (df['num_comments'] >= min_comments) & (df['num_comments'] <= max_comments)
        if selected_sentiments:
            filt = filt & df['sentiment'].isin(selected_sentiments)
        if keywords:
            kw_mask = pd.Series([False]*len(df))
            for kw in keywords:
                kw_mask = kw_mask | df['title'].str.contains(kw, case=False, na=False)
            filt = filt & kw_mask

        filtered = df[filt].copy()
        st.sidebar.markdown(f"Filtered posts: **{len(filtered)}**")

        # Top metrics (KPI boxes with colored backgrounds)
        st.markdown("## Summary")
        total = len(df)
        pos = int((df['sentiment'] == 'Positive').sum())
        neu = int((df['sentiment'] == 'Neutral').sum())
        neg = int((df['sentiment'] == 'Negative').sum())
        k1, k2, k3, k4 = st.columns(4)
        kpi_style = "padding: 12px; border-radius: 8px; color: white; font-weight: 600; text-align:center;"
        # Adjusted KPI colors for new palette
        k1.markdown(f"<div style='background:{PALETTE['primary']}; {kpi_style}'>Total posts<br><span class='value'>{total}</span></div>", unsafe_allow_html=True)
        # Using accent color for positive KPI with dark text for contrast, as positive is a very light color
        k2.markdown(f"<div style='background:{PALETTE['accent']}; color:#4C4C4C; {kpi_style}'>Positive<br><span class='value'>{pos}</span></div>", unsafe_allow_html=True)
        # Make the Neutral tile use the ivory/light background and dark text for visibility
        k3.markdown(f"<div style='background:{PALETTE['positive']}; color:#2B2F3A; {kpi_style}'>Neutral<br><span class='value'>{neu}</span></div>", unsafe_allow_html=True)
        k4.markdown(f"<div style='background:{PALETTE['negative']}; {kpi_style}'>Negative<br><span class='value'>{neg}</span></div>", unsafe_allow_html=True)

        # Layout for charts (uniform sizes)
        # We'll use a single chart_height for consistent alignment across columns
        chart_height = 340

        # Always coerce created_dt to datetime if present
        if 'created_dt' in df.columns:
            try:
                df['created_dt'] = pd.to_datetime(df['created_dt'], errors='coerce')
            except Exception:
                pass

        # Choose the source DataFrame for plotting (filtered if available)
        source = filtered if 'filtered' in locals() else df

        # Wrap plotting in try/except so table still shows if a chart errors
        try:
            # Three-column dashboard layout for better presentation
            c1, c2, c3 = st.columns([1.2, 1, 1])

            # Column 1: Sentiment, Score distribution, Top words
            with c1:
                st.subheader("Sentiment Breakdown")
                if 'sentiment' in source.columns and not source['sentiment'].dropna().empty:
                    counts = source["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
                    pie_fig = make_pie_chart_plotly(counts)
                    render_plotly(pie_fig, key=f"pie_{st.session_state['run_id']}", height=chart_height)
                else:
                    st.info("No sentiment data available for selected filters.")

                st.subheader("Score distribution")
                hist_fig = plot_score_hist_plotly(source)
                render_plotly(hist_fig, key=f"hist_{st.session_state['run_id']}", height=chart_height)

                st.subheader("Top words in titles")
                bar_fig = top_words_bar_plotly(source, n=10)
                render_plotly(bar_fig, key=f"bar_{st.session_state['run_id']}", height=chart_height)

            # Column 2: Wordcloud, Avg sentiment over time, Title length
            with c2:
                st.subheader("Word Cloud")
                combined = " ".join(source["title"].astype(str).tolist())
                if combined.strip():
                    wc_buf = make_wordcloud_image(combined)
                    st.image(wc_buf, width=380, caption='Common words in titles', use_container_width=True)
                else:
                    st.info("No titles available for word cloud.")

                st.subheader("Average sentiment (Positive count) over time")
                compound_fig = plot_sentiment_over_time_plotly(source)
                render_plotly(compound_fig, key=f"compound_{st.session_state['run_id']}", height=chart_height)

                st.subheader("Title length vs Score")
                tl_fig = title_length_scatter(source)
                render_plotly(tl_fig, key=f"titlelen_{st.session_state['run_id']}", height=chart_height)

            # Column 3: Heatmap, Avg score over time, Comments vs Upvotes
            with c3:
                st.subheader("Feature correlation (heatmap)")
                heat_fig = correlation_heatmap_plotly(source)
                render_plotly(heat_fig, key=f"heat_{st.session_state['run_id']}", height=chart_height)

                st.subheader("Average score over time")
                avg_score_fig = plot_avg_score_over_time(source, value_col='score')
                render_plotly(avg_score_fig, key=f"avgscore_{st.session_state['run_id']}", height=chart_height)

                st.subheader("Comments vs Upvotes")
                cu_fig = comments_vs_upvotes_scatter(source)
                render_plotly(cu_fig, key=f"cu_{st.session_state['run_id']}", height=chart_height)

        except Exception as e:
            st.error("An error occurred while generating charts:")
            st.exception(e)

        st.subheader("Results Table")
        # show main columns — keep this outside the plotting try/except so it always renders
        cols = ['title', 'subreddit', 'author', 'score', 'num_comments', 'sentiment']
        present = [c for c in cols if c in df.columns]
        # Render a colored Plotly table where sentiment column cells have background colors
        try:
            table_df = source[present].fillna('')
            header_vals = list(table_df.columns)
            cell_values = [table_df[c].astype(str).tolist() for c in table_df.columns]
            
            # Apply background color to sentiment column cells
            sentiment_colors = [SENTIMENT_COLOR_MAP.get(s, '#FFFFFF') for s in table_df['sentiment']]
            cell_fills = ['white'] * len(table_df.columns)
            if 'sentiment' in table_df.columns:
                sentiment_idx = table_df.columns.get_loc('sentiment')
                cell_fills[sentiment_idx] = sentiment_colors

            fig_table = go.Figure(data=[go.Table(
                header=dict(values=header_vals, fill_color='#D99499', font=dict(color='white')), # Header uses primary color
                cells=dict(values=cell_values, align='left', fill_color=cell_fills)
            )])
            fig_table.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=chart_height)
            render_plotly(fig_table, key=f"table_{st.session_state['run_id']}", height=chart_height)
        except Exception as e:
            st.warning("Could not render table; showing a sample instead.")
            st.dataframe(source[present].head(100).fillna(''))

        # CSV download (for the filtered view)
        csv = source.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download CSV", data=csv, file_name=f"{subreddit}_posts_filtered.csv", mime="text/csv")
else:
    st.info("Enter a subreddit/topic and click 'Analyze' to fetch posts.")