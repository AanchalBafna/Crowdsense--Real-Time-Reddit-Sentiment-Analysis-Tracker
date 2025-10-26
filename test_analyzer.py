from analyzer import analyze_sentiment


def test_analyze_sentiment_positive():
    text = "I love this project! It's amazing and helpful."
    assert analyze_sentiment(text) == "Positive"


def test_analyze_sentiment_negative():
    text = "This is awful. I hate it."
    assert analyze_sentiment(text) == "Negative"


def test_analyze_sentiment_neutral_empty():
    assert analyze_sentiment("") == "Neutral"
