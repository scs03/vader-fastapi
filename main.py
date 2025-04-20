from fastapi import FastAPI, Request
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()
# Boost financial terms
analyzer.lexicon.update({
    "soared": 3.0,
    "beat": 2.0,
    "beats": 2.0,
    "raised": 1.8,
    "guidance": 1.5,
    "record": 1.5,
    "expectations": 1.2,
    "strong": 2.0,
    "surged": 3.2,
    "revenue": 1.7,
    "plummeted": -3.5,
    "missed": -2.5,
    "disappointing": -2.2,
    "bleak": -2.8
})

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
def get_sentiment(payload: SentimentRequest):
    scores = analyzer.polarity_scores(payload.text)
    sentiment = "Positive" if scores["compound"] > 0.2 else "Negative" if scores["compound"] < -0.2 else "Neutral"
    return {
        "sentiment": sentiment,
        "compound": scores["compound"],
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"]
    }