from fastapi import FastAPI, Request
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

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