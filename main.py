from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# 🔓 Allow CORS for local and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to ["https://your-site.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()

# Optional: Patch the lexicon for financial terms
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

@app.post("/sentiment")
async def analyze_sentiment(request: Request):
    body = await request.json()
    text = body.get("text")

    if not text:
        return {"error": "Missing 'text' field."}

    scores = analyzer.polarity_scores(text)

    # Determine sentiment label
    sentiment = (
        "positive" if scores["compound"] >= 0.05
        else "negative" if scores["compound"] <= -0.05
        else "neutral"
    )

    return {
        "sentiment": sentiment,
        "compound": scores["compound"],
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"]
    }