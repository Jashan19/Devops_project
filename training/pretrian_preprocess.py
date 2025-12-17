import pandas as pd
from transformers import pipeline

# Load pretrained sentiment pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Load your clean journal data
df = pd.read_csv("clean_journal_data.csv")

# Run predictions
results = classifier(df["text"].tolist())

# Attach predictions
df["sentiment"] = [r["label"] for r in results]
df["confidence"] = [r["score"] for r in results]

# Save output
df.to_csv("journal_sentiment_predictions.csv", index=False)

print("✅ Sentiment predictions saved")
