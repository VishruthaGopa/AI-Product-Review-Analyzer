import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hard-coded aspects to check in reviews
ASPECTS = [
    "Battery", "Display", "Price", "Features", "Audio", "Camera",
    "Quality", "Design", "Comfort", "Speed", "Storage", "Durability",
    "Size", "Weight", "User Experience"
]

class ProductReviewAnalyzer:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print("Model loaded successfully!\n")

    def predict_sentiment(self, text, aspect):
        inputs = self.tokenizer(
            text,
            aspect,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
        return self.id2label[pred_id]

    def analyze_review(self, review_text):
        results = []
        text_lower = review_text.lower()
        for aspect in ASPECTS:
            if aspect.lower() in text_lower:
                sentiment = self.predict_sentiment(review_text, aspect)
                results.append({
                    "review": review_text,
                    "aspect": aspect,
                    "sentiment": sentiment
                })
        return results

    def analyze_reviews(self, reviews):
        all_rows = []
        for r in reviews:
            all_rows.extend(self.analyze_review(r))
        return pd.DataFrame(all_rows)


def summarize_results(df):
    if df.empty:
        print("No aspect mentions found — cannot summarize.")
        return None, None, None

    freq = df.groupby(["aspect", "sentiment"]).size().reset_index(name="count")
    pivot = freq.pivot(index="aspect", columns="sentiment", values="count").fillna(0)
    pivot["total"] = pivot.sum(axis=1)

    # Ensure lowercase column names
    pivot.columns = [col.lower() for col in pivot.columns]

    strengths = pivot.sort_values(by="positive", ascending=False).head(5)
    weaknesses = pivot.sort_values(by="negative", ascending=False).head(5)

    print("\n=== ASPECT FREQUENCY TABLE ===\n")
    print(pivot)
    print("\n=== TOP STRENGTHS (Most Positive Mentions) ===\n")
    print(strengths[["positive", "total"]])
    print("\n=== TOP WEAKNESSES (Most Negative Mentions) ===\n")
    print(weaknesses[["negative", "total"]])

    return pivot, strengths, weaknesses


def main():
    parser = argparse.ArgumentParser(description="Run analyzer on product reviews")
    parser.add_argument("--model_dir", type=str, default="kfold_models/best_model")
    parser.add_argument("--input_txt", type=str, required=True)
    args = parser.parse_args()
    
    # Create results folder if it doesn't exist
    output_folder = "analyzer_results"
    os.makedirs(output_folder, exist_ok=True)

    # Load reviews
    with open(args.input_txt, "r", encoding="utf-8") as f:
        reviews = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(reviews)} reviews.\n")

    analyzer = ProductReviewAnalyzer(args.model_dir)
    df = analyzer.analyze_reviews(reviews)

    print("\nSample predictions:\n", df.head(), "\n")

    # Save prediction table
    predictions_file = os.path.join(output_folder, "predictions.csv")
    df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    # Create summary tables
    pivot, strengths, weaknesses = summarize_results(df)

    if pivot is not None:
        pivot.to_csv(os.path.join(output_folder, "aspect_frequencies.csv"))
        strengths.to_csv(os.path.join(output_folder, "top_strengths.csv"))
        weaknesses.to_csv(os.path.join(output_folder, "top_weaknesses.csv"))
        print("\nSummary saved to:")
        print(f" - {os.path.join(output_folder, 'aspect_frequencies.csv')}")
        print(f" - {os.path.join(output_folder, 'top_strengths.csv')}")
        print(f" - {os.path.join(output_folder, 'top_weaknesses.csv')}")

if __name__ == "__main__":
    main()