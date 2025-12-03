import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Seaborn styling
sns.set_style("whitegrid")
sns.set_palette("husl")

# Aspects to check in reviews
ASPECTS = [
    "Battery", "Display", "Price", "Features", "Audio", "Camera",
    "Quality", "Design", "Comfort", "Speed", "Storage", "Durability",
    "Size", "Weight", "User Experience"
]

# Sentiment color mapping
SENTIMENT_COLORS = {
    'negative': '#FF6B6B',
    'neutral': '#FFD166',
    'positive': '#06D6A0'
}

class ProductReviewAnalyzer:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.model = checkpoint['model']
        self.model.eval()
        
        self.id2label = checkpoint['id2label']        
        self.tokenizer = checkpoint['tokenizer']

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
        return self.id2label[pred_id], probs[0].tolist()

    def analyze_review(self, review_text):
        results = []
        for aspect in ASPECTS:
            sentiment, confidence = self.predict_sentiment(review_text, aspect)
            results.append({
                "review": review_text,
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": max(confidence)#highest class probability
            })
        return results

    def analyze_reviews(self, reviews):
        all_rows = []
        for r in reviews:
            all_rows.extend(self.analyze_review(r))
        return pd.DataFrame(all_rows)


def visualize_sentiment_distribution(df, output_folder):    
    # 1. Bar chart showing total positive vs negative mentions
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='sentiment', order=['negative', 'positive'], palette=SENTIMENT_COLORS)
    plt.title('Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sentiment_distribution.png'), dpi=300)
    plt.show()
    
    # 2. Heatmap showing sentiment counts for each aspect
    plt.figure(figsize=(14, 8))
    pivot_df = df.pivot_table(index='aspect', columns='sentiment', aggfunc='size', fill_value=0) #aggfunc='size' counts occurrences
    pivot_df = pivot_df.reindex(ASPECTS)  #keep aspect order
    pivot_df = pivot_df[['negative', 'positive']]
    
    sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
    plt.title('Sentiment Heatmap by Aspect', fontsize=14)
    plt.xlabel('Sentiment')
    plt.ylabel('Aspect')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sentiment_heatmap.png'), dpi=300)
    plt.show()


def calculate_sentiment_scores(df, output_folder):    
    # 1. Sentiment scores per aspect
    aspect_scores = []
    for aspect in ASPECTS:
        aspect_data = df[df['aspect'] == aspect]
        total = len(aspect_data)
        if total > 0:
            pos_count=(aspect_data['sentiment'] == 'positive').sum()
            neg_count=(aspect_data['sentiment'] == 'negative').sum()
            
            # Net sentiment score (-1 to 1)
            #Ex. 8 positive, 2 negative, 10 total --> (8-2)/10 = 0.6 (mostly positive)
            net_score=(pos_count - neg_count)/total if total>0 else 0
            
            # Positive ratio
            #Ex. 8 positive, 2 negative, 10 total --> 8/10 = 0.8
            pos_ratio =pos_count / total if total > 0 else 0
            
            aspect_scores.append({
                'aspect': aspect,
                'total_mentions': total,
                'positive_count': pos_count,
                'negative_count': neg_count,
                'neutral_count': (aspect_data['sentiment'] =='neutral').sum(),
                'net_sentiment_score': net_score,
                'positive_ratio': pos_ratio,
                'confidence_avg': aspect_data['confidence'].mean()
            })
    
    scores_df = pd.DataFrame(aspect_scores)
    
    # 2. Create net sentiment horizontal bar chart
    plt.figure(figsize=(12, 6))
    scores_sorted = scores_df.sort_values('net_sentiment_score')
    
    colors = []
    for score in scores_sorted['net_sentiment_score']:
        if score > 0:
            colors.append(SENTIMENT_COLORS['positive']) #green
        elif score < 0:
            colors.append(SENTIMENT_COLORS['negative']) #red
        else:
            colors.append(SENTIMENT_COLORS['neutral']) #yellow
    
    bars = plt.barh(scores_sorted['aspect'], scores_sorted['net_sentiment_score'], 
                    color=colors, edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Net Sentiment Score (-1 to 1)')
    plt.title('Net Sentiment Score by Aspect')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'net_sentiment_scores.png'), dpi=300)
    plt.show()
    
    return scores_df


def create_summary_report(df, scores_df, output_folder):
    report = []
    report.append("=" * 60)
    report.append("PRODUCT REVIEW ANALYSIS SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal Reviews Analyzed: {df['review'].nunique()}")
    
    # Overall sentiment
    overall_sentiment = df['sentiment'].value_counts(normalize=True)
    report.append(f"\nOverall Sentiment Distribution:")
    for sentiment, percentage in overall_sentiment.items():
        report.append(f"  {sentiment.capitalize()}: {percentage:.1%}")
    
    #create aspect frequency table
    freq = df.groupby(["aspect", "sentiment"]).size().reset_index(name="count")
    aspect_sentiment_table = freq.pivot(index="aspect", columns="sentiment", values="count").fillna(0)
    aspect_sentiment_table.columns = [col.lower() for col in aspect_sentiment_table.columns]
    aspect_sentiment_table["total"] = aspect_sentiment_table["negative"] + aspect_sentiment_table["neutral"] + aspect_sentiment_table["positive"]
    
    # Reorder aspects based on ASPECTS list
    aspect_sentiment_table = aspect_sentiment_table.reindex(ASPECTS)
    
    #get top strengths and weaknesses
    strengths = aspect_sentiment_table.sort_values(by="positive", ascending=False).head(3)
    weaknesses = aspect_sentiment_table.sort_values(by="negative", ascending=False).head(3)
    
    report.append(f"\n=== ASPECT FREQUENCY TABLE ===")
    report.append('')
    report.append(f"{'aspect':20} {'negative':<10} {'positive':<10}")
    report.append('-' * 40)
    for aspect in ASPECTS:
        if aspect in aspect_sentiment_table.index:
            row = aspect_sentiment_table.loc[aspect]
            report.append(f"{aspect:<20} {int(row.get('negative', 0)):<10}" f"{int(row.get('positive', 0)):<10}")
    
    report.append("\n=== TOP STRENGTHS (Most Positive Mentions) ===")
    report.append('')
    report.append(f"{'aspect':20} {'positive':<10} {'total':<10}")
    report.append('-' * 40)
    for _, (aspect, row) in enumerate(strengths.iterrows(), 1):
        report.append(f"{aspect:<20} {int(row.get('positive', 0)):<10} {int(row.get('total', 0)):<10}")
    
    report.append("\n=== TOP WEAKNESSES (Most Negative Mentions) ===")
    report.append('')
    report.append(f"{'aspect':20} {'negative':<10} {'total':<10}")
    report.append('-' * 40)
    for _, (aspect, row) in enumerate(weaknesses.iterrows(), 1):
        report.append(f"{aspect:<20} {int(row.get('negative', 0)):<10} {int(row.get('total', 0)):<10}")
        
    #save report
    report_text = "\n".join(report)
    with open(os.path.join(output_folder, 'analysis_report.txt'), 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text, aspect_sentiment_table, strengths, weaknesses

def main():
    parser = argparse.ArgumentParser(description="Run analyzer on product reviews")
    parser.add_argument("--model_dir", type=str, default="kfold_models/best_model")
    parser.add_argument("--input_txt", type=str, required=True)
    args = parser.parse_args()
    
    # Create results folder if it doesn't exist
    output_folder = "analyzer_results"
    os.makedirs(output_folder, exist_ok=True)

    # Load reviews from txt file
    with open(args.input_txt, "r", encoding="utf-8") as f:
        reviews = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(reviews)} reviews.\n")

    analyzer = ProductReviewAnalyzer(args.model_dir)
    df = analyzer.analyze_reviews(reviews)

    print("\nSample predictions:\n", df.head(), "\n")

    # Save the prediction table
    predictions_file = os.path.join(output_folder, "predictions.csv")
    df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_sentiment_distribution(df, output_folder)
    
    # Calculate sentiment scores
    scores_df = calculate_sentiment_scores(df, output_folder)
    scores_df.to_csv(os.path.join(output_folder, 'sentiment_scores.csv'), index=False)
        
    # Create summary report
    report, aspect_sentiment_table, strengths, weaknesses = create_summary_report(df, scores_df, output_folder)

    # Also save the aspect_sentiment_table for reference
    aspect_sentiment_table.to_csv(os.path.join(output_folder, 'aspect_frequency_table.csv'))

    
    print(f"\nAll analysis completed!")
    print(f"Results saved to: {output_folder}")
    print(f"Files created:")
    print(f"  - predictions.csv (raw predictions)")
    print(f"  - sentiment_scores.csv (aspect-level metrics)")
    print(f"  - 3 visualization PNG files")
    print(f"  - analysis_report.txt (summary report)")

if __name__ == "__main__":
    main()