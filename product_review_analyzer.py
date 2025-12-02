import argparse
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Hard-coded aspects to check in reviews
ASPECTS = [
    "Battery", "Display", "Price", "Features", "Audio", "Camera",
    "Quality", "Design", "Comfort", "Speed", "Storage", "Durability",
    "Size", "Weight", "User Experience"
]

class ProductReviewAnalyzer:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.model = checkpoint['model']
        self.model.eval()
        
        self.id2label = checkpoint['id2label']
        self.label2id = checkpoint['label2id']
        
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
                "confidence": max(confidence)  # Highest class probability
            })
        return results

    def analyze_reviews(self, reviews):
        all_rows = []
        for r in reviews:
            all_rows.extend(self.analyze_review(r))
        return pd.DataFrame(all_rows)


def visualize_sentiment_distribution(df, output_folder):
    """Create comprehensive visualizations for sentiment analysis"""
    
    # 1. Sentiment Distribution Across All Aspects
    plt.figure(figsize=(12, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'negative': '#FF6B6B', 'neutral': '#FFD166', 'positive': '#06D6A0'}
    
    plt.subplot(1, 2, 1)
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', colors=[colors.get(x, '#999') for x in sentiment_counts.index])
    plt.title('Overall Sentiment Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='sentiment', order=['negative', 'neutral', 'positive'],
                  palette=colors)
    plt.title('Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sentiment_distribution.png'), dpi=300)
    plt.show()
    
    # 2. Heatmap of Sentiment by Aspect
    plt.figure(figsize=(14, 8))
    pivot_df = df.pivot_table(index='aspect', columns='sentiment', 
                             aggfunc='size', fill_value=0)
    pivot_df = pivot_df.reindex(ASPECTS)  # Keep aspect order
    pivot_df = pivot_df[['negative', 'neutral', 'positive']]  # Keep sentiment order
    
    sns.heatmap(pivot_df, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
    plt.title('Sentiment Heatmap by Aspect', fontsize=14)
    plt.xlabel('Sentiment')
    plt.ylabel('Aspect')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sentiment_heatmap.png'), dpi=300)
    plt.show()
    
    # 3. Aspect Performance Radar Chart
    from math import pi
    sentiments = ['negative', 'neutral', 'positive']
    aspect_sentiment = df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    aspect_sentiment = aspect_sentiment.reindex(ASPECTS)
    
    # Normalize by total mentions per aspect
    aspect_sentiment_pct = aspect_sentiment.div(aspect_sentiment.sum(axis=1), axis=0)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for idx, aspect in enumerate(ASPECTS[:15]):  # Limit to 15 for display
        ax = axes[idx]
        values = aspect_sentiment_pct.loc[aspect].reindex(sentiments).fillna(0).tolist()
        values += values[:1]  # Close the polygon
        
        angles = [n/float(len(sentiments))*2*pi for n in range(len(sentiments))]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(sentiments)
        ax.set_title(aspect, size=10, y=1.1)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Sentiment Distribution by Aspect (Radar Charts)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'aspect_radar_charts.png'), dpi=300, bbox_inches='tight')
    plt.show()


def calculate_sentiment_scores(df, output_folder):
    """Calculate various sentiment metrics"""
    
    # 1. Sentiment Scores per Aspect
    aspect_scores = []
    for aspect in ASPECTS:
        aspect_data = df[df['aspect'] == aspect]
        total = len(aspect_data)
        if total > 0:
            pos_count = (aspect_data['sentiment'] == 'positive').sum()
            neg_count = (aspect_data['sentiment'] == 'negative').sum()
            
            # Net Sentiment Score (-1 to 1)
            net_score = (pos_count - neg_count) / total if total > 0 else 0
            
            # Positive Ratio
            pos_ratio = pos_count / total if total > 0 else 0
            
            aspect_scores.append({
                'aspect': aspect,
                'total_mentions': total,
                'positive_count': pos_count,
                'negative_count': neg_count,
                'neutral_count': (aspect_data['sentiment'] == 'neutral').sum(),
                'net_sentiment_score': net_score,
                'positive_ratio': pos_ratio,
                'confidence_avg': aspect_data['confidence'].mean()
            })
    
    scores_df = pd.DataFrame(aspect_scores)
    
    # 2. Create Net Sentiment Bar Plot
    plt.figure(figsize=(12, 6))
    scores_sorted = scores_df.sort_values('net_sentiment_score')
    
    colors = []
    for score in scores_sorted['net_sentiment_score']:
        if score > 0:
            colors.append('#06D6A0')  # Green for positive
        elif score < 0:
            colors.append('#FF6B6B')  # Red for negative
        else:
            colors.append('#FFD166')  # Yellow for neutral
    
    bars = plt.barh(scores_sorted['aspect'], scores_sorted['net_sentiment_score'], 
                    color=colors, edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Net Sentiment Score (-1 to 1)')
    plt.title('Net Sentiment Score by Aspect')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'net_sentiment_scores.png'), dpi=300)
    plt.show()
    
    # 3. Confidence Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='confidence', hue='sentiment', 
                 palette={'negative': '#FF6B6B', 'neutral': '#FFD166', 'positive': '#06D6A0'},
                 kde=True, bins=20)
    plt.title('Confidence Distribution by Sentiment')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='sentiment', y='confidence', 
                order=['negative', 'neutral', 'positive'],
                palette={'negative': '#FF6B6B', 'neutral': '#FFD166', 'positive': '#06D6A0'})
    plt.title('Confidence by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confidence_analysis.png'), dpi=300)
    plt.show()
    
    return scores_df


def analyze_review_length_impact(df, output_folder):
    """Analyze relationship between review length and sentiment"""
    
    # Add review length feature
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    df['review_length_category'] = pd.cut(df['review_length'], 
                                         bins=[0, 50, 100, 200, 500, 1000],
                                         labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
    
    # 1. Review Length vs Sentiment
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    length_sentiment = df.groupby(['review_length_category', 'sentiment']).size().unstack()
    length_sentiment.plot(kind='bar', stacked=True, 
                         color=['#FF6B6B', '#FFD166', '#06D6A0'])
    plt.title('Sentiment Distribution by Review Length')
    plt.xlabel('Review Length Category')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x='sentiment', y='review_length', 
                   order=['negative', 'neutral', 'positive'],
                   palette={'negative': '#FF6B6B', 'neutral': '#FFD166', 'positive': '#06D6A0'})
    plt.title('Review Length Distribution by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Review Length (words)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'review_length_analysis.png'), dpi=300)
    plt.show()
    
    return df


def create_summary_report(df, scores_df, output_folder):
    """Generate a comprehensive summary report"""
    
    report = []
    report.append("=" * 60)
    report.append("PRODUCT REVIEW ANALYSIS SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal Reviews Analyzed: {df['review'].nunique()}")
    report.append(f"Total Aspect Mentions: {len(df)}")
    report.append(f"Aspects Analyzed: {len(ASPECTS)}")
    
    # Overall sentiment
    overall_sentiment = df['sentiment'].value_counts(normalize=True)
    report.append(f"\nOverall Sentiment Distribution:")
    for sentiment, percentage in overall_sentiment.items():
        report.append(f"  {sentiment.capitalize()}: {percentage:.1%}")
    
    # Top aspects by mention frequency
    top_aspects = df['aspect'].value_counts().head(5)
    report.append(f"\nTop 5 Most Discussed Aspects:")
    for aspect, count in top_aspects.items():
        report.append(f"  {aspect}: {count} mentions")
    
    # Best and worst performing aspects
    if not scores_df.empty:
        best_aspect = scores_df.loc[scores_df['net_sentiment_score'].idxmax()]
        worst_aspect = scores_df.loc[scores_df['net_sentiment_score'].idxmin()]
        
        report.append(f"\nBest Performing Aspect:")
        report.append(f"  {best_aspect['aspect']} (Net Score: {best_aspect['net_sentiment_score']:.3f}, "
                     f"Positive Ratio: {best_aspect['positive_ratio']:.1%})")
        
        report.append(f"\nWorst Performing Aspect:")
        report.append(f"  {worst_aspect['aspect']} (Net Score: {worst_aspect['net_sentiment_score']:.3f}, "
                     f"Negative Ratio: {worst_aspect['negative_count']/worst_aspect['total_mentions']:.1%})")
    
    # Sentiment confidence
    avg_confidence = df.groupby('sentiment')['confidence'].mean()
    report.append(f"\nAverage Confidence by Sentiment:")
    for sentiment, conf in avg_confidence.items():
        report.append(f"  {sentiment.capitalize()}: {conf:.3f}")
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(output_folder, 'analysis_report.txt'), 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text


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

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_sentiment_distribution(df, output_folder)
    
    # Calculate sentiment scores
    scores_df = calculate_sentiment_scores(df, output_folder)
    scores_df.to_csv(os.path.join(output_folder, 'sentiment_scores.csv'), index=False)
    
    # Analyze review length impact
    df_with_length = analyze_review_length_impact(df, output_folder)
    df_with_length.to_csv(os.path.join(output_folder, 'predictions_with_length.csv'), index=False)
    
    # Create comprehensive report
    report = create_summary_report(df, scores_df, output_folder)
    
    print(f"\nAll analysis completed!")
    print(f"Results saved to: {output_folder}")
    print(f"Files created:")
    print(f"  - predictions.csv (raw predictions)")
    print(f"  - predictions_with_length.csv (with length features)")
    print(f"  - sentiment_scores.csv (aspect-level metrics)")
    print(f"  - 5 visualization PNG files")
    print(f"  - analysis_report.txt (summary report)")

if __name__ == "__main__":
    main()