# simple_inference.py
import pandas as pd
from transformers import pipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_on_dataset():
    # Load your dataset
    df = pd.read_csv('train_dataset.csv')
    
    # Load pre-trained ABSA model
    print("Loading ABSA model...")
    classifier = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1"
    )
    
    # Test on first 10 samples
    print("\nTesting on first 10 samples from train_dataset.csv:")
    print("=" * 80)
    
    correct = 0
    total = min(493, len(df))
    
    for i in range(total):
        row = df.iloc[i]
        text = row['text']
        aspect = row['aspect']
        true_sentiment = row['sentiment'].lower()
        
        try:
            # ABSA format: (text, aspect)
            result = classifier(text, text_pair=aspect)
            pred = result[0]['label'].lower()
            confidence = result[0]['score']
            
            # Standardize label
            if 'pos' in pred:
                pred = 'positive'
            elif 'neg' in pred:
                pred = 'negative'
            elif 'neu' in pred:
                pred = 'neutral'
            
            is_correct = pred == true_sentiment
            if is_correct:
                correct += 1
            
            symbol = "✓" if is_correct else "✗"
            print(f"{symbol} '{text[:50]}...'")
            print(f"  Aspect: {aspect}")
            print(f"  True: {true_sentiment} | Pred: {pred} ({confidence:.2%})")
            print()
            
        except Exception as e:
            print(f"❌ Error on sample {i}: {e}")
    
    print(f"\n📊 Accuracy on {total} samples: {correct/total:.0%}")

if __name__ == "__main__":
    test_on_dataset()