import pandas as pd
from transformers import pipeline
import os
from sklearn.metrics import f1_score, accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_on_dataset():
    # Load the dataset
    test_path = 'train_dataset.csv'
    df = pd.read_csv(test_path)
    
    # Load pre-trained base ABSA model
    print("Loading ABSA model...")
    classifier = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1"
    )
    
    # Test on the samples
    print("\nTesting on all samples from test_path:")
    print("=" * 80)
    
    total = len(df)
    true_labels = []
    pred_labels = []
    
    for i in range(total):
        row = df.iloc[i]
        text = row['text']
        aspect = row['aspect']
        true_sentiment = row['sentiment'].lower()
        
        # ABSA format
        result = classifier(text, text_pair=aspect)
        pred = result[0]['label'].lower()
        confidence = result[0]['score']
        
        true_labels.append(true_sentiment)
        pred_labels.append(pred)
        is_correct = pred == true_sentiment
        
        symbol = "PASS" if is_correct else "FAIL"
        print(f"{symbol} '{text}'")
        print(f"  Aspect: {aspect}")
        print(f"  True: {true_sentiment} | Pred: {pred} ({confidence:.2%})")
        print()
            
    
    # Calculate metrics
    macro_f1 = f1_score(true_labels, pred_labels, labels=["negative", "neutral", "positive"], average="macro")
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print(f"\nMacro F1 on {total} samples: {macro_f1:.4f}")
    print(f"Accuracy on {total} samples: {accuracy:.4%}")

if __name__ == "__main__":
    test_on_dataset()
