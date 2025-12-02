import argparse
import os
import sys
import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Disable tokenizer parallel processing to prevent potential conflicts with multi-process training
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ABSAKFoldTrainer:
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        
        # Initialize the ABSA trainer with K-Fold CV
        print(f"Loading base model: {model_name}")

        # Using standard tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model_name = model_name
        
        # Define label mapping

        # Map sentiment labels to numeric IDs for model training
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        # Create reverse mapping to interpret predictions back to human-readable labels
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        print(f"Base model loaded")
    
    def load_dataset(self, csv_path):
        # Load dataset from CSV file
        print(f"Loading dataset from: {csv_path}")
        
        df = pd.read_csv(csv_path)
                                
        # Format data
        df['text'] = df['text'].astype(str)
        df['aspect'] = df['aspect'].astype(str)
        df['sentiment'] = df['sentiment'].str.lower()
        
        # Map sentiments to IDs
        df['label'] = df['sentiment'].map(self.label2id)
        
        # Remove rows with NaN labels
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        print(f"Loaded {len(df)} samples")
        print(f"Sentiment distribution:")
        for sentiment, label_id in self.label2id.items():
            count = (df['label'] == label_id).sum()
            percentage = (count / len(df)) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess dataset for model training
        """
        texts = df['text'].tolist()
        aspects = df['aspect'].tolist()
        labels = df['label'].tolist()
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            aspects,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_fold(self, train_df, val_df, fold_num, output_dir, 
                   epochs=5, batch_size=8, learning_rate=5e-5):
        """
        Train a single fold
        """
        print(f"\n🔧 Training Fold {fold_num + 1}")
        print("-" * 40)
        
        # Create model for this fold
        model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name)
        model.config.id2label = self.id2label
        model.config.label2id = self.label2id
        
        # Preprocess data
        train_encodings = self.preprocess_data(train_df)
        val_encodings = self.preprocess_data(val_df)
        
        train_dataset = Dataset.from_dict(train_encodings)
        val_dataset = Dataset.from_dict(val_encodings)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_num + 1}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f"{fold_output_dir}/logs",
            logging_steps=20,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print(f"   Training samples: {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")
        
        train_result = trainer.train()
        
        # Save the fold model
        trainer.save_model(fold_output_dir)
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        
        return {
            'fold': fold_num + 1,
            'train_loss': train_result.metrics['train_loss'],
            'eval_loss': eval_results['eval_loss'],
            'accuracy': eval_results['eval_accuracy'],
            'f1': eval_results['eval_f1'],
            'precision': eval_results['eval_precision'],
            'recall': eval_results['eval_recall']
        }
    
    def cross_validate(self, df, k=5, output_dir="kfold_models", 
                       epochs=5, batch_size=8, learning_rate=5e-5,
                       random_state=42):
        """
        Perform K-Fold cross-validation
        """
        print(f"\n🎯 Starting {k}-Fold Cross Validation")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize K-Fold (use StratifiedKFold for imbalanced classes)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        # Perform K-Fold CV
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(df, df['label'])):
            # Split data
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            
            # Train this fold
            fold_result = self.train_fold(
                train_df=train_df,
                val_df=val_df,
                fold_num=fold_num,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            fold_results.append(fold_result)
            
            # Print fold results
            print(f"\n📊 Fold {fold_num + 1} Results:")
            for key, value in fold_result.items():
                if key != 'fold':
                    print(f"   {key}: {value:.4f}")
        
        # Calculate overall statistics
        print("\n" + "=" * 60)
        print("📈 CROSS-VALIDATION RESULTS")
        print("=" * 60)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(fold_results)
        
        # Print fold-wise results
        print("\n📋 Fold-wise Performance:")
        print(results_df.to_string(index=False))
        
        # Calculate mean and std
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'train_loss', 'eval_loss']
        
        print("\n📊 Overall Statistics:")
        print("-" * 40)
        for metric in metrics:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"{metric.capitalize():12} : {mean_val:.4f} (±{std_val:.4f})")
        
        # Save results to CSV
        results_df.to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
        
        # Save average model (optional: ensemble of all folds)
        self.save_average_model(results_df, output_dir)
        
        return results_df
    
    def save_average_model(self, results_df, output_dir):
        """
        Create an average model from all folds (simple approach)
        """
        print(f"\n💾 Saving results to: {output_dir}/cv_results.csv")
        
        # You could implement model averaging here
        # For simplicity, we'll just note which fold performed best
        best_fold = results_df.loc[results_df['accuracy'].idxmax(), 'fold']
        print(f"🏆 Best performing fold: {int(best_fold)}")
        print(f"   Accuracy: {results_df.loc[results_df['accuracy'].idxmax(), 'accuracy']:.4f}")
        
        # Copy best model as the final model
        import shutil
        best_fold_dir = os.path.join(output_dir, f"fold_{int(best_fold)}")
        final_model_dir = os.path.join(output_dir, "best_model")
        
        if os.path.exists(best_fold_dir):
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            shutil.copytree(best_fold_dir, final_model_dir)
            print(f"📦 Best model copied to: {final_model_dir}")
    
    def final_test(self, test_df, model_dir):
        """
        Test the best model on a separate test set
        """
        print(f"\n🧪 Final Testing on Separate Test Set")
        print("=" * 50)
        
        # Load the best model
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        predictions = []
        true_labels = []
        
        for _, row in test_df.iterrows():
            inputs = tokenizer(
                row['text'],
                row['aspect'],
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                pred_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(pred_probs, dim=1).item()
            
            predictions.append(predicted_class)
            true_labels.append(self.label2id[row['sentiment'].lower()])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        print(f"📊 Test Set Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Test Samples: {len(test_df)}")
        
        return accuracy, f1

def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation for ABSA Model')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to CSV dataset file')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of folds (default: 5)')
    parser.add_argument('--output', type=str, default='kfold_models',
                       help='Output directory (default: kfold_models)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Epochs per fold (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Test set size for final evaluation (default: 0.2)')
    
    args = parser.parse_args()
    
    print("🎯 ABSA Model K-Fold Cross Validation")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ABSAKFoldTrainer()
        
        # Load dataset
        df = trainer.load_dataset(args.data)
        
        # Split into train+validation and test sets
        from sklearn.model_selection import train_test_split
        train_val_df, test_df = train_test_split(
            df, 
            test_size=args.test_split, 
            random_state=42,
            stratify=df['label']
        )
        
        print(f"\n📈 Dataset Split:")
        print(f"   Train+Validation: {len(train_val_df)} samples")
        print(f"   Test (holdout): {len(test_df)} samples")
        
        # Perform K-Fold cross-validation on train+validation set
        results_df = trainer.cross_validate(
            df=train_val_df,
            k=args.k,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Final test on holdout set
        test_df['sentiment'] = test_df['label'].map(trainer.id2label)
        accuracy, f1 = trainer.final_test(test_df, os.path.join(args.output, "best_model"))
        
        print("\n" + "=" * 60)
        print("✅ K-FOLD CROSS VALIDATION COMPLETE!")
        print("=" * 60)
        print(f"\n📁 Results saved in: {os.path.abspath(args.output)}")
        print(f"📋 Individual fold models saved in subdirectories")
        print(f"🏆 Best model saved as: {args.output}/best_model")
        print(f"📊 Final test accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()