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
from sklearn.model_selection import train_test_split

# Disable tokenizer parallel processing to prevent potential conflicts with multi-process training
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

class ABSATrainer:
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1"):

        # Initialize the ABSA trainer with K-Fold CV
        print(f"Loading base model: {model_name}")

        # Using standard tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model_name = model_name

        # Define label mapping

        # Map sentiment labels to numeric IDs for model training
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        # Create reverse mapping
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

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
        # Type cast as int
        df['label'] = df['label'].astype(int)

        # gets sentiment distribution
        print(f"Loaded {len(df)} samples")
        print(f"Sentiment distribution:")
        for sentiment, label_id in self.label2id.items():
            count = (df['label'] == label_id).sum()
            percentage = (count / len(df)) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")

        return df

    def preprocess_data(self, df):
        # Preprocess dataset for model training

        # Convert each column to list
        texts = df['text'].tolist()
        aspects = df['aspect'].tolist()
        labels = df['label'].tolist()

        # Tokenize
        encodings = self.tokenizer(
            texts,              # List of text sentences
            aspects,            # List of aspects to analyze
            truncation=True,    # Cut off text longer than 512 tokens
            padding=True,       # Add padding to make all inputs same length
            max_length=512,     # Maximum length allowed
            return_tensors='pt' # Return as PyTorch tensors
        )

        return {
            'input_ids': encodings['input_ids'], # The token IDs (text as numbers)
            'attention_mask': encodings['attention_mask'], # Which tokens to pay attention to, not the padding
            'labels': torch.tensor(labels, dtype=torch.long) # Convert labels to proper tensor format
        }


    def compute_metrics(self, eval_pred):
        # Compute metrics for evaluation

        predictions_matrix, labels = eval_pred
        predictions = np.argmax(predictions_matrix, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train_fold(self, train_df, val_df, fold_num, output_dir, epochs=5, batch_size=8, learning_rate=0.00005):
        # Train a single fold

        print(f"\n Training Fold {fold_num + 1}")
        print("-" * 40)

        # Create new untrained model for this fold and set up label mappings
        model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name)
        model.config.id2label = self.id2label
        model.config.label2id = self.label2id

        # Preprocess data
        train_encodings = self.preprocess_data(train_df) # Convert text to numbers
        val_encodings = self.preprocess_data(val_df)

        # Convert training data to HuggingFace Dataset format 
        train_dataset = Dataset.from_dict(train_encodings) 
        # Convert validation data to HuggingFace Dataset format for evaluation
        val_dataset = Dataset.from_dict(val_encodings)

        # Data collator makes sure all examples in a batch have same length,
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Create output directory for each fold
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_num + 1}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_ratio=0.1,# gradually increase learning rate at start
            weight_decay=0.01,# penalty to prevent overfitting
            save_strategy="no",  # Don't save checkpoints during training
            eval_strategy="no",
            save_total_limit=0,
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,
            logging_strategy="no",
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
        }, model # Return the model object

    def cross_validate(self, df, k=5, output_dir="kfold_models", epochs=5, batch_size=8, learning_rate=0.00005, random_state=42):
        #Perform K-Fold cross-validation

        print(f"\n Starting {k}-Fold Cross Validation")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize K-Fold (use StratifiedKFold for imbalanced classes)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state) # random_state: Set fixed random seed for reproducible data splits

        fold_performance_results = []
        fold_saved_models = []  # Store model objects to find the best one

        # Perform K-Fold Cross Validation
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(df, df['label'])):
            # Split data
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            # Train this fold
            fold_result, fold_model_result = self.train_fold(
                train_df=train_df,
                val_df=val_df,
                fold_num=fold_num,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            # Store saved model object for that fold
            fold_saved_models.append(fold_model_result)
            # store performance results for that fold
            fold_performance_results.append(fold_result)

            # Print fold results
            print(f"\n Fold {fold_num + 1} Results:")
            for key, value in fold_result.items():
                if key != 'fold':
                    print(f"   {key}: {value:.4f}")

        # Calculate overall statistics
        print("\n" + "=" * 60)
        print(">> CROSS-VALIDATION RESULTS >>")
        print("=" * 60)

        # Convert results to DataFrame
        result_performance_metrics_df = pd.DataFrame(fold_performance_results)

        # Print fold-wise results
        print("\nFold-wise Performance:")
        print(result_performance_metrics_df.to_string(index=False))

        # Calculate mean and std
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'train_loss', 'eval_loss']

        print("\nOverall Statistics:")
        print("-" * 40)
        for metric in metrics:
            mean_val = result_performance_metrics_df[metric].mean()
            std_val = result_performance_metrics_df[metric].std()
            print(f"{metric.capitalize():12} : {mean_val:.4f} (±{std_val:.4f})")

        # Save results to CSV
        result_performance_metrics_df.to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)

        # Save the best model as a .pth file
        self.save_best_model_single_file(result_performance_metrics_df, fold_saved_models, output_dir)

    def save_best_model_single_file(self, result_performance_metrics_df, models, output_dir):
        #Save the best model as a single .pth file
        print(f"\nSaving results to: {output_dir}/cv_results.csv")

        # Find the best fold based on validation f1
        best_fold_index = result_performance_metrics_df['f1'].idxmax()
        best_fold_num = int(result_performance_metrics_df.loc[best_fold_index, 'fold'])
        best_model = models[best_fold_index]

        print(f"\nBest performing fold: {best_fold_num}")
        print(f"   Accuracy: {result_performance_metrics_df.loc[best_fold_index, 'accuracy']:.4f}")
        print(f"   F1-Score: {result_performance_metrics_df.loc[best_fold_index, 'f1']:.4f}")

        # Save the entire model object
        best_model_path = os.path.join(output_dir, "best_model.pth")
        torch.save({
            'model': best_model,  # Save the entire trained model object
            'label2id': self.label2id,
            'id2label': self.id2label,
            'tokenizer': self.tokenizer,
            'fold_num': best_fold_num,
            'accuracy': result_performance_metrics_df.loc[best_fold_index, 'accuracy'],
            'f1': result_performance_metrics_df.loc[best_fold_index, 'f1']
        }, best_model_path)

        print(f"\nBest model saved as single file: {best_model_path}")

    def final_test(self, test_df, model_path):
        #Test the best model loaded from a single .pth file

        print(f"\nFinal Testing on Separate Test Set")
        print("=" * 50)

        # Load the model from .pth file
        print(f"Loading model from: {model_path}")
        saved_model_load = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # load the best trained model directly
        model = saved_model_load['model']
        # Set model to evaluation model
        model.eval()
        # Get tokenizer
        tokenizer = saved_model_load.get('tokenizer', self.tokenizer)

        predictions = []
        true_labels = []

        for _, row in test_df.iterrows():
            # Tokenize the review text and aspect
            inputs = tokenizer(
                row['text'],
                row['aspect'],
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad(): # don't compute gradients during evaluation
                scores = model(**inputs).logits # pass elements in dictionary as separate arguments
                pred_probs = torch.nn.functional.softmax(scores, dim=-1) # get model predictions probabilities
                predicted_class = torch.argmax(pred_probs, dim=1).item()

            predictions.append(predicted_class)
            true_labels.append(self.label2id[row['sentiment'].lower()])
            
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, labels=[0, 1, 2], average="macro")

        print(f"Test Set Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {macro_f1:.4f}")
        print(f"   Test Samples: {len(test_df)}")

        return accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation for ABSA Model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset file')
    parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--output', type=str, default='kfold_models', help='Output directory (default: kfold_models)')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per fold (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set size for final evaluation (default: 0.2)')
    
    args = parser.parse_args()
    
    print("ABSA Model K-Fold Cross Validation")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ABSATrainer()
        
        # Load dataset
        dataset_df = trainer.load_dataset(args.data)
        
        # Split into train+validation and test sets
        train_val_df, test_df = train_test_split(
            dataset_df, 
            test_size=args.test_split, 
            random_state=42,
            stratify=dataset_df['label']
        )
        
        print(f"\n Dataset Split:")
        print(f"   Train+Validation: {len(train_val_df)} samples")
        print(f"   Test (holdout): {len(test_df)} samples")
        
        # Perform K-Fold cross-validation on train+validation set
        trainer.cross_validate(
            df=train_val_df,
            k=args.k,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Final test on holdout set
        test_df['sentiment'] = test_df['label'].map(trainer.id2label)
        
        # Test with the file model
        best_model_path = os.path.join(args.output, "best_model.pth")
        accuracy, f1 = trainer.final_test(test_df, best_model_path)
        
        print("\n" + "=" * 60)
        print("K-FOLD CROSS VALIDATION COMPLETE")
        print("=" * 60)
        print(f"\n Results saved in: {os.path.abspath(args.output)}")
        print(f" Each fold saved as: fold_X.pth")
        print(f" Best model saved as single file: {args.output}/best_model.pth")
        print(f" Final test accuracy: {accuracy:.4f}")
        print(f" Final test f1: {f1:.4f}")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()