# AI Product Review Analyzer

A tool for **Aspect-Based Sentiment Analysis (ABSA)** that automatically extracts customer opinions about specific product features from electronics reviews. This system processes product reviews and identifies sentiment (positive, negative, or neutral) toward 15 predefined product aspects such as Camera, Battery, Design, and User Experience. It transforms raw customer feedback into structured insights with visualizations.

## Tech Stack
- **Model**: HuggingFace DeBERTa-v3-base model
- **Frameworks**: PyTorch, Hugging Face Transformers, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: pandas, NumPy
- **Environment**: Google Colab (GPU-enabled)

## Setup
### Quick Start in Google Colab
Simply open any of the notebooks in Google Colab:

1. **Open trained_and_test_model_output.ipynb** - Complete pipeline: trains model, saves best model, runs test data on best saved model, and generates analysis reports with visualizations. All parameters can be modified through the script’s argparse command-line arguments.
2. **Open untrained_model_output.ipynb** - Tests and evaluates the base pre-trained model without fine-tuning

No installation needed since all dependencies are automatically handled in Colab!

## Project Structure

### Core Scripts
- **train_model.py**: Trains ABSA models using K-Fold cross-validation, saves the best model
- **product_review_analyzer.py**: Takes trained model + reviews, generates reports and visualizations
- **run_untrained_model.py**: Tests the base DeBERTa model without fine-tuning

### Data Files
- **traindataset.csv**: Primary training data with text, aspect, and sentiment labels
- **review_data/**: Domain-specific datasets for different product categories

### Output Examples
- **phone_example_analyzer_results/**: Sample output including reports and visualizations
  - 3 CSV files with detailed data
  - 3 PNG visualizations (charts and heatmaps)
  - Text report with summary statistics

### Colab Notebooks
- **trained_and_test_model_output.ipynb**: Full training + analysis pipeline with existing output 
- **untrained_model_output.ipynb**: Base model testing with existing output