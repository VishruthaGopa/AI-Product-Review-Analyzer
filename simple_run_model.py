import os
import sys
import subprocess


# Install required packages
def install_packages():
    required_packages = ["tiktoken", "protobuf", "sentencepiece"]
    for package in required_packages:
        try:
            if package == "tiktoken":
                import tiktoken
            elif package == "protobuf":
                import google.protobuf
            elif package == "sentencepiece":
                import sentencepiece
            print(f"{package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_packages()


# Now import and use the model
from transformers import pipeline


# Set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_absa_model():
    try:
        # Try with fast tokenizer first
        classifier = pipeline("text-classification", model="yangheng/deberta-v3-large-absa-v1.1")
    except Exception as e:
        print(f"Fast tokenizer failed: {e}")
        print("Trying with slow tokenizer...")
        # Fall back to slow tokenizer
        classifier = pipeline(
            "text-classification",
            model="yangheng/deberta-v3-large-absa-v1.1",
            use_fast=False
        )
   
    # Test sentences and aspects
    test_cases = [
        {
            "sentence": "The perfume smelled awful, but I liked the way it looked.",
            "aspect": "battery"
        },
        {
            "sentence": "The food was excellent but the service was terrible.",
            "aspect": "food"
        },
        {
            "sentence": "The camera quality is amazing but the battery life is poor.",
            "aspect": "camera"
        }
    ]
   
    for test in test_cases:
        sentence = test["sentence"]
        aspect = test["aspect"]
       
        print(f"\nSentence: {sentence}")
        print(f"Aspect: {aspect}")
       
        result = classifier(sentence, text_pair=aspect)
        print(f"Sentiment: {result}")


if __name__ == "__main__":
    test_absa_model()
