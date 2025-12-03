import os
import sys
import subprocess

#Install required packages
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


from transformers import pipeline

#set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_absa_model():
    classifier = pipeline("text-classification", model="yangheng/deberta-v3-large-absa-v1.1")

    # Test sentences and aspects
    test_cases = [
        {
            "sentence": "The battery drains quicker than I hoped if I leave all features on.",
            "aspect": "battery"
        },
        {
            "sentence": "The battery drains quicker than I hoped if I leave all features on.",
            "aspect": "price"
        },
        {
            "sentence": "The battery drains quicker than I hoped if I leave all features on.",
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
    print("Starting simple run of untrained model...")
    test_absa_model()
