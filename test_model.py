from transformers import pipeline

# Load the trained model
trained_model_path = "./my-trained-absa-model"

# Create pipeline with trained model
absa_pipeline = pipeline(
    "token-classification",
    model=trained_model_path,
    tokenizer=trained_model_path,
    aggregation_strategy="simple",
)

# Test the model
test_texts = [
    "The phone has great performance but awful camera.",
    "The screen is brilliant however the battery drains quickly.",
    "Comfortable keyboard and responsive touchpad but heavy weight."
]

for text in test_texts:
    print(f"\nText: {text}")
    results = absa_pipeline(text)
    for result in results:
        print(f"  Aspect: '{result['word']}' -> {result['entity_group']} (score: {result['score']:.3f})")