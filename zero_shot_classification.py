import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from transformers import CLIPModel, CLIPProcessor

def encode_image(image, model, processor, device):
    """Encode an image into a normalized embedding using CLIP."""
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        return F.normalize(image_features, dim=-1)

def encode_texts(texts, model, processor, device, batch_size=32):
    """Encode a list of texts into normalized embeddings using CLIP."""
    text_features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        text_inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**text_inputs)
            features = F.normalize(features, dim=-1)
        text_features.append(features)
    return torch.cat(text_features, dim=0)

def zero_shot_classification(image_path, categories, prompts, model, processor, device, temperature=0.05):
    """Perform zero-shot classification using CLIP with temperature scaling."""
    # Load and encode the image
    image = Image.open(image_path).convert("RGB")
    image_features = encode_image(image, model, processor, device)  # [1, 512]

    # Encode the category prompts
    text_features = encode_texts(prompts, model, processor, device)  # [num_categories, 512]

    # Compute similarities (cosine similarity)
    raw_similarities = image_features @ text_features.T  # [1, num_categories]
    raw_similarities = raw_similarities.squeeze()  # [num_categories]
    print(f"\nDebug: Raw cosine similarities - Max: {raw_similarities.max().item():.4f}, Min: {raw_similarities.min().item():.4f}")

    # Apply temperature scaling and softmax
    scaled_similarities = raw_similarities / temperature
    softmax_scores = F.softmax(scaled_similarities, dim=0).cpu()  # [num_categories]
    print(f"Debug: Softmax scores - Max: {softmax_scores.max().item():.4f}, Min: {softmax_scores.min().item():.4f}")

    # Predict the category
    predicted_idx = softmax_scores.argmax().item()
    predicted_category = categories[predicted_idx]
    raw_scores = raw_similarities.tolist()
    softmax_scores = softmax_scores.tolist()

    # Print results
    print("\n✅ Zero-Shot Classification Results:")
    print(f"Predicted Category: {predicted_category}")
    print("\nRaw Cosine Similarities for each category:")
    for category, score in zip(categories, raw_scores):
        print(f"{category}: {score:.4f}")
    print("\nSoftmax Scores for each category:")
    for category, score in zip(categories, softmax_scores):
        print(f"{category}: {score:.4f}")

    # Visualize the image with the predicted category
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_category}\nCosine Sim: {raw_scores[predicted_idx]:.4f}\nSoftmax Score: {softmax_scores[predicted_idx]:.4f}")
    plt.axis('off')
    plt.savefig("zero_shot_result.png", dpi=300, bbox_inches='tight')
    plt.show()

    return predicted_category, raw_scores, softmax_scores

def main():
    parser = argparse.ArgumentParser(description="Zero-shot classification with pre-trained CLIP")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Pre-trained CLIP model name")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for softmax scaling")
    args = parser.parse_args()

    # Define categories and improved prompts
    categories = ["Child Climbing Stairs", "Dog Running", "Person Swimming", "Car on Road"]
    prompts = [
        "A young girl in a pink dress climbing stairs indoors",
        "A dog running in a park",
        "A person swimming in a pool",
        "A car driving on a road"
    ]

    # Set device to CPU
    device = torch.device("cpu")

    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    # Perform zero-shot classification
    predicted_category, raw_scores, softmax_scores = zero_shot_classification(
        args.image, categories, prompts, model, processor, device, temperature=args.temperature
    )

if __name__ == "__main__":
    main()