import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse
import textwrap
from transformers import CLIPModel, CLIPProcessor

# --- Dataset for Inference ---
class Flickr8kInferenceDataset(Dataset):
    def __init__(self, image_dir, caption_file, processor, limit=None):
        self.image_dir = image_dir
        self.processor = processor
        self.captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                img, caption = line.strip().split('\t')
                img = img.split('#')[0]
                self.captions.setdefault(img, []).append(caption)
        self.image_names = list(self.captions.keys())
        if limit:
            self.image_names = self.image_names[:limit]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_name = img_name.split('.')[0] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        caption = self.captions[img_name][0]
        return image, caption, img_name

def encode_image(image, model, processor, device):
    """Encode an image into a normalized embedding using CLIP."""
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        return F.normalize(image_features, dim=-1)

def encode_captions(captions, model, processor, device, batch_size=32):
    """Encode a list of captions in batches using CLIP."""
    text_features = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        text_inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**text_inputs)
            features = F.normalize(features, dim=-1)
        text_features.append(features)
    return torch.cat(text_features, dim=0)

def image_text_retrieval(image_path, captions_dict, model, processor, device, top_k=2, limit=500, test_caption=None):
    """Perform image-to-text retrieval with softmax-normalized scores."""
    # Get test image filename for caption lookup
    test_image_name = os.path.basename(image_path)
    test_caption_info = None
    ground_truth_rank = None

    # Check for ground-truth caption if test_caption is not provided
    if test_caption is None and test_image_name in captions_dict:
        test_caption = captions_dict[test_image_name][0]

    # Prepare captions and image names for retrieval
    all_captions = []
    image_names = []
    for img_name, caps in list(captions_dict.items())[:limit]:
        all_captions.append(caps[0])
        image_names.append(img_name)

    if not all_captions:
        raise ValueError("No captions found in the dataset.")

    # Encode test image
    test_image = Image.open(image_path).convert("RGB")
    image_features = encode_image(test_image, model, processor, device)  # [1, 512]

    # Encode captions (including the test caption if provided)
    captions_to_encode = all_captions.copy()
    test_caption_idx = None
    if test_caption:
        captions_to_encode.append(test_caption)
        test_caption_idx = len(captions_to_encode) - 1

    text_features = encode_captions(captions_to_encode, model, processor, device)  # [limit + 1, 512]

    # Compute similarities (cosine similarity scaled by temperature)
    temperature = 0.02  # Adjusted temperature for sharper softmax
    similarities = (image_features @ text_features.T) / temperature  # [1, limit + 1]
    if similarities.dim() == 1:
        similarities = similarities.unsqueeze(0)

    # Debug: Print raw cosine similarities
    raw_similarities = image_features @ text_features.T  # [1, limit + 1]
    print(f"Debug: Raw cosine similarities - Max: {raw_similarities.max().item():.4f}, Min: {raw_similarities.min().item():.4f}")

    # Apply softmax to normalize scores into probabilities
    softmax_scores = F.softmax(similarities, dim=1)  # [1, limit + 1]

    # Extract test caption score (consistent with top-k scores)
    if test_caption and test_caption_idx is not None:
        test_score = softmax_scores[0, test_caption_idx].item()
        test_caption_info = {'caption': test_caption, 'score': test_score}

    # Get top-k results (excluding the test caption if appended)
    softmax_scores = softmax_scores[:, :len(all_captions)]  # [1, limit]
    top_k_results = softmax_scores.topk(top_k, dim=1)
    top_scores = top_k_results.values.squeeze().cpu()
    top_indices = top_k_results.indices.squeeze().cpu()

    # Handle top_k=1 case
    if top_k == 1:
        top_scores = torch.tensor([top_scores]) if top_scores.dim() == 0 else top_scores
        top_indices = torch.tensor([top_indices]) if top_indices.dim() == 0 else top_indices

    # Collect results
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            'rank': rank + 1,
            'caption': all_captions[idx],
            'image_name': image_names[idx],
            'score': top_scores[rank].item()
        })

    # Check if ground-truth caption is in top-k
    if test_caption and test_image_name in captions_dict and test_caption == captions_dict[test_image_name][0]:
        for idx, caption in enumerate(all_captions):
            if caption == test_caption:
                top_k_indices = softmax_scores.topk(top_k, dim=1).indices.squeeze().cpu()
                if idx in top_k_indices:
                    ground_truth_rank = (top_k_indices == idx).nonzero(as_tuple=True)[0].item() + 1
                break

    # Debug: Print range of softmax scores
    print(f"Debug: Softmax score range - Max: {softmax_scores.max().item():.4f}, Min: {softmax_scores.min().item():.4f}")

    return results, test_caption_info, ground_truth_rank

def visualize_results(test_image_path, results, test_caption_info, image_dir, output_file=None, max_caption_len=40):
    """Visualize the test image and top-k matching images with consistent annotations."""
    top_k = len(results)
    fig, axs = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))

    # Plot test image
    axs[0].imshow(Image.open(test_image_path))
    if test_caption_info:
        caption = textwrap.shorten(test_caption_info['caption'], width=max_caption_len, placeholder="...")
        title = f"Query: {caption}\nScore: {test_caption_info['score']:.4f}"
    else:
        title = f"Query: {os.path.basename(test_image_path)}"
    axs[0].set_title(title, fontsize=10)
    axs[0].axis('off')

    # Plot top-k matches
    for i, result in enumerate(results):
        img_path = os.path.join(image_dir, result['image_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        axs[i + 1].imshow(Image.open(img_path))
        caption = textwrap.shorten(result['caption'], width=max_caption_len, placeholder="...")
        title = f"Top {result['rank']}: {caption}\nScore: {result['score']:.4f}"
        axs[i + 1].set_title(title, fontsize=10)
        axs[i + 1].axis('off')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Image-to-text retrieval with fine-tuned CLIP")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--caption_file", type=str, default="Flickr8k_text/Flickr8k.token.txt", help="Path to caption file")
    parser.add_argument("--image_dir", type=str, default="Flickr8k_Dataset", help="Directory containing dataset images")
    parser.add_argument("--model_dir", type=str, default="clip_finetuned_best", help="Path to fine-tuned CLIP model")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top matches to retrieve")
    parser.add_argument("--limit", type=int, default=500, help="Limit number of captions to process")
    parser.add_argument("--test_caption", type=str, default=None, help="Optional caption for test image")
    parser.add_argument("--output", type=str, default="topk_results.png", help="Output file for visualization")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Test image not found: {args.image}")
    if not os.path.exists(args.caption_file):
        raise FileNotFoundError(f"Caption file not found: {args.caption_file}")
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if args.top_k < 1:
        raise ValueError("top_k must be at least 1")
    if args.limit < args.top_k:
        raise ValueError("limit must be at least top_k")

    # Initialize device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_dir).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_dir)
    model.eval()

    # Load captions
    dataset = Flickr8kInferenceDataset(args.image_dir, args.caption_file, processor, limit=args.limit)
    captions_dict = {img_name: [caption] for _, caption, img_name in dataset}

    # Perform retrieval
    print(f"\n✅ Querying with test image: {args.image}")
    results, test_caption_info, ground_truth_rank = image_text_retrieval(
        args.image, captions_dict, model, processor, device,
        top_k=args.top_k, limit=args.limit, test_caption=args.test_caption
    )

    # Print results
    print("\n✅ Test Image Info:")
    if test_caption_info:
        print(f"Caption: {test_caption_info['caption']} (score={test_caption_info['score']:.4f})")
        if ground_truth_rank:
            print(f"Ground-truth caption ranked #{ground_truth_rank} in top-{args.top_k}")
        elif test_caption_info['caption'] == captions_dict.get(os.path.basename(args.image), [None])[0]:
            print("Ground-truth caption not in top-{} results".format(args.top_k))
    else:
        print("No caption provided or found for test image.")
    print("\n✅ Top {} Matches:\n".format(args.top_k))
    for result in results:
        print(f"Rank {result['rank']}: {result['caption']} (score={result['score']:.4f})")

    # Visualize results
    visualize_results(
        args.image, results, test_caption_info, args.image_dir,
        output_file=args.output if args.output.lower() != "none" else None
    )

if __name__ == "__main__":
    main()