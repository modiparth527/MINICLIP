import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from transformers import CLIPModel, CLIPProcessor

# --- Dataset ---
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, processor, transform=None, limit=None):
        self.image_dir = image_dir
        self.transform = transform
        self.processor = processor
        self.captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                img, caption = line.strip().split('\t')
                img = img.split('#')[0]  # e.g., "2258277193_586949ec62.jpg#0" -> "2258277193_586949ec62.jpg"
                self.captions.setdefault(img, []).append(caption)
        self.image_names = list(self.captions.keys())
        # Debug: Print the first few image names to verify
        print("Sample image names:", self.image_names[:5])
        if limit:
            self.image_names = self.image_names[:limit]
        random.shuffle(self.image_names)  # Shuffle for train/val split

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        # Ensure img_name has the correct extension
        img_name = img_name.split('.')[0] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error: Could not find image at {img_path}")
            raise e
        
        # Apply CLIP processor for images (includes resizing and normalization)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        image_tensor = inputs['pixel_values'].squeeze(0)  # [3, 224, 224]

        # Apply additional augmentations if specified
        if self.transform:
            image_tensor = self.transform(image_tensor)

        caption = self.captions[img_name][0]  # Use first caption
        return image_tensor, caption, img_name

# --- Contrastive Loss ---
def contrastive_loss(image_features, text_features, temperature=0.07):
    logits = (image_features @ text_features.T) / temperature  # [batch_size, batch_size]
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# --- Data Augmentation ---
preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
])

# --- Training Function ---
def train_model(model, processor, train_loader, val_loader, device, epochs=10, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-5)  # Lower LR for fine-tuning
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (images, captions, _) in enumerate(train_loader):
            images = images.to(device)  # [batch_size, 3, 224, 224]
            
            # Process text with CLIP processor
            text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # Forward pass
            outputs = model(pixel_values=images, **text_inputs, return_dict=True)
            image_features = outputs.image_embeds  # [batch_size, 512]
            text_features = outputs.text_embeds    # [batch_size, 512]
            
            # Normalize embeddings
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Compute contrastive loss
            loss = contrastive_loss(image_features, text_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        rank_1_correct = 0
        total = 0
        with torch.no_grad():
            for images, captions, img_names in val_loader:
                images = images.to(device)
                text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                outputs = model(pixel_values=images, **text_inputs, return_dict=True)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                loss = contrastive_loss(image_features, text_features)
                val_loss += loss.item()

                # Compute rank@1 for retrieval
                similarities = image_features @ text_features.T
                predictions = similarities.argmax(dim=1)
                labels = torch.arange(len(images), device=device)
                rank_1_correct += (predictions == labels).sum().item()
                total += len(images)

        val_loss /= len(val_loader)
        rank_1 = rank_1_correct / total
        print(f"Validation Loss: {val_loss:.4f}, Rank@1: {rank_1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save the entire CLIP model
            model.save_pretrained("clip_finetuned_best")
            processor.save_pretrained("clip_finetuned_best")
            print("Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best model at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    return model

def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Flickr8k")
    parser.add_argument("--image_dir", type=str, default="Flickr8k_Dataset", help="Directory of images")
    parser.add_argument("--caption_file", type=str, default="Flickr8k_text/Flickr8k.token.txt", help="Caption file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of image-caption pairs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Max number of epochs")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Dataset
    dataset = Flickr8kDataset(args.image_dir, args.caption_file, processor, transform=preprocess, limit=args.limit)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Debug: Print sample indices and image names
    print("Sample train indices:", train_dataset.indices[:5])
    print("Sample train image names:", [dataset.image_names[i] for i in train_dataset.indices[:5]])
    print("Sample val indices:", val_dataset.indices[:5])
    print("Sample val image names:", [dataset.image_names[i] for i in val_dataset.indices[:5]])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train
    model = train_model(model, processor, train_loader, val_loader, device, args.epochs, args.patience)

    # Save final model
    model.save_pretrained("clip_finetuned_final")
    processor.save_pretrained("clip_finetuned_final")

if __name__ == "__main__":
    main()