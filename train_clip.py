import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import random
import argparse

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(ImageEncoder, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.base_model.last_channel, output_dim)
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.mean([2, 3])
        x = self.base_model.classifier(x)
        return F.normalize(x, dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(TextEncoder, self).__init__()
        self.base_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.projection = nn.Linear(384, output_dim)

    def forward(self, texts):
        embeddings = self.base_model.encode(texts, convert_to_tensor=True, device=texts.device)
        embeddings = self.projection(embeddings)
        return F.normalize(embeddings, dim=-1)

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, limit=None):
        self.image_dir = image_dir
        self.transform = transform
        self.captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                img, caption = line.strip().split('\t')
                img = img.split('#')[0]
                self.captions.setdefault(img, []).append(caption)
        self.image_names = list(self.captions.keys())
        if limit:
            self.image_names = self.image_names[:limit]
        random.shuffle(self.image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_name = img_name.split('.')[0] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.captions[img_name][0]
        return image, caption, img_name

def contrastive_loss(image_features, text_features, temperature=0.07):
    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(image_features), device=image_features.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

def train_model(image_encoder, text_encoder, train_loader, val_loader, device, epochs=20, patience=5):
    optimizer = torch.optim.Adam(image_encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        image_encoder.train()
        text_encoder.eval()
        train_loss = 0
        for images, captions, _ in train_loader:
            images = images.to(device)
            image_features = image_encoder(images)
            with torch.no_grad():
                text_features = text_encoder(captions)
            loss = contrastive_loss(image_features, text_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

        # Validation
        image_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for images, captions, _ in val_loader:
                images = images.to(device)
                image_features = image_encoder(images)
                text_features = text_encoder(captions)
                loss = contrastive_loss(image_features, text_features)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(image_encoder.state_dict(), "image_encoder_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def main():
    parser = argparse.ArgumentParser(description="Train a CLIP-like model on Flickr8k")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to Flickr8k images")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to Flickr8k captions")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Flickr8kDataset(args.image_dir, args.caption_file, transform=transform, limit=args.limit)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_model(image_encoder, text_encoder, train_loader, val_loader, device, epochs=args.epochs, patience=args.patience)

if __name__ == "__main__":
    main()