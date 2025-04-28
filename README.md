# CLIP-Flickr8k Project

This repository contains the code for my project on fine-tuning CLIP for image-text alignment on the Flickr8k dataset, as detailed in my Medium blog post: [Building a CLIP like model from scratch](https://medium.com/@modiparth527/building-a-clip-like-model-for-image-text-alignment-a-step-by-step-guide-347d3230fc20).

## Project Overview
- Trained a CLIP-like model from scratch using MobileNetV2 and SentenceTransformer.
- Fine-tuned a pre-trained CLIP model (`openai/clip-vit-base-patch32`) for image-to-text retrieval.
- Achieved a cosine similarity of 0.6102 for retrieval, with the ground-truth caption ranked #1.
- Explored zero-shot classification, achieving correct predictions but with low cosine similarities (~0.2770).

## Files
- `train_clip.py`: Script for training a CLIP-like model from scratch.
- `finetune_clip.py`: Script for fine-tuning a pre-trained CLIP model.
- `image_text_retrieval_clip.py`: Script for image-to-text retrieval with the fine-tuned model.
- `zero_shot_classification_initial.py`: Initial script for zero-shot classification.
- `zero_shot_classification_final.py`: Final script with ensemble prompts for zero-shot classification.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- SentenceTransformers
- Pillow
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision transformers sentence-transformers pillow matplotlib