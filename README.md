CLIP-Flickr8k Project
Welcome to the CLIP-Flickr8k Project! This repository contains the code and resources for my exploration of multimodal learning using CLIP (Contrastive Language-Image Pre-Training) on the Flickr8k dataset. The project focuses on image-text alignment, fine-tuning a pre-trained CLIP model for image-to-text retrieval, and experimenting with zero-shot classification. This work is detailed in my Medium blog post: Building a CLIP-Like Model for Image-Text Alignment: A Step-by-Step Guide.
Project Overview
The goal of this project was to explore the capabilities of CLIP for image-text alignment using the Flickr8k dataset, which contains 8,091 image-caption pairs. I undertook the following steps:

Training a CLIP-Like Model from Scratch: Built a model using MobileNetV2 as the image encoder and SentenceTransformer as the text encoder, trained with contrastive loss (InfoNCE).
Fine-Tuning a Pre-Trained CLIP Model: Fine-tuned openai/clip-vit-base-patch32 for image-to-text retrieval, achieving a cosine similarity of 0.6102 for the ground-truth caption, ranked #1.
Zero-Shot Classification: Used the pre-trained CLIP model to classify images into categories like "Child Climbing Stairs," "Dog Running," "Person Swimming," and "Car on Road," achieving correct predictions but with low cosine similarities (~0.2770).
Challenges and Future Work: Identified limitations due to dataset size, hardware constraints (CPU-only), and low zero-shot similarities, with plans for future improvements.

This project was constrained by the lack of GPU access, limiting the extent of fine-tuning. However, it provided valuable insights into multimodal AI, dataset size impacts, and prompt engineering.
Dataset
The Flickr8k dataset consists of 8,091 images, each with 5 captions. I used the first caption per image, resulting in 8,091 image-caption pairs, split into 90% training (7,282 pairs) and 10% validation (809 pairs). The dataset is not included in this repository due to its size, but you can download it as follows:
Downloading the Dataset

Option 1: Kaggle:

The Flickr8k dataset is available on Kaggle: Flickr8k Dataset.
Download the dataset and extract it to a folder named flickr8k_images in your project directory.
The folder structure should be:MINICLIP/
├── flickr8k_images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── ...
├── captions.txt  # The caption file (e.g., Flickr8k.token.txt)




Option 2: Original Source:

Request access from the official Flickr8k dataset page: Flickr8k Official.
Follow the instructions to download the images and captions.
Place the images in a flickr8k_images folder and the captions in a captions.txt file (or rename Flickr8k.token.txt to captions.txt).



Note: The flickr8k_images/ folder is ignored in this repository (via .gitignore) to keep the repository lightweight. Ensure you have the dataset locally before running the scripts.
Requirements
To run the scripts in this repository, you’ll need the following:

Python: 3.8 or higher
Dependencies:
PyTorch
Torchvision
Transformers (Hugging Face)
SentenceTransformers
Pillow
Matplotlib



Installation

Clone this repository:
git clone https://github.com/modiparth527/MINICLIP.git
cd MINICLIP


Install the required dependencies using the provided requirements.txt:
pip install -r requirements.txt

If you prefer to install manually, use:
pip install torch torchvision transformers sentence-transformers pillow matplotlib



Note: Since this project was run on a CPU-only setup, no GPU is required. However, if you have a GPU, the scripts will automatically utilize it for faster training.
Files
The repository contains the following scripts:

train_clip.py: Trains a CLIP-like model from scratch using MobileNetV2 (image encoder) and SentenceTransformer (text encoder) with contrastive loss.
finetune_clip.py: Fine-tunes a pre-trained CLIP model (openai/clip-vit-base-patch32) for image-to-text retrieval on Flickr8k.
image_text_retrieval_clip.py: Performs image-to-text retrieval using the fine-tuned CLIP model, ranking captions based on cosine similarity.
zero_shot_classification_initial.py: Initial script for zero-shot classification with simple prompts.
zero_shot_classification_final.py: Improved zero-shot classification script with ensemble prompts and temperature scaling.
requirements.txt: Lists the Python dependencies needed to run the scripts.
images/: Contains output visualizations (topk_results.png and zero_shot_result.png).

Usage
Below are instructions to replicate the experiments in this project. Ensure you have the Flickr8k dataset downloaded and placed in the flickr8k_images/ folder as described in the "Dataset" section.
1. Fine-Tune the CLIP Model
Fine-tune the pre-trained CLIP model on Flickr8k using finetune_clip.py:
python finetune_clip.py --image_dir flickr8k_images --caption_file captions.txt --model_name openai/clip-vit-base-patch32 --batch_size 32 --epochs 10 --patience 3


Arguments:
--image_dir: Path to the folder containing Flickr8k images.
--caption_file: Path to the captions file (e.g., captions.txt).
--model_name: Pre-trained CLIP model (default: openai/clip-vit-base-patch32).
--batch_size: Batch size (default: 32).
--epochs: Number of epochs (default: 10).
--patience: Patience for early stopping (default: 3).



The fine-tuned model will be saved in a folder named clip_finetuned_best.
2. Image-to-Text Retrieval
Use the fine-tuned model to retrieve captions for a test image with image_text_retrieval_clip.py:
python image_text_retrieval_clip.py --image flickr8k_images/1000268201_693b08cb0e.jpg --image_dir flickr8k_images --caption_file captions.txt --model_dir clip_finetuned_best --top_k 2 --limit 8091 --output topk_results.png


Arguments:
--image: Path to the test image.
--image_dir: Path to the folder containing Flickr8k images.
--caption_file: Path to the captions file.
--model_dir: Path to the fine-tuned CLIP model.
--top_k: Number of top matches to retrieve (default: 2).
--limit: Number of captions to search (default: 500; use 8091 for the full dataset).
--output: Output path for the visualization (default: topk_results.png).



This script will output the top-2 captions with their cosine similarities and save a visualization to topk_results.png.
3. Zero-Shot Classification
Perform zero-shot classification with zero_shot_classification_final.py:
python zero_shot_classification_final.py --image flickr8k_images/1000268201_693b08cb0e.jpg --model_name openai/clip-vit-base-patch32 --temperature 0.04


Arguments:
--image: Path to the test image.
--model_name: Pre-trained CLIP model (default: openai/clip-vit-base-patch32).
--temperature: Temperature for softmax scaling (default: 0.04).



This script will classify the image into categories ("Child Climbing Stairs," "Dog Running," "Person Swimming," "Car on Road") and save a visualization to zero_shot_result.png.
Results
Here are the key results from the project, including visualizations generated by the scripts.
Image-to-Text Retrieval
The fine-tuned CLIP model was used to retrieve captions for a test image (1000268201_693b08cb0e.jpg). The ground-truth caption was ranked #1 with a cosine similarity of 0.6102, indicating good alignment but room for improvement (ideal similarity: 0.9–1.0).
Results:

Raw Cosine Similarities: Max 0.6102, Min -0.2820
Softmax Scores (temperature=0.02): Max 0.4900
Test Image Info:
Caption: "A child in a pink dress is climbing up a set of stairs in an entry way." (cosine_sim=0.6102)
Ground-truth ranked #1


Top-2 Matches:
Rank 1: "A child in a pink dress is climbing up a set of stairs in an entry way." (cosine_sim=0.6102)
Rank 2: "A girl in pink and purple is climbing a ladder." (cosine_sim=0.5500)



Visualization:

Zero-Shot Classification
The pre-trained CLIP model was used for zero-shot classification with ensemble prompts and a temperature of 0.04. The test image was correctly classified as "Child Climbing Stairs," but the cosine similarity was low (0.2770), suggesting potential issues with prompt alignment or image complexity.
Results:

Raw Cosine Similarities:
Child Climbing Stairs: 0.2770
Dog Running: 0.1383
Person Swimming: 0.1668
Car on Road: 0.1211


Softmax Scores (temperature=0.04):
Child Climbing Stairs: 0.9200
Dog Running: 0.0300
Person Swimming: 0.0400
Car on Road: 0.0100



Visualization:

Challenges and Lessons Learned

Dataset Size:
Flickr8k’s 8,091 pairs were insufficient for training a CLIP-like model from scratch, leading to poor initial alignment (cosine similarity ~0.7800). Fine-tuning a pre-trained model improved results but still fell short of the ideal 0.9–1.0 similarity.


Hardware Constraints:
Without GPU access, fine-tuning was limited to 5 epochs (best at epoch 2), preventing further optimization.


Zero-Shot Performance:
Despite correct predictions, zero-shot cosine similarities were low (~0.2770), even with prompt engineering and ensemble prompts, indicating potential mismatches with CLIP’s pre-trained knowledge.


Prompt Sensitivity:
Zero-shot performance was highly sensitive to prompt phrasing, but improvements were limited, suggesting deeper issues with the image or model.



Future Work
To improve the project, I plan to explore the following:

Test Zero-Shot on Other Images:
Apply zero-shot classification to other Flickr8k images to determine if the low cosine similarities are image-specific.


Binary Classification:
Simplify to binary classification (e.g., "Is this a child climbing stairs? Yes/No") to improve cosine similarities.


Zero-Shot Retrieval:
Use CLIP for zero-shot retrieval, comparing the image against specific captions instead of broad categories.


Larger CLIP Model:
Experiment with a larger model like openai/clip-vit-large-patch14 for better zero-shot performance.


Re-Fine-Tune with GPU:
With GPU access, re-fine-tune with more epochs (e.g., 15), a higher learning rate (1e-5), and gradient clipping to target a cosine similarity of 0.9–1.0.


Larger Dataset:
Fine-tune on a larger dataset like CC3M (3M pairs) for better generalization.



Contributing
Contributions are welcome! If you have ideas for improvements, feel free to open an issue or submit a pull request. Please ensure your changes align with the project’s goals and include appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
MIT License

Copyright (c) 2025 Parth Modi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contact
For questions or feedback, feel free to reach out via email at modiparth527@example.com or open an issue on GitHub.

Date: April 28, 2025
