---
base_model: Salesforce/blip-image-captioning-base
library_name: peft
tags:
- base_model:adapter:Salesforce/blip-image-captioning-base
- lora
- transformers
- image-captioning
- vision
- flickr8k
license: apache-2.0
language:
- en
---

# BLIP-Flickr8k-LoRA (Fine-Tuned)

This is a fine-tuned version of the **BLIP (Bootstrapping Language-Image Pre-training)** model, optimized for image captioning using **LoRA (Low-Rank Adaptation)**. It was trained on the **Flickr8k** dataset to generate accurate and descriptive captions for natural images.

## Model Details

### Model Description

This model adapts `Salesforce/blip-image-captioning-base` using the PEFT library. By injecting trainable LoRA adapters into the attention mechanisms (`query` and `value` layers), the model has been tuned to better align with human-annotated captions while keeping the original pre-trained weights frozen. This results in a lightweight, efficient, and high-performance captioning model.

- **Developed by:** Vishnu Purohitham
- **Model type:** Vision-Encoder Text-Decoder (Multimodal)
- **Language(s) (NLP):** English
- **License:** Apache 2.0 (Academic Purpose)
- **Finetuned from model:** [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

## Uses

### Direct Use

- **Automated Image Captioning:** Generating descriptions for images in datasets or applications.
- **Accessibility:** Creating alt-text for visually impaired users.
- **Content Indexing:** Tagging and organizing image collections based on visual content.

### Out-of-Scope Use

- **Medical Diagnosis:** Not intended for interpreting X-rays or medical imaging.
- **Surveillance:** Should not be used for facial recognition or surveillance tasks.
- **Optical Character Recognition (OCR):** While it can read some text, it is not specialized for dense text extraction (OCR).

## Bias, Risks, and Limitations

- **Hallucinations:** Like many VLM models, it may occasionally describe objects that are not present in the image.
- **Bias:** The model inherits biases from the pre-trained BLIP model and the Flickr8k dataset (which consists mostly of natural, Western-centric everyday scenes).

## How to Get Started with the Model

Use the code below to run inference with the model:

```python
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Base Model and Processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 3. Load Fine-Tuned LoRA Adapters
# Replace 'path_to_model' with the local folder or HF Repo ID
model = PeftModel.from_pretrained(base_model, "path_to_your_unzipped_model")
model.to(device)
model.eval()

# 4. Inference
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Example
print(generate_caption("example_image.jpg"))
```

## Training Details

### Training Data

- **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Dataset Size:** ~8,000 images, each paired with 5 human-written captions.
- **Data Splits:** - **Train:** 80%
    - **Validation:** 10%
    - **Test:** 10%
- **Preprocessing:** Images were resized and normalized to 224x224 pixels using the standard `BlipImageProcessor`. Captions were tokenized and padded/truncated to a maximum length.

### Training Procedure

The model was fine-tuned using **LoRA (Low-Rank Adaptation)**, a Parameter-Efficient Fine-Tuning (PEFT) technique. This allowed us to freeze the massive pre-trained weights of the BLIP model and only train a small number of adapter layers, significantly reducing memory usage and training time.

#### Preprocessing

- **Image Resolution:** 224x224
- **Data Augmentation:** None (Standard resizing/normalization only)
- **Precision:** Mixed Precision (`bfloat16`)

#### Training Hyperparameters

- **Training Regime:** `bfloat16` Mixed Precision
- **Optimizer:** AdamW
- **Learning Rate:** 5e-5
- **Batch Size:** 128
- **Epochs:** 5
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05
- **Target Modules:** `query`, `value` (in the attention mechanism)

#### Speeds, Sizes, Times

- **Hardware:** NVIDIA H200 SXM (141GB VRAM)
- **Total Training Time:** ~17 minutes (1067 seconds)
- **Throughput:** High throughput achieved via large batch size (128) and 16 CPU workers.
- **Final Model Size:** The LoRA adapters are extremely lightweight (~MBs) compared to the full model (~GBs).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
The model was evaluated on the **Flickr8k Test Split**, consisting of approximately 1,000 unseen images.

#### Metrics
- **BLEU-4 (Bilingual Evaluation Understudy):** Measures the precision of n-grams (up to 4-grams) between the generated caption and the reference captions.
- **ROUGE-L:** Measures the longest common subsequence to evaluate sentence structure and fluency.

### Results

The model achieved the following scores on the test set:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **BLEU-4** | **0.2534** (25.3%) | Indicates good content overlap with human references (Standard benchmark is >0.20). |
| **ROUGE-L** | **0.4980** (49.8%) | Indicates high fluency and structural similarity to human text. |
| **Training Loss** | **2.48** | Converged from an initial loss of ~2.88. |

## Environmental Impact

- **Hardware Type:** NVIDIA H200 SXM
- **Hours used:** ~0.3 hours
- **Cloud Provider:** University High Performance Computing (Open OnDemand)
- **Compute Region:** USA (Academic Cluster)
- **Carbon Emitted:** Negligible (< 0.05 kg CO2eq)

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** BLIP (Bootstrapping Language-Image Pre-training) is a Vision-Language Pre-training framework. It uses a Vision Transformer (ViT) as an image encoder and a BERT-based text decoder.
- **Objective:** Causal Language Modeling (Image Captioning).

### Compute Infrastructure

#### Hardware
- **GPU:** 1x NVIDIA H200 (141GB HBM3e VRAM)
- **Driver Version:** 570.86.15
- **CUDA Version:** 12.8

#### Software
- **Python:** 3.12+
- **PyTorch:** 2.x with CUDA 12.x support
- **Transformers:** 4.x
- **PEFT:** 0.18.0
- **Accelerate:** Latest
