# InfoRetrieval – AI-Powered Unified Content Summarization & Retrieval System

## 🚀 Overview

**InfoRetrieval** is an AI-powered system designed to help users store, summarize, and semantically search across multiple forms of information—**URLs, text notes, and images/screenshots**.

The system extracts meaningful content from these sources using:

* **T5-small** and **T5-base** pre-trained text summarization models
* A **fine-tuned BLIP image captioning model** (PEFT + LoRA)
* Trafilatura for high-quality **webpage text extraction**
* A vector database powered by **sqlite-vector**

This project was developed in **Python (PyCharm)** as part of an academic AI course project.

---

## 📌 Features

### 🔹 Multi-input Processing

* **URLs** → Cleaned text extraction using **Trafilatura** → Summarized
* **Text Notes** → Summarized using **T5-small** or **T5-base** (both pre-trained, not fine-tuned)
* **Images/Screenshots** → Captioned using our **fine-tuned BLIP model**

### 🔹 AI Models Used

#### 📘 Text Summarization (HuggingFace)

We used two variants depending on the use case:

* **T5-small**

  * Faster and lightweight
  * Useful for quick inference during development

* **T5-base**

  * Higher quality summaries
  * Used mainly for final summarization outputs

Both models were used **as pre-trained** (no fine-tuning).
**Next improvement:** fine-tuning on domain-specific datasets like XSum or custom bookmark summaries.

---

#### 🖼️ Image Captioning (Fine-Tuned BLIP)

* Base model: **BLIP**
* Fine-tuned using:

  * **PEFT + LoRA**
  * **Flickr8k** dataset
  * Split: **80% training / 10% validation / 10% testing**
* Training hardware: **NVIDIA H200 GPU (HPC cluster)**

**Post-fine-tuning evaluation:**

* Performed inference on Flickr8k test split
* Generated captions for **LightShot13k** screenshot dataset
* Fine-tuned BLIP produced significantly more accurate and descriptive captions

---

### 🔹 Storage & Retrieval

* Vector embeddings stored using **sqlite-vector**
* Search using **semantic similarity**
* Unified summary database

---

## 🧠 System Architecture

The system processes different inputs using the following flow:

```
URLs ───► Trafilatura Extraction ───►   T5-small / T5-base Summarization ───► Summary Database
                                      
Text Notes ─────────────────────────►   T5-small / T5-base Summarization ───► Summary Database
                                                                                
Images ───► Fine-tuned BLIP Captioning ──────────────────────────────────────► Summary Database
```


## 📂 Project Structure

```
InfoRetrieval/
│── api.py                 # Backend API server + vector DB logic
│── streamlit_app.py       # Streamlit user interface
│── summarization.py       # t5 model summarization
│── requirements.txt       # Python dependencies
│── image_captioning.py    # Fine-tuned Image Caption generation script
│── URL_Test.py            # Script for testing URL text summarization (Can be skipped for use as this was created for testing the summarization)
│── utils.py               # Helper scripts (Trafilatura extraction, embeddings)
│── config/                # Setup for model/db path
│── packages/              # sqlite-vector DB files
│── models/                # Blip Fine tuned model
│── README.md
```

---

## 🛠️ Installation & Setup

### ✔ 1. Clone the repository

```bash
git clone https://github.com/TheJASSZ/InfoRetrieval
cd InfoRetrieval
```

### ✔ 2. Install Python dependencies

Make sure you have Python 3.9+ installed.

```
pip install -r requirements.txt
```

### ✔ 3. Install **sqlite-vector** (required for vector search)

Download the version appropriate for your OS from the official repo:

🔗 [https://github.com/sqliteai/sqlite-vector](https://github.com/asg017/sqlite-vec)
After clicking the link above scroll down to and click on release (https://github.com/asg017/sqlite-vec/releases)

Scroll down to the "Assets" section of the latest release. Download and extract the file:

Windows: sqlite-vec-X.Y.Z-loadable-windows-x86_64.zip (Look for vec0.dll inside)

macOS: sqlite-vec-X.Y.Z-loadable-macos-x86_64.tar.gz (Look for vec0.dylib)

Linux: sqlite-vec-X.Y.Z-loadable-linux-x86_64.tar.gz (Look for vec0.so)

Important: Place the extracted file (vec0.dll, vec0.dylib, or vec0.so) directly in the project root folder so the app can find it.

### ✔ 4. Run the backend API

```
python api.py
```

Keep this terminal running.

### ✔ 5. Launch the Streamlit interface

Open a new terminal and run:

```
streamlit run streamlit_app.py
```

A browser window (Google Chrome recommended) will automatically open with the UI.

---

## 🖥️ How to Use

The Streamlit app provides three input options:

### 🔹 1. **Paste URL**

Uses **Trafilatura → T5-small/T5-base → embedding → vector DB**

### 🔹 2. **Upload / Paste File Path**

* Images → fine-tuned BLIP captions
* Text files → T5 summarization

### 🔹 3. **Search Query**

Enter any phrase or concept and the system performs:

* Vector similarity search
* Returns relevant summaries or captions

---

## 🧪 Image Captioning – Dataset & Results

### 📂 Dataset Summary

| Dataset      | Size               | Purpose                           |
| ------------ | ------------------ | --------------------------------- |
| Flickr8k     | 8000 images        | Fine-tuning BLIP (80-10-10 split) |
| LightShot13k | 13,000 screenshots | Caption generation inference      |

### 🏋️ Fine-Tuning Setup

* Model: BLIP
* Techniques: PEFT + LoRA
* Hardware: H200 GPU
* Goal: Improve caption quality for real-world images/screenshots

### 📊 Results

* BLIP showed strong improvement after fine-tuning
* Generated high-quality captions for LightShot13k

---

## 🎯 Future Improvements

* Fine-tune **T5-base** for summarization
* OCR integration for text-heavy images
* Better JS-rendered webpage scraping (Playwright/Selenium)
* Full cloud backend + user authentication
* Topic clustering and automatic tagging

---

## 🤝 Contributions

### **Vishnu Purohitham**

* Image captioning pipeline
* Flickr8k dataset preparation
* BLIP fine-tuning (PEFT + LoRA)
* Caption generation on test split + LightShot13k
* Joint work on URL extraction

### **Raghav Jadia**

* Text summarization using T5-small and T5-base
* Backend API development
* sqlite-vector integration
* URL extraction logic (Trafilatura)

### **Joint Contributions**

* URL pipeline
* End-to-end system architecture & testing

---

## 📄 License

This project is for academic purposes.

---

If you found this project useful, please ⭐ star the repository!
