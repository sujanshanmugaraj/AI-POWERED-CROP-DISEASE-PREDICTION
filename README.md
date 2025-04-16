# 🌿 AI-POWERED CROP DISEASE PREDICTION

**DEEP LEARNING LAB (20XD68)**  
**Developed by:**  
👩‍💻 Akila R (22PD04)  
👨‍💻 Sujan S (22PD35)

---

## 🧩 Problem Statement

Crop diseases threaten agricultural productivity and often lead to economic losses. Traditional detection methods are slow, labor-intensive, and heavily reliant on expert knowledge, often ignoring critical environmental conditions.  
This project proposes an intelligent, scalable system using **Vision Transformers (ViT)** and environmental features for **multimodal disease prediction**, aiding timely decision-making in precision agriculture.

---

## 📄 Abstract

This project presents a **Multimodal Deep Learning Framework** combining:

- 🌿 **Leaf Image Analysis** using Vision Transformer (ViT)  
- 🌡️ **Environmental Sensor Data** (e.g., temperature, humidity, soil moisture)

A **cross-attention mechanism** fuses both modalities, enhancing classification performance.  
This intelligent and scalable system helps farmers detect crop diseases early and accurately, advancing smart agriculture.

---

## 📊 Dataset Description

The dataset is sourced from Kaggle:  
🔗 [Multimodal Plant Disease Dataset by Subham Divakar](https://www.kaggle.com/datasets/shubhamdivakar/multimodal-plant-disease-dataset-by-subham-divakar)

Each sample includes:

- **Leaf Images**: High-resolution RGB images labeled with disease categories.
- **Numerical Features**: 7 tabular features (e.g., temperature, humidity, pH, rainfall).

> All data are stored in `mapped_data_with_images.csv`, with rows linking image paths and environmental attributes.  
> Dataset split: **80% training**, **20% testing** (stratified sampling).

---

## 📚 Literature Study

**Paper Title:**  
_A Channel Attention-Driven Optimized CNN for Efficient Early Detection of Plant Diseases in Resource-Constrained Environment_  
**Authors:** Sana Parez, Naqqash Dilshad, Jong Weon Lee  
🔗 [DOI Link](https://doi.org/10.3390/agriculture15020127)

✅ The study's lightweight LeafNet CNN inspired our decision to use efficient architectures and focus on **multimodal fusion** for performance enhancement.

---

## 🛠️ Tools and Technologies

### Development Environment

- Python 3.x  
- Jupyter Notebook / VS Code  
- Google Colab  
- Git & GitHub  
- Streamlit (web app)  
- Google Gemini (via LangChain)  

### Key Libraries

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, scikit-learn |
| **Visualization** | matplotlib, seaborn |
| **Computer Vision** | OpenCV, PIL |
| **Deep Learning** | PyTorch, torchvision, EfficientNet, ResNet50, MobileNetV2, ViT |
| **Voice/Chat AI** | speech_recognition, pyttsx3, langchain_google_genai |

---

## 🧠 Methods & Implementation

### 1. Preprocessing

- **Images**: Resized (224×224), augmented (flip, rotation), normalized  
- **Numerical Data**: Standard Scaler used; class labels encoded  

### 2. Model Architecture

#### Multimodal Neural Network

- **Visual Branch**: ViT-B_16 (ImageNet pretrained)
- **Numerical Branch**: Feedforward NN (Linear → ReLU → LayerNorm → Dropout)
- **Fusion**: Multihead Cross-Attention
- **Classifier**: MLP with final softmax output

### 3. Training

- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam (lr=0.0001)  
- **Epochs**: 15 (early stopping)  
- **Metrics**: Accuracy, F1-score, Confusion Matrix  

### 4. Inference

- Model file: `vit_multimodal_best.pth`  
- Inputs: Leaf image + optional numerical features  
- Outputs: Disease class + confidence score  

### 5. Streamlit Web Interface

- 🔼 Upload images  
- 📈 Enter optional environmental features  
- 🎙️ Voice or 💬 text interaction with Gemini-powered chatbot  
- 🔊 Text-to-speech support via `pyttsx3`

---

## 📈 Model Evaluation

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.95%** |
| **F1-Score** | 0.94 |
| **Precision** | High (most > 90%) |
| **Recall** | Balanced across classes |

### 🔍 Observations

- Multimodal ViT outperformed image-only models  
- Cross-attention reduced misclassification  
- High confidence in: Apple Scab, Grape Blight, Potato Early Blight  
- Lower confidence when environmental data was missing

---

## 🖥️ Results – Web App Overview

### 🌐 Homepage Interface

- **Text Input** – Manual queries  
- **Voice Input** – Hands-free diagnosis  
- **Image Upload** – Triggers ViT-based prediction  

> A minimalistic and intuitive UI built with **Streamlit**, tailored for farmers and researchers

### 🧠 ViT Inference Example

- **Input:** Leaf image (Citrus tree)  
- **Prediction:** Orange_Huanglongbing (Citrus greening) – **99.06% confidence**  
- **Follow-up:**  
  > _"What diseases can affect this?"_  
  - Gemini responds contextually, explaining secondary infections and symptoms

---

## 🔗 GitHub Repository

📁 [GitHub Repo](https://github.com/sujanshanmugaraj/AI-POWERED-CROP-DISEASE-PREDICTION)

---

## 🎥 Output Demo Video

📹 [Watch Output Video](https://drive.google.com/file/d/1ZEFja6Dy00ZySzIR5okJeSpYQ0QSYZbs/view?usp=sharing)

---

Let me know if you want this converted into a `README.md` file directly or added to your existing repo!
