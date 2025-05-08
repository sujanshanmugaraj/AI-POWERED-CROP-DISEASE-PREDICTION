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
 
  - ![image](https://github.com/user-attachments/assets/79ad0cb3-e5a1-4ceb-a5d6-ac16232358d6)


The above image showcases the homepage interface of the Crop Health Assistant, a
user-friendly web application designed to facilitate easy access to crop disease
diagnosis. The interface allows users to select from three input methods—Text, Voice,
and Image—to interact with the system, making it adaptable to various user preferences
and environmental situations.

● The Text option enables users to manually enter crop-related queries.

● The Voice option supports verbal interactions, useful in hands-free or field
conditions.

● The Image option allows users to upload leaf images for visual disease diagnosis
using the Vision Transformer (ViT) model.

This clean, minimalistic design ensures that farmers, agricultural officers, and
researchers can efficiently navigate the platform and access AI-driven crop health
insights. The input box invites users to ask a question about crop health, which then
triggers the multimodal model pipeline for prediction and recommendation.

![image](https://github.com/user-attachments/assets/4926ab5d-bd8b-498e-91fc-b0ed7642a047)
![image](https://github.com/user-attachments/assets/71d569b4-b9ae-417d-9b18-3e63e8012371)
![image](https://github.com/user-attachments/assets/46e9aa39-491b-433b-96aa-2a5762d3f906)

Upon uploading a leaf image, the system utilizes a Vision Transformer (ViT) model to
analyze visual features and predict the crop disease. In this instance, the model
identified the disease as:

ViT Prediction: Orange_Huanglongbing (Citrus greening) with 99.06%
confidence

Following the prediction, the user posed a natural-language follow-up question:
"What diseases can affect this?"

The system responds with contextual information, explaining that:

● Citrus greening (Huanglongbing or HLB) is the primary disease.

● While not directly caused by other diseases, it weakens the tree.

● This makes the plant more prone to secondary infections from fungi and
pathogens.

● These secondary infections can accelerate the decline of the tree’s health.

![image](https://github.com/user-attachments/assets/76413d03-da3c-49ea-b358-8f40e0b07f2c)

Upon uploading a leaf image (detected as affected by Citrus
Greening/Huanglongbing (HLB)), the user interacts through sequential prompts
using natural language. The system preserves the entire prompt history, creating a
context-aware conversational experience.




---

## 🔗 GitHub Repository

📁 [GitHub Repo](https://github.com/sujanshanmugaraj/AI-POWERED-CROP-DISEASE-PREDICTION)

---

## 🎥 Output Demo Video

📹 [Watch Output Video](https://drive.google.com/file/d/1ZEFja6Dy00ZySzIR5okJeSpYQ0QSYZbs/view?usp=sharing)


---

## 💡 Learning Outcomes

Throughout this project, we gained hands-on experience and a deeper understanding in the following areas:

- **Multimodal Deep Learning**: Learned to integrate image and tabular data for improved model accuracy using cross-attention mechanisms.
- **Vision Transformers (ViT)**: Explored transformer-based architectures for computer vision tasks, replacing traditional CNNs for global feature extraction.
- **Data Preprocessing**: Practiced image augmentation, feature scaling, and label encoding techniques for both image and numerical data.
- **Model Fusion Techniques**: Implemented fusion of multiple modalities using PyTorch’s `nn.MultiheadAttention` for cross-modal learning.
- **Performance Evaluation**: Analyzed models using accuracy, confusion matrix, and classification reports to assess real-world effectiveness.
- **Streamlit App Development**: Built an intuitive web interface that supports text, voice, and image inputs for real-time disease prediction.
- **Voice & AI Integration**: Integrated Google Gemini via LangChain for interactive question-answering and used `speech_recognition` and `pyttsx3` for voice support.
- **Collaborative Development**: Used Git and GitHub for version control and team collaboration throughout the development process.

