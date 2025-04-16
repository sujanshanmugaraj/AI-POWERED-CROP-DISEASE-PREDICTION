
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import speech_recognition as sr
import pyttsx3
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from langchain_google_genai import ChatGoogleGenerativeAI

# --- ViT Model Class ---
class MultimodalViT(nn.Module):
    def __init__(self, num_numerical_features, num_classes, dropout_rate=0.3):
        super(MultimodalViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()
        self.num_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate)
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, numerical_features):
        img_features = self.vit(image).unsqueeze(0)
        num_features = self.num_encoder(numerical_features).unsqueeze(0)
        num_features = nn.functional.pad(num_features, (0, 768 - 128))
        batch_size = image.size(0)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        combined = torch.cat((cls_tokens, img_features, num_features), dim=0)
        attn_output, _ = self.cross_attention(cls_tokens, combined, combined)
        output = self.fc(attn_output.squeeze(0))
        return output

# --- Init Model ---
num_numerical_features = 7
num_classes = 22
model = MultimodalViT(num_numerical_features, num_classes)
model.load_state_dict(torch.load("vit_multimodal_best.pth", map_location='cpu'))
model.eval()

# --- Class Mapping ---
class_mapping = {
    0: 'Apple_Apple_scab', 1: 'Apple_Black_rot', 2: 'Apple_Cedar_apple_rust', 3: 'Apple_healthy',
    4: 'Blueberry__healthy', 5: 'Cherry(including_sour)Powdery_mildew', 6: 'Cherry(including_sour)_healthy',
    7: 'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn(maize)Common_rust',
    9: 'Corn_(maize)Northern_Leaf_Blight', 10: 'Corn(maize)healthy', 11: 'Grape__Black_rot',
    12: 'Grape_Esca(Black_Measles)', 13: 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)', 14: 'Grape__healthy',
    15: 'Orange_Haunglongbing(Citrus_greening)', 16: 'PeachBacterial_spot', 17: 'Peach_healthy',
    18: 'Pepper,bell_Bacterial_spot', 19: 'Pepper,bell_healthy', 20: 'Potato_Early_blight', 21: 'Potato_healthy'
}

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Prediction Function ---
@torch.no_grad()
def predict_leaf_class(image, numerical_features=None):
    image_tensor = transform(image).unsqueeze(0)
    if numerical_features is None:
        numerical_features = torch.zeros((1, num_numerical_features))
    output = model(image_tensor, numerical_features)
    prob = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(prob, 1)
    class_label = class_mapping[predicted.item()]
    return class_label, confidence.item()

# --- TTS ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- Gemini Setup ---
api_key = "AIzaSyAu-weudZlsrpiyCeqD8cbKI8OPTAMWKWs"
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    google_api_key=api_key
)

def chat_with_ai(user_input, history=""):
    prompt = (
        "You are a crop and plant disease assistant. "
        "Only answer questions related to crop health, leaf diseases, and remedies. "
        "If the question is unrelated, say 'Sorry, I can only help with crop-related queries.'\n"
        f"Previous conversation context: {history}\n"
        f"User: {user_input}\n"
    )
    try:
        response = chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {e}"

# --- Session State ---
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'vit_result' not in st.session_state:
    st.session_state.vit_result = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# --- UI Setup ---
st.set_page_config(page_title="Crop Health Assistant", page_icon="🌿")
st.title("🌿 Crop Health Assistant")

mode = st.radio("Choose input method:", ["Text", "Voice", "Image"], horizontal=True)

# --- CHATGPT-LIKE HISTORY DISPLAY ---
for entry in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["response"])

# --- Text Mode ChatGPT-style ---
if mode == "Text":
    with st.chat_message("user"):
        question = st.text_input("Ask a question about crop health:", key="text_input")
    if question:
        with st.spinner("Thinking..."):
            history = "\n".join([f"Q: {e['question']}\nA: {e['response']}" for e in st.session_state.conversation_history])
            response = chat_with_ai(question, history)
            st.session_state.conversation_history.append({"question": question, "response": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            if st.button("🔊 Speak", key="speak_button"):
                speak(response)

# --- Voice Mode ---
elif mode == "Voice":
    if st.button("🎤 Start Listening"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Please speak.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=10)
                question = recognizer.recognize_google(audio)
                with st.chat_message("user"):
                    st.markdown(question)
                history = "\n".join([f"Q: {e['question']}\nA: {e['response']}" for e in st.session_state.conversation_history])
                response = chat_with_ai(question, history)
                st.session_state.conversation_history.append({"question": question, "response": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                speak(response)
            except Exception as e:
                st.error(f"Error: {e}")

# --- Image Mode ---
elif mode == "Image":
    uploaded_image = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
        with st.spinner("Analyzing with ViT..."):
            vit_result, confidence = predict_leaf_class(image)
            st.session_state.vit_result = vit_result
            st.session_state.confidence = confidence
            st.success(f"🧠 ViT Prediction: *{vit_result}* ({confidence*100:.2f}% confidence)")

    with st.chat_message("user"):
        image_question = st.text_input("What would you like to know about this leaf?", key="image_question")
    if image_question:
        if st.session_state.uploaded_image:
            history = "\n".join([f"Q: {e['question']}\nA: {e['response']}" for e in st.session_state.conversation_history])
            response = chat_with_ai(image_question, history + f"\nViT Result: {st.session_state.vit_result}")
            st.session_state.conversation_history.append({"question": image_question, "response": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.warning("Please upload an image first.")
