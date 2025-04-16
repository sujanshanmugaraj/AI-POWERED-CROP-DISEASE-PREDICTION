import torch
import os
from torchvision import models

# Define the model architecture (e.g., ViT or another model architecture)
# For example, if you're using Vision Transformer (ViT)
from transformers import ViTForImageClassification  # Ensure this is the correct model import

# Create the model (use the same architecture used for training)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Absolute path to the model
model_path = os.path.join(os.getcwd(), 'vit_multimodal_best.pth')

# Load the model weights (state_dict) into the model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

# Set the model to evaluation mode
model.eval()

print("Model loaded and ready for evaluation.")
