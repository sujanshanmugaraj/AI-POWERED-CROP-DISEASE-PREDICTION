import gdown
import os

# Google Drive File ID
file_id = "1E3BBh_EH9UeZXR_xYhwIGl3nH_ER2cQq"
# Direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Absolute path to where you want to save the file
output = r"C:\Users\Sujan.S\OneDrive\Documents\GitHub\AI-POWERED-CROP-DISEASE-PREDICTION\DL Package\vit_multimodal_best.pth"

# Ensure the folder exists, if not, create it
os.makedirs(os.path.dirname(output), exist_ok=True)

# Download the file
gdown.download(url, output, quiet=False)
