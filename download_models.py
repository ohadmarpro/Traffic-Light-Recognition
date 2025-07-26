import gdown
import os

# Create a models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define the files and their Google Drive IDs
files = {
    "yolo_all_classes.pt": "1JFyOcf-URedVqBN75cJy-uJvGmTDCD27",
    "yolo_traffic_only.pt": "1qMRlc7DgiwvgdTSmahEFlCb2MmKha3Ti",
    "cnn_classifier.pth": "1UmoKKbhXeZbOfl3wLezBZYP-dC5opyml"
}

# Download each file into the models directory
for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join("models", filename)
    print(f"⬇ Downloading {filename}...")
    gdown.download(url, output_path, quiet=False)

print("✅ All models downloaded to the 'models/' directory.")
