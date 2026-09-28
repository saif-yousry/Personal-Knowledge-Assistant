# download_model.py
from sentence_transformers import SentenceTransformer

# Replace with the exact model name configured in your settings/config (e.g., 'all-mpnet-base-v2')
MODEL_NAME = "all-mpnet-base-v2" 

print(f"Downloading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model downloaded and cached successfully!")