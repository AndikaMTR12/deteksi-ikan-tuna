import os
import gdown

# URL Google Drive - Ganti dengan ID file kamu
MODEL_URL = "https://drive.google.com/uc?id=1NqsaKb6WpvzbTdrZPK3lELgkpH9pm_Pg"
ANNOTATION_URL = "https://drive.google.com/uc?id=1NVF-CMGa8FfZUYYFITUusSC8JgToOnIO"

def download_files():
    if not os.path.exists("model_final.pth"):
        print("Downloading model_final.pth...")
        gdown.download(MODEL_URL, "model_final.pth", quiet=False)
    
    if not os.path.exists("annotations_coco_resized.json"):
        print("Downloading annotations_coco_resized.json...")
        gdown.download(ANNOTATION_URL, "annotations_coco_resized.json", quiet=False)

# Jalankan otomatis saat diimport
download_files()
