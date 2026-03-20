import os
import requests

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return
        
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def main():
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # URL for Buffalo_L recognition model (w600k_r50.onnx) from HuggingFace
    model_url = "https://huggingface.co/fofr/comfyui/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx"
    download_file(model_url, os.path.join(model_dir, "w600k_r50.onnx"))

    # URL for yolov8n-face model from a public reliable source (adetailer)
    face_model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt"
    download_file(face_model_url, os.path.join(model_dir, "yolov8n-face.pt"))

    # URL for SCRFD face detector (InsightFace)
    scrfd_url = "https://huggingface.co/ykk648/face_lib/resolve/main/face_detect/scrfd_onnx/scrfd_500m_bnkps.onnx"
    download_file(scrfd_url, os.path.join(model_dir, "scrfd_500m_bnkps.onnx"))

if __name__ == "__main__":
    main()
