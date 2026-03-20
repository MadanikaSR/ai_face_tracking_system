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

if __name__ == "__main__":
    main()
