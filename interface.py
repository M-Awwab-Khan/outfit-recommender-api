import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import certifi
import ssl
from torchvision import transforms
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel

ssl_context = ssl.create_default_context(cafile=certifi.where())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def download_image(image_url):
    try:
        response = requests.get(image_url, verify=certifi.where(), timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        print(f"Failed to load image: {e}")
        return None


def extract_clip_features(image_url):
    image = download_image(image_url)
    if image is None:
        return None

    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    return torch.nn.functional.normalize(features, p=2, dim=1).cpu().numpy().flatten()


def get_dominant_color(image_url, k=3):
    img = download_image(image_url)
    if img is None:
        return None

    img = np.array(img).reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(img)
    return np.mean(kmeans.cluster_centers_, axis=0)


def color_distance(color1, color2):
    if color1 is None or color2 is None:
        return 255
    return np.linalg.norm(color1 - color2)


def color_harmony_score(shirt_color, pant_color):
    if shirt_color is None or pant_color is None:
        return 0.5

    distance = color_distance(shirt_color, pant_color)

    if distance < 60:
        return 1.0
    elif 100 < distance < 160:
        return 0.8
    elif 160 < distance < 220:
        return 0.6
    else:
        return 0.2


def find_best_match(shirt_urls, pant_urls, min_similarity=0.7):
    results = []
    shirt_features = {url: extract_clip_features(url) for url in shirt_urls}
    pant_features = {url: extract_clip_features(url) for url in pant_urls}
    shirt_colors = {url: get_dominant_color(url) for url in shirt_urls}
    pant_colors = {url: get_dominant_color(url) for url in pant_urls}

    for shirt_url, shirt_vec in shirt_features.items():
        if shirt_vec is None:
            continue

        best_match, best_score = None, -1
        for pant_url, pant_vec in pant_features.items():
            if pant_vec is None:
                continue

            similarity = np.dot(shirt_vec, pant_vec) / (
                np.linalg.norm(shirt_vec) * np.linalg.norm(pant_vec)
            )
            harmony_score = color_harmony_score(
                shirt_colors[shirt_url], pant_colors[pant_url]
            )
            final_score = (0.7 * similarity) + (0.3 * harmony_score)

            if final_score > best_score and final_score > min_similarity:
                best_score, best_match = final_score, pant_url

        if best_match:
            results.append(
                {
                    "shirt": shirt_url,
                    "best_matching_pant": best_match,
                    "similarity_score": f"{best_score:.2f}",
                }
            )

    return results
