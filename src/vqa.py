import base64
import os
import requests
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_path_or_url: str) -> str:
    """Read image file or download from URL and return base64 string."""
    if os.path.isfile(image_path_or_url):
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        response = requests.get(image_path_or_url, stream=True)
        return base64.b64encode(response.content).decode('utf-8')

def answer_visual_question(image_path_or_url: str, question: str = "") -> str:
    """
    High-accuracy Visual Question Answering using Ollama's LLaVA (Vision-Language Model).
    Accurately understands diagrams, documents, photos, and answers in full detail.
    """
    try:
        b64_image = encode_image_to_base64(image_path_or_url)
        
        if not question or question.strip() == "":
            prompt = (
                "Describe this image thoroughly and accurately. Identify all key objects, text, "
                "diagrams, or relevant details visible."
            )
        else:
            prompt = (
                f"Examine this image with extreme accuracy and answer the following question in detail:\n"
                f"Question: {question.strip()}"
            )
            
        payload = {
            "model": "llava",
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {
                "temperature": 0.2  # Low temperature for factual precision
            }
        }
        
        res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=90)
        if res.status_code == 200:
            return res.json().get("response", "Could not generate response from image.").strip()
        else:
            return f"Error from Vision model: HTTP {res.status_code} - {res.text}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Ollama service is not running. Please make sure Ollama is open and running."
    except Exception as e:
        return f"Error analyzing image: {str(e)}"