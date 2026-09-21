import base64
import os
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def preprocess_and_enhance_image(image_input) -> str:
    """
    Auto-enhances blurry, low-contrast, or low-resolution images:
    - Auto-contrast correction
    - Sharpness enhancement (deblurring)
    - Contrast optimization for faint text and diagrams
    - High-quality bicubic/Lanczos scaling if low-res
    Returns a base64 encoded string of the enhanced image.
    """
    if isinstance(image_input, str):
        if os.path.isfile(image_input):
            img = Image.open(image_input).convert('RGB')
        else:
            response = requests.get(image_input, stream=True)
            img = Image.open(BytesIO(response.content)).convert('RGB')
    elif hasattr(image_input, 'read'):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    # 1. Upscale if image is too small for optical clarity
    min_dim = min(img.size)
    if min_dim < 600:
        scale_factor = 600 / min_dim
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # 2. Auto-contrast to fix washed out or dark lighting
    try:
        img = ImageOps.autocontrast(img, cutoff=0.5)
    except Exception:
        pass

    # 3. Enhance Sharpness (De-blurring filter)
    enhancer_sharpness = ImageEnhance.Sharpness(img)
    img = enhancer_sharpness.enhance(1.8)

    # 4. Enhance Contrast (makes text and line drawings stand out)
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(1.25)

    # 5. Export to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def answer_visual_question(image_path_or_url: str, question: str = "") -> str:
    """
    High-accuracy Visual Question Answering using Ollama's LLaVA (Vision-Language Model)
    with automated image enhancement for handling blurry or low-quality captures.
    """
    try:
        # Preprocess and de-blur image before sending to vision model
        b64_image = preprocess_and_enhance_image(image_path_or_url)
        
        if not question or question.strip() == "":
            prompt = (
                "You are an expert computer vision and OCR specialist. Carefully examine this image. "
                "Even if parts of the image appear slightly blurry, out-of-focus, or noisy, analyze all visible elements, "
                "reconstruct any readable text or labels, and provide a thorough, accurate description of everything shown."
            )
        else:
            prompt = (
                f"You are an expert vision and OCR analyst. The user has provided an image that might be slightly blurry or degraded.\n"
                f"Carefully analyze all shapes, diagrams, and text visible in the image to answer the user's question with extreme precision.\n\n"
                f"User Question: {question.strip()}\n"
                f"Answer:"
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