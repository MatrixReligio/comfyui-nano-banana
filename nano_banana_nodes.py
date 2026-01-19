"""
ComfyUI Custom Nodes for Google Gemini 3 Pro Image (Nano Banana Pro)
Supports text-to-image and image-to-image generation
"""

import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")

# Default image generation models
DEFAULT_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-preview-native-audio-dialog",
    "gemini-2.5-flash-image-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def load_api_key():
    """Load API key from .env file or environment variable"""
    api_key = None

    # First, try to load from .env file in the node directory
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key == 'GEMINI_API_KEY':
                            api_key = value
                            logger.info(f"Loaded GEMINI_API_KEY from {ENV_FILE}")
                            break
        except Exception as e:
            logger.warning(f"Failed to read .env file: {e}")

    # Fallback to environment variable
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            logger.info("Using GEMINI_API_KEY from environment variable")

    return api_key


# Load API key at module initialization
_CACHED_API_KEY = load_api_key()


def get_api_key():
    """Get the cached API key"""
    global _CACHED_API_KEY
    if not _CACHED_API_KEY:
        _CACHED_API_KEY = load_api_key()
    return _CACHED_API_KEY


def genai_image_to_pil(genai_image) -> Image.Image:
    """Convert Google genai Image object to PIL Image"""
    if hasattr(genai_image, 'image_bytes') and genai_image.image_bytes:
        # Google genai Image has image_bytes attribute
        return Image.open(BytesIO(genai_image.image_bytes))
    elif hasattr(genai_image, '_pil_image'):
        # Some versions may have internal PIL image
        return genai_image._pil_image
    elif hasattr(genai_image, 'data'):
        # Fallback for other data formats
        return Image.open(BytesIO(genai_image.data))
    else:
        raise ValueError(f"Cannot convert genai Image to PIL: {type(genai_image)}")


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert ComfyUI tensor (B, H, W, C) to list of PIL Images"""
    if tensor is None:
        return []

    # Ensure tensor is on CPU and convert to numpy
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    images = []
    # Handle batch dimension
    if len(tensor.shape) == 4:
        for i in range(tensor.shape[0]):
            img_np = tensor[i].numpy()
            # Convert from 0-1 float to 0-255 uint8
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
    elif len(tensor.shape) == 3:
        img_np = tensor.numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))

    return images


def pil_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert list of PIL Images to ComfyUI tensor (B, H, W, C)"""
    if not images:
        # Return a placeholder image
        placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
        return torch.from_numpy(placeholder.astype(np.float32) / 255.0).unsqueeze(0)

    tensors = []
    for img in images:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))

    # Stack into batch tensor
    return torch.stack(tensors, dim=0)


def create_placeholder_image(width=512, height=512, message="No image generated"):
    """Create a placeholder image with text"""
    img = Image.new('RGB', (width, height), color=(64, 64, 64))
    return pil_to_tensor([img])


class NanoBananaTextToImage:
    """
    Text to Image generation using Google Gemini 3 Pro Image (Nano Banana Pro)
    API key is loaded from .env file in the node directory
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape with mountains and a lake at sunset"
                }),
                "model_name": (DEFAULT_MODELS, {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": ([
                    "1:1",
                    "2:3",
                    "3:2",
                    "3:4",
                    "4:3",
                    "4:5",
                    "5:4",
                    "9:16",
                    "16:9",
                    "21:9"
                ], {"default": "1:1"}),
                "image_size": ([
                    "1K",
                    "2K",
                    "4K"
                ], {"default": "1K"}),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"

    def generate(self, prompt: str, model_name: str, aspect_ratio: str, image_size: str, seed: int = 0):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            error_msg = "Please install google-genai: pip install google-genai"
            logger.error(error_msg)
            return (create_placeholder_image(), error_msg)

        # Get API key from .env or environment
        api_key = get_api_key()

        if not api_key:
            error_msg = f"No API key found. Please create a .env file at:\n{ENV_FILE}\n\nWith content:\nGEMINI_API_KEY=your_api_key_here"
            logger.error(error_msg)
            return (create_placeholder_image(), error_msg)

        try:
            # Create client
            client = genai.Client(api_key=api_key)

            # Build generation config
            config_args = {
                "response_modalities": ["Text", "Image"],
                "image_config": types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            }

            # Add seed if provided
            if seed > 0:
                config_args["seed"] = seed

            generation_config = types.GenerateContentConfig(**config_args)

            logger.info(f"Generating image with model: {model_name}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Config: aspect_ratio={aspect_ratio}, image_size={image_size}, seed={seed}")

            # Generate content
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config
            )

            # Process response
            generated_images = []
            response_text = ""

            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text + "\n"
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Decode image data
                    try:
                        genai_img = part.as_image()
                        # Convert Google genai Image to PIL Image
                        pil_img = genai_image_to_pil(genai_img)
                        generated_images.append(pil_img)
                        logger.info(f"Generated image: {pil_img.size}")
                    except Exception as e:
                        logger.warning(f"Failed to decode image: {e}")

            if generated_images:
                return (pil_to_tensor(generated_images), response_text.strip())
            else:
                return (create_placeholder_image(), response_text.strip() or "No image was generated")

        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return (create_placeholder_image(), error_msg)


class NanoBananaImageToImage:
    """
    Image to Image generation/editing using Google Gemini 3 Pro Image (Nano Banana Pro)
    API key is loaded from .env file in the node directory
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform this image into a watercolor painting style"
                }),
                "model_name": (DEFAULT_MODELS, {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": ([
                    "1:1",
                    "2:3",
                    "3:2",
                    "3:4",
                    "4:3",
                    "4:5",
                    "5:4",
                    "9:16",
                    "16:9",
                    "21:9"
                ], {"default": "1:1"}),
                "image_size": ([
                    "1K",
                    "2K",
                    "4K"
                ], {"default": "1K"}),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
                "max_input_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 14
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate"
    CATEGORY = "Nano Banana"

    def generate(self, image: torch.Tensor, prompt: str, model_name: str,
                 aspect_ratio: str, image_size: str, seed: int = 0, max_input_images: int = 1):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            error_msg = "Please install google-genai: pip install google-genai"
            logger.error(error_msg)
            return (create_placeholder_image(), error_msg)

        # Get API key from .env or environment
        api_key = get_api_key()

        if not api_key:
            error_msg = f"No API key found. Please create a .env file at:\n{ENV_FILE}\n\nWith content:\nGEMINI_API_KEY=your_api_key_here"
            logger.error(error_msg)
            return (create_placeholder_image(), error_msg)

        try:
            # Convert input tensor to PIL images
            input_images = tensor_to_pil(image)

            if not input_images:
                error_msg = "No input image provided"
                logger.error(error_msg)
                return (create_placeholder_image(), error_msg)

            # Limit number of input images
            input_images = input_images[:max_input_images]
            logger.info(f"Using {len(input_images)} input image(s)")

            # Create client
            client = genai.Client(api_key=api_key)

            # Build generation config
            config_args = {
                "response_modalities": ["Text", "Image"],
                "image_config": types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            }

            # Add seed if provided
            if seed > 0:
                config_args["seed"] = seed

            generation_config = types.GenerateContentConfig(**config_args)

            logger.info(f"Generating image with model: {model_name}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Config: aspect_ratio={aspect_ratio}, image_size={image_size}, seed={seed}")

            # Build content with images and prompt
            # Gemini expects: [prompt, image1, image2, ...]
            contents = [prompt] + input_images

            # Generate content
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config
            )

            # Process response
            generated_images = []
            response_text = ""

            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text + "\n"
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Decode image data
                    try:
                        genai_img = part.as_image()
                        # Convert Google genai Image to PIL Image
                        pil_img = genai_image_to_pil(genai_img)
                        generated_images.append(pil_img)
                        logger.info(f"Generated image: {pil_img.size}")
                    except Exception as e:
                        logger.warning(f"Failed to decode image: {e}")

            if generated_images:
                return (pil_to_tensor(generated_images), response_text.strip())
            else:
                return (create_placeholder_image(), response_text.strip() or "No image was generated")

        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return (create_placeholder_image(), error_msg)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "NanoBananaTextToImage": NanoBananaTextToImage,
    "NanoBananaImageToImage": NanoBananaImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaTextToImage": "Nano Banana Text to Image",
    "NanoBananaImageToImage": "Nano Banana Image to Image",
}
