"""
API routes for Nano Banana nodes
Provides endpoints for API key validation and model listing
"""

import os
import logging
from aiohttp import web

logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")


def load_api_key_from_env():
    """Load API key from .env file"""
    api_key = None
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
                            break
        except Exception as e:
            logger.warning(f"Failed to read .env file: {e}")

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    return api_key


def check_gemini_api_key(api_key: str) -> tuple[bool, str]:
    """
    Validate a Gemini API key by making a simple API call
    Returns (is_valid, message)
    """
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Try to list models to verify the key
        models = list(client.models.list())

        if models:
            model_count = len(models)
            return True, f"API key is valid. Found {model_count} models."
        else:
            return True, "API key is valid but no models found."

    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "api key" in error_str:
            return False, "Invalid API key. Please check your key."
        elif "quota" in error_str or "exceeded" in error_str:
            return False, "API quota exceeded."
        else:
            return False, f"Error validating API key: {str(e)}"


def get_image_generation_models(api_key: str) -> list[str]:
    """
    Get list of available image generation models
    Returns list of model names
    """
    # Known Gemini image generation models
    known_image_models = [
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-preview-native-audio-dialog",
        "gemini-2.5-flash-image-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Get all models from API
        all_models = []
        for model in client.models.list():
            model_name = model.name
            # Remove "models/" prefix if present
            if model_name.startswith("models/"):
                model_name = model_name[7:]
            all_models.append(model_name)

        # Filter for image-capable models
        # Include known image models and any model with "image" in the name
        image_models = []
        for model in all_models:
            if model in known_image_models:
                image_models.append(model)
            elif "image" in model.lower() and model not in image_models:
                image_models.append(model)

        # Add known models that might not be in the list
        for known in known_image_models:
            if known not in image_models:
                # Check if this model exists
                for m in all_models:
                    if known in m or m in known:
                        if m not in image_models:
                            image_models.append(m)

        # Sort and return, prioritizing gemini-3-pro-image-preview
        if "gemini-3-pro-image-preview" in image_models:
            image_models.remove("gemini-3-pro-image-preview")
            image_models.insert(0, "gemini-3-pro-image-preview")

        return image_models if image_models else known_image_models

    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return known models as fallback
        return known_image_models


# Register routes with ComfyUI server
try:
    from server import PromptServer

    # Check if routes are already registered
    routes_registered = getattr(PromptServer.instance, '_nano_banana_routes_registered', False)

    if not routes_registered:
        @PromptServer.instance.routes.post("/nano_banana/check_api_key")
        async def check_api_key_route(request):
            """Check if the Gemini API key is valid"""
            try:
                # Get API key from .env file
                api_key = load_api_key_from_env()

                if not api_key:
                    return web.json_response({
                        "status": "error",
                        "message": f"No API key found. Please add GEMINI_API_KEY to:\n{ENV_FILE}"
                    })

                is_valid, message = check_gemini_api_key(api_key)

                return web.json_response({
                    "status": "success" if is_valid else "error",
                    "message": message
                })

            except Exception as e:
                logger.error(f"Error checking API key: {e}")
                return web.json_response({
                    "status": "error",
                    "message": f"Error: {str(e)}"
                })

        @PromptServer.instance.routes.get("/nano_banana/get_models")
        async def get_models_route(request):
            """Get available image generation models"""
            try:
                # Get API key from .env file
                api_key = load_api_key_from_env()

                if not api_key:
                    # Return default models if no API key
                    return web.json_response({
                        "status": "error",
                        "message": "No API key configured",
                        "models": ["gemini-3-pro-image-preview"]
                    })

                models = get_image_generation_models(api_key)

                return web.json_response({
                    "status": "success",
                    "models": models
                })

            except Exception as e:
                logger.error(f"Error getting models: {e}")
                return web.json_response({
                    "status": "error",
                    "message": str(e),
                    "models": ["gemini-3-pro-image-preview"]
                })

        # Mark routes as registered
        PromptServer.instance._nano_banana_routes_registered = True
        logger.info("Nano Banana API routes registered successfully")

except ImportError:
    logger.warning("PromptServer not available - API routes not registered")
except Exception as e:
    logger.error(f"Error registering API routes: {e}")
