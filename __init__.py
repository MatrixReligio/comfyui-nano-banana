"""
ComfyUI Nano Banana - Google Gemini 3 Pro Image Generation Nodes
"""

from .nano_banana_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Import API routes to register them
from . import api_routes

# Path to web directory for frontend extensions
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
