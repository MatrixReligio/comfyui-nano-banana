# ComfyUI Nano Banana

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that enable image generation using Google's Gemini 3 Pro Image API (Nano Banana Pro).

[中文文档 (Chinese Documentation)](README_CN.md)

## Features

- **Text to Image**: Generate images from text descriptions
- **Image to Image**: Transform/edit existing images with text prompts
- **Dynamic Model Selection**: Fetch and select from available Gemini models
- **API Key Validation**: Built-in button to verify your API key
- **Multiple Aspect Ratios**: Support for 10 different aspect ratios
- **Flexible Resolution**: 1K, 2K, and 4K output options
- **Reproducible Results**: Optional seed parameter for consistent outputs

## Installation

### Method 1: Manual Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ~/Documents/ComfyUI/custom_nodes/
   # Or for ComfyUI portable version:
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/matrixreligion/comfyui-nano-banana.git
   ```

3. Install dependencies:
   ```bash
   cd comfyui-nano-banana
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Method 2: ComfyUI Manager

Search for "Nano Banana" in ComfyUI Manager and install directly.

## Configuration

### API Key Setup

1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

2. Create a `.env` file in the node directory:
   ```bash
   cd ~/Documents/ComfyUI/custom_nodes/comfyui-nano-banana/
   ```

3. Add your API key to the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

> **Security Note**: The `.env` file is included in `.gitignore` to prevent accidental exposure of your API key.

## Nodes

### Nano Banana Text to Image

Generate images from text descriptions.

#### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | String (multiline) | - | Text description of the image you want to generate. Be detailed for best results. |
| `model_name` | Dropdown | `gemini-3-pro-image-preview` | The Gemini model to use for generation |
| `aspect_ratio` | Dropdown | `1:1` | Output image aspect ratio |
| `image_size` | Dropdown | `1K` | Output image resolution |
| `seed` | Integer (optional) | `0` | Random seed. 0 = random, fixed value = reproducible results |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image tensor |
| `response_text` | STRING | API response text (if any) |

---

### Nano Banana Image to Image

Transform or edit existing images using text prompts.

#### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | IMAGE | (required) | Input image from another node (e.g., LoadImage) |
| `prompt` | String (multiline) | - | Instructions for how to transform the image |
| `model_name` | Dropdown | `gemini-3-pro-image-preview` | The Gemini model to use |
| `aspect_ratio` | Dropdown | `1:1` | Output image aspect ratio |
| `image_size` | Dropdown | `1K` | Output image resolution |
| `seed` | Integer (optional) | `0` | Random seed for reproducibility |
| `max_input_images` | Integer (optional) | `1` | Maximum number of input images to process (1-14) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Transformed image tensor |
| `response_text` | STRING | API response text (if any) |

---

### UI Buttons

Both nodes include two utility buttons:

| Button | Function |
|--------|----------|
| **Verify API Key** | Validates that your API key in `.env` is working correctly |
| **Update Models** | Fetches the latest available models from Google and updates the dropdown |

## Parameter Options

### Aspect Ratios

| Value | Use Case |
|-------|----------|
| `1:1` | Square - avatars, icons, social media |
| `2:3` | Portrait - vertical photos |
| `3:2` | Landscape - horizontal photos |
| `3:4` | Portrait - close to phone screen ratio |
| `4:3` | Landscape - traditional monitor ratio |
| `4:5` | Portrait - Instagram recommended |
| `5:4` | Landscape - slightly wide |
| `9:16` | Portrait - phone fullscreen, video covers |
| `16:9` | Landscape - video, widescreen displays |
| `21:9` | Ultra-wide - cinematic format |

### Image Sizes

| Value | Approximate Resolution | Use Case |
|-------|----------------------|----------|
| `1K` | ~1024px | Quick preview, web images |
| `2K` | ~2048px | High quality display, printing |
| `4K` | ~4096px | Ultra HD, large format printing |

## Example Workflows

Example workflow files are included in the `workflows/` directory:

- `text_to_image_workflow.json` - Basic text-to-image generation
- `image_to_image_workflow.json` - Image transformation example

To use a workflow:
1. Open ComfyUI
2. Drag and drop the JSON file into the ComfyUI interface
3. Or use Menu → Load → select the workflow file

## Troubleshooting

### "No API key found" Error

Make sure you have created the `.env` file with your API key:
```
GEMINI_API_KEY=your_actual_api_key
```

### Black/Placeholder Image Output

1. Check the `response_text` output for error messages
2. Verify your API key using the "Verify API Key" button
3. Ensure you have sufficient API quota

### Model Dropdown Shows Wrong Value

1. Delete the existing node
2. Restart ComfyUI
3. Add the node fresh from: Right-click → Add Node → Nano Banana

### Buttons Don't Show Response

After clicking a button, an alert dialog will appear with the result. Make sure pop-ups are not blocked.

## Dependencies

- `google-genai` - Google's Generative AI Python SDK
- `Pillow` - Image processing
- `torch` - PyTorch for tensor operations
- `numpy` - Numerical operations

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The powerful node-based UI
- [Google Gemini](https://deepmind.google/technologies/gemini/) - The AI model powering image generation
