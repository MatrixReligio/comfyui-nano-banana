# ComfyUI Nano Banana

使用 Google Gemini 3 Pro Image API (Nano Banana Pro) 的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 自定义节点。

[English Documentation](README.md)

## 功能特点

- **文生图 (Text to Image)**: 根据文字描述生成图像
- **图生图 (Image to Image)**: 使用文字提示转换/编辑现有图像
- **动态模型选择**: 获取并选择可用的 Gemini 模型
- **API Key 验证**: 内置按钮验证 API key 有效性
- **多种宽高比**: 支持 10 种不同的宽高比
- **灵活分辨率**: 1K、2K、4K 输出选项
- **可复现结果**: 可选的种子参数确保结果一致性

## 安装方法

### 方法一：手动安装

1. 进入 ComfyUI 自定义节点目录：
   ```bash
   cd ~/Documents/ComfyUI/custom_nodes/
   # 或者 ComfyUI 便携版：
   cd ComfyUI/custom_nodes/
   ```

2. 克隆仓库：
   ```bash
   git clone https://github.com/matrixreligion/comfyui-nano-banana.git
   ```

3. 安装依赖：
   ```bash
   cd comfyui-nano-banana
   pip install -r requirements.txt
   ```

4. 重启 ComfyUI

### 方法二：ComfyUI Manager

在 ComfyUI Manager 中搜索 "Nano Banana" 直接安装。

## 配置

### API Key 设置

1. 从 [Google AI Studio](https://aistudio.google.com/apikey) 获取 Gemini API key

2. 在节点目录创建 `.env` 文件：
   ```bash
   cd ~/Documents/ComfyUI/custom_nodes/comfyui-nano-banana/
   ```

3. 在 `.env` 文件中添加 API key：
   ```
   GEMINI_API_KEY=你的API密钥
   ```

> **安全提示**: `.env` 文件已包含在 `.gitignore` 中，防止意外泄露 API key。

## 节点说明

### Nano Banana Text to Image (文生图)

根据文字描述生成图像。

#### 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | 多行文本 | - | 描述想要生成的图像内容，越详细效果越好 |
| `model_name` | 下拉选择 | `gemini-3-pro-image-preview` | 用于生成的 Gemini 模型 |
| `aspect_ratio` | 下拉选择 | `1:1` | 输出图像宽高比 |
| `image_size` | 下拉选择 | `1K` | 输出图像分辨率 |
| `seed` | 整数 (可选) | `0` | 随机种子。0=随机，固定值=可复现结果 |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `image` | IMAGE | 生成的图像张量 |
| `response_text` | STRING | API 响应文本（如有） |

---

### Nano Banana Image to Image (图生图)

使用文字提示转换或编辑现有图像。

#### 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | IMAGE | (必需) | 来自其他节点的输入图像（如 LoadImage） |
| `prompt` | 多行文本 | - | 描述如何转换图像的指令 |
| `model_name` | 下拉选择 | `gemini-3-pro-image-preview` | 使用的 Gemini 模型 |
| `aspect_ratio` | 下拉选择 | `1:1` | 输出图像宽高比 |
| `image_size` | 下拉选择 | `1K` | 输出图像分辨率 |
| `seed` | 整数 (可选) | `0` | 用于复现的随机种子 |
| `max_input_images` | 整数 (可选) | `1` | 最大处理图像数量 (1-14) |

#### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| `image` | IMAGE | 转换后的图像张量 |
| `response_text` | STRING | API 响应文本（如有） |

---

### 界面按钮

两个节点都包含两个实用按钮：

| 按钮 | 功能 |
|------|------|
| **Verify API Key** | 验证 `.env` 中的 API key 是否正常工作 |
| **Update Models** | 从 Google 获取最新可用模型并更新下拉列表 |

## 参数选项详解

### 宽高比 (Aspect Ratio)

| 值 | 用途 |
|----|------|
| `1:1` | 正方形 - 头像、图标、社交媒体 |
| `2:3` | 竖版 - 人像照片 |
| `3:2` | 横版 - 风景照片 |
| `3:4` | 竖版 - 接近手机屏幕比例 |
| `4:3` | 横版 - 传统显示器比例 |
| `4:5` | 竖版 - Instagram 推荐比例 |
| `5:4` | 横版 - 略微偏宽 |
| `9:16` | 竖版 - 手机全屏、短视频封面 |
| `16:9` | 横版 - 视频、宽屏显示器 |
| `21:9` | 超宽横版 - 电影画幅 |

### 图像尺寸 (Image Size)

| 值 | 大约分辨率 | 用途 |
|----|-----------|------|
| `1K` | ~1024px | 快速预览、网页用图 |
| `2K` | ~2048px | 高质量展示、打印 |
| `4K` | ~4096px | 超高清、大幅印刷 |

## 示例工作流

示例工作流文件位于 `workflows/` 目录：

- `text_to_image_workflow.json` - 基础文生图工作流
- `image_to_image_workflow.json` - 图像转换示例

使用方法：
1. 打开 ComfyUI
2. 将 JSON 文件拖放到 ComfyUI 界面
3. 或使用菜单 → Load → 选择工作流文件

## 常见问题

### "No API key found" 错误

确保已创建包含 API key 的 `.env` 文件：
```
GEMINI_API_KEY=你的实际API密钥
```

### 输出黑色/占位图像

1. 检查 `response_text` 输出中的错误信息
2. 使用 "Verify API Key" 按钮验证 API key
3. 确保 API 配额充足

### 模型下拉显示错误值

1. 删除现有节点
2. 重启 ComfyUI
3. 重新添加节点：右键 → Add Node → Nano Banana

### 按钮点击无响应

点击按钮后会弹出提示对话框显示结果，确保浏览器未阻止弹窗。

## 依赖项

- `google-genai` - Google 生成式 AI Python SDK
- `Pillow` - 图像处理
- `torch` - PyTorch 张量操作
- `numpy` - 数值运算

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的节点式 UI
- [Google Gemini](https://deepmind.google/technologies/gemini/) - 支持图像生成的 AI 模型
