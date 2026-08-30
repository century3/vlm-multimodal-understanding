# 视觉大模型与多模态理解

课程案例仓库：用视觉大模型（VLM）做文档解析、保险影像识别和视频理解。

仓库地址：https://github.com/century3/vlm-multimodal-understanding

## 目录

| 目录 | 内容 |
|------|------|
| `CASE-MinerU使用` | MinerU 解析 PDF：本地 `magic_pdf` + 云端 Agent API |
| `CASE-VLM在车险中的应用` | Qwen-VL 识别车险影像（里程表、验车、损伤、危险驾驶等） |
| `CASE-VLM在寿险中的应用` | Qwen-VL 识别寿险单据、多语种文档和本地图片 |
| `CASE-汽车剐蹭视频理解` | InternVideo2.5 理解剐蹭视频 `car.mp4` |
| `视觉大模型与多模态理解.pdf` | 课程讲义 |

本仓库包含课件 PDF、案例 PDF、示例图片和 `car.mp4`。本地模型权重（`modelscope_models/`）体积过大，没有上传，需要按各案例说明自行下载。

## 环境

建议使用已有的 `pytorch` conda 环境（Python 3.11 + CUDA）。不同案例依赖不同，不要强行统一成一套包。

云端 Qwen-VL 案例需要阿里云百炼 API Key：

```powershell
$env:DASHSCOPE_API_KEY = "你的Key"
```

## 1. MinerU 文档解析

进入 `CASE-MinerU使用`。

**云端（免 Token）**：打开 `1-MinerU.ipynb`，按单元格运行。走 MinerU Agent 轻量接口，提交远程 PDF → 轮询 `task_id` → 下载 Markdown。

**本地**：先安装 `magic-pdf` 并下载模型到 `CASE-MinerU使用/modelscope_models`（可用同目录的 `download_models.py`），再运行：

```powershell
python CASE-MinerU使用/1-MinerU.py
```

脚本会优先使用目录里已有的 PDF（如 `Qwen3-tech_report.pdf`），结果写到 `CASE-MinerU使用/output/`。

## 2. Qwen-VL 车险识别

进入 `CASE-VLM在车险中的应用`，设置 `DASHSCOPE_API_KEY` 后运行：

```powershell
python CASE-VLM在车险中的应用/1-Qwen-VL-保险识别-cn.py
python CASE-VLM在车险中的应用/2-Qwen-VL-chat1.py
```

或打开对应 `.ipynb`。示例图在同目录（里程表、验车、事故要素、损伤评估等）。Prompt 模板见 `prompt_template_cn.xlsx` / `prompt_template_en.xlsx`。

## 3. Qwen-VL 寿险识别

进入 `CASE-VLM在寿险中的应用`：

```powershell
python CASE-VLM在寿险中的应用/1-Qwen-VL-保险识别-cn.py
python CASE-VLM在寿险中的应用/2-Qwen-VL-本地图片.py
```

`2-Qwen-VL-本地图片.py` 读取本目录图片；多语种单据示例为 `1-Chinese-document-extraction.jpg` 等到 `5-Korean-document-extraction.jpg`。

## 4. 汽车剐蹭视频理解

进入 `CASE-汽车剐蹭视频理解`。依赖见 `requirements.txt`。Windows 上脚本用 OpenCV 读视频，不依赖 `decord`。

首次运行会通过 ModelScope 下载 `OpenGVLab/InternVideo2_5_Chat_8B`（默认缓存到 `D:\AI-AGent-Learning\models`，可按本机路径改 `video-understand.py` 里的 `CACHE_DIR`）。

```powershell
python CASE-汽车剐蹭视频理解/video-understand.py
```

输入视频是同目录的 `car.mp4`。8B 模型显存占用较高，同一时间只跑一份脚本。

## 说明

- 保险案例走 DashScope OpenAI 兼容接口，没有 Key 无法调用云端模型。
- MinerU 本地解析依赖 `magic_pdf` 和对应权重；云端 notebook 不需要本机模型。
- 视频理解脚本针对当前 `transformers` / Windows / 16GB 显卡做过兼容处理，换环境时可能需要再调显存和注意力实现。
