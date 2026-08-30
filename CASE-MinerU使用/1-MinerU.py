#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from pathlib import Path

# 无论从哪个目录启动，都按脚本所在目录定位 PDF 和输出
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# 使用magic-pdf本地处理文件
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


def resolve_pdf_path() -> Path:
    preferred = [
        BASE_DIR / "INTERNVIDEO2.5.pdf",
        BASE_DIR / "Qwen3-tech_report.pdf",  # 11 页，适合演示
        BASE_DIR / "三国演义.pdf",  # 380 页，全本会非常慢
    ]
    for path in preferred:
        if path.is_file():
            return path
    pdfs = sorted(p for p in BASE_DIR.glob("*.pdf") if p.is_file())
    if pdfs:
        return pdfs[0]
    raise FileNotFoundError(f"在 {BASE_DIR} 下未找到 PDF 文件")


# 参数设置
pdf_path = resolve_pdf_path()
pdf_file_path = str(pdf_path)
name_without_suff = pdf_path.stem  # 去除文件扩展名

# 准备环境
local_image_dir = str(BASE_DIR / "output" / "images")
local_md_dir = str(BASE_DIR / "output")  # 图片和输出目录
image_dir = os.path.basename(local_image_dir)  # 获取图片目录名

# 创建输出目录（如果不存在）
os.makedirs(local_image_dir, exist_ok=True)
os.makedirs(local_md_dir, exist_ok=True)

# 初始化数据写入器
image_writer = FileBasedDataWriter(local_image_dir)
md_writer = FileBasedDataWriter(local_md_dir)

print(f"开始处理文件: {pdf_file_path}")

# 读取PDF文件内容
reader = FileBasedDataReader("")  # 初始化数据读取器
pdf_bytes = reader.read(pdf_file_path)  # 读取PDF文件内容为字节流

# 处理流程
## 创建PDF数据集实例
ds = PymuDocDataset(pdf_bytes)  # 使用PDF字节流初始化数据集

## 分析文档类型并选择处理方式
print("正在分析文档类型...")
doc_type = ds.classify()
print(f"文档类型: {'OCR类型' if doc_type == SupportedPdfParseMethod.OCR else '文本类型'}")

## 推理阶段
print("开始推理分析...")
if doc_type == SupportedPdfParseMethod.OCR:
    # 如果是OCR类型的PDF（扫描件/图片型PDF）
    print("使用OCR模式进行分析...")
    infer_result = ds.apply(doc_analyze, ocr=True, formula_enable=False)  # 应用OCR模式的分析

    ## 处理管道
    print("处理OCR结果...")
    pipe_result = infer_result.pipe_ocr_mode(image_writer)  # OCR模式的处理管道

else:
    # 如果是文本型PDF
    print("使用文本模式进行分析...")
    infer_result = ds.apply(doc_analyze, ocr=False, formula_enable=False)  # 应用普通文本模式的分析

    ## 处理管道
    print("处理文本结果...")
    pipe_result = infer_result.pipe_txt_mode(image_writer)  # 文本模式的处理管道

print("分析完成，开始生成输出...")

### 绘制模型分析结果到每页PDF
print("生成模型分析可视化...")
infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

### 获取模型推理结果
model_inference_result = infer_result.get_infer_res()

### 绘制布局分析结果到每页PDF
print("生成布局分析可视化...")
pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

### 绘制文本块(span)分析结果到每页PDF
print("生成文本块分析可视化...")
pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

### 获取Markdown格式的内容
print("生成Markdown内容...")
md_content = pipe_result.get_markdown(image_dir)  # 包含图片相对路径

### 保存Markdown文件
print("保存Markdown文件...")
pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

### 获取内容列表（JSON格式）
print("生成内容列表...")
content_list_content = pipe_result.get_content_list(image_dir)

### 保存内容列表到JSON文件
print("保存内容列表JSON...")
pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

### 获取中间JSON格式数据
print("生成中间JSON数据...")
middle_json_content = pipe_result.get_middle_json()

### 保存中间JSON数据
print("保存中间JSON数据...")
pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

print(f"处理完成，输出文件保存在 {os.path.abspath(local_md_dir)} 目录")
print(f"生成的文件：")
print(f"- {name_without_suff}.md - Markdown格式文档")
print(f"- {name_without_suff}_content_list.json - 内容结构JSON")
print(f"- {name_without_suff}_middle.json - 中间处理数据")
print(f"- {name_without_suff}_model.pdf - 模型分析可视化")
print(f"- {name_without_suff}_layout.pdf - 布局分析可视化")
print(f"- {name_without_suff}_spans.pdf - 文本块分析可视化")
print(f"- {local_image_dir} - 提取的图像文件")

