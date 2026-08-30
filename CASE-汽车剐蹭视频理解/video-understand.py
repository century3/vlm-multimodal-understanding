"""
属于 InternVL 2.5系列
视频理解与生成：可以用于视频内容的分析、总结和生成相关的文本描述。
视觉问答：能够回答与图像或视频内容相关的问题。
多模态对话：支持与用户进行包含视觉信息的对话。
"""


# In[2]:


# 模型下载（断点续传）
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
from modelscope import snapshot_download

CACHE_DIR = r'D:\AI-AGent-Learning\models'
model_dir = snapshot_download('OpenGVLab/InternVideo2_5_Chat_8B', cache_dir=CACHE_DIR)
# model_dir = snapshot_download('internlm/internlm2_5-7b-chat', cache_dir='/root/autodl-tmp/models')
# model_dir = snapshot_download('LLM-Research/Mistral-7B-Instruct-v0.3', cache_dir='/root/autodl-tmp/models')
# model_dir = snapshot_download('AI-ModelScope/bert-base-uncased', cache_dir='/root/autodl-tmp/models')


# In[1]:


# 导入必要的库
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import inspect
import json
import time

# #region agent log
_DBG_LOG = r"d:\AI-AGent-Learning\49-视觉大模型与多模态理解\debug-92fc8b.log"


def _dbg(hypothesisId, location, message, data):
    rec = {
        "sessionId": "92fc8b",
        "runId": "pre-fix",
        "hypothesisId": hypothesisId,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(_DBG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_dbg("boot", "video-understand.py:boot", "instrumented script started", {"pid": os.getpid()})
# #endregion


class VideoReader:
    """OpenCV 替代 decord.VideoReader（Windows 上 decord 经常装不上）。"""

    def __init__(self, video_path, ctx=None, num_threads=1):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f'无法打开视频: {video_path}')
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

    def __len__(self):
        return self._n

    def get_avg_fps(self):
        return self._fps

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        if not ok:
            raise IndexError(f'无法读取第 {idx} 帧')
        return _Frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_frames(self, indices):
        """按时间顺序解码，避免对每一帧 cap.set 随机跳转。"""
        indices = [int(i) for i in indices]
        needed = set(indices)
        frames = {}
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        max_i = max(needed) if needed else -1
        while idx <= max_i:
            ok, frame = self.cap.read()
            if not ok:
                break
            if idx in needed:
                frames[idx] = _Frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        missing = [i for i in indices if i not in frames]
        if missing:
            raise IndexError(f'无法读取帧: {missing[:8]}')
        return [frames[i] for i in indices]


class _Frame:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


def cpu(*_args, **_kwargs):
    return None


def patch_internvl_config(model_dir):
    """让 InternVL 自定义配置兼容 transformers 5.x（空构造时缺少 llm_config）。"""
    src_path = os.path.join(model_dir, 'configuration_internvl_chat.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    if 'has_no_defaults_at_init' not in src:
        src = src.replace(
            "    model_type = 'internvl_chat'\n    is_composition = True",
            "    model_type = 'internvl_chat'\n    is_composition = True\n    has_no_defaults_at_init = True",
        )
        src = src.replace(
            "        if llm_config.get('architectures', None) is not None:\n"
            "            if llm_config.get('architectures')[0] == 'LlamaForCausalLM':\n"
            "                self.llm_config = LlamaConfig(**llm_config)\n"
            "            elif llm_config.get('architectures')[0] == 'InternLM2ForCausalLM':\n"
            "                self.llm_config = InternLM2Config(**llm_config)\n"
            "            else:\n"
            "                pass",
            "        arch = (llm_config.get('architectures') or [None])[0]\n"
            "        if arch == 'LlamaForCausalLM':\n"
            "            self.llm_config = LlamaConfig(**llm_config)\n"
            "        elif arch == 'InternLM2ForCausalLM':\n"
            "            self.llm_config = InternLM2Config(**llm_config)\n"
            "        else:\n"
            "            self.llm_config = InternLM2Config(**llm_config)",
        )
        src = src.replace(
            "        output['llm_config'] = self.llm_config.to_dict()",
            "        output['llm_config'] = self.llm_config.to_dict() if getattr(self, 'llm_config', None) is not None else {}",
        )
        with open(src_path, 'w', encoding='utf-8') as f:
            f.write(src)

    vit_path = os.path.join(model_dir, 'modeling_intern_vit.py')
    with open(vit_path, encoding='utf-8') as f:
        vit = f.read()
    old_dpr = 'dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]'
    if old_dpr in vit:
        vit = vit.replace(
            old_dpr,
            'n_layers = int(config.num_hidden_layers)\n'
            '        drop_rate = float(config.drop_path_rate or 0)\n'
            '        dpr = [drop_rate] if n_layers <= 1 else [drop_rate * i / (n_layers - 1) for i in range(n_layers)]',
        )
        with open(vit_path, 'w', encoding='utf-8') as f:
            f.write(vit)

    chat_path = os.path.join(model_dir, 'modeling_internvl_chat_hico2.py')
    with open(chat_path, encoding='utf-8') as f:
        chat = f.read()
    chat_changed = False
    if 'self.system_message = self.conv_template.system_message\n        self.post_init()' not in chat:
        chat = chat.replace(
            'self.system_message = self.conv_template.system_message',
            'self.system_message = self.conv_template.system_message\n        self.post_init()',
        )
        chat_changed = True
    if 'input_ids_2d' not in chat:
        # transformers 5.x：只传 inputs_embeds 时会造出长度为 0 的 input_ids，
        # 生成立刻停在 EOS 上，chat() 解码后就是空字符串。
        chat = chat.replace(
            "            input_embeds = self.language_model.get_input_embeddings()(input_ids)\n"
            "            B, N, C = input_embeds.shape\n"
            "            input_embeds = input_embeds.reshape(B * N, C)\n"
            "\n"
            "            input_ids = input_ids.reshape(B * N)\n"
            "            selected = (input_ids == self.img_context_token_id)\n"
            "            assert selected.sum() != 0\n"
            "            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)\n"
            "\n"
            "            input_embeds = input_embeds.reshape(B, N, C)\n"
            "        else:\n"
            "            input_embeds = self.language_model.get_input_embeddings()(input_ids)\n"
            "\n"
            "        outputs = self.language_model.generate(\n"
            "            inputs_embeds=input_embeds,\n"
            "            attention_mask=attention_mask,\n"
            "            generation_config=generation_config,\n"
            "            output_hidden_states=output_hidden_states,\n"
            "            use_cache=True,\n"
            "            **generate_kwargs,\n"
            "        )\n"
            "\n"
            "        return outputs",
            "            input_embeds = self.language_model.get_input_embeddings()(input_ids)\n"
            "            B, N, C = input_embeds.shape\n"
            "            input_ids_2d = input_ids.reshape(B, N)\n"
            "            input_embeds = input_embeds.reshape(B * N, C)\n"
            "\n"
            "            input_ids = input_ids.reshape(B * N)\n"
            "            selected = (input_ids == self.img_context_token_id)\n"
            "            assert selected.sum() != 0\n"
            "            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)\n"
            "\n"
            "            input_embeds = input_embeds.reshape(B, N, C)\n"
            "        else:\n"
            "            input_ids_2d = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)\n"
            "            input_embeds = self.language_model.get_input_embeddings()(input_ids_2d)\n"
            "\n"
            "        outputs = self.language_model.generate(\n"
            "            input_ids=input_ids_2d,\n"
            "            inputs_embeds=input_embeds,\n"
            "            attention_mask=attention_mask,\n"
            "            generation_config=generation_config,\n"
            "            output_hidden_states=output_hidden_states,\n"
            "            use_cache=True,\n"
            "            **generate_kwargs,\n"
            "        )\n"
            "        if not isinstance(outputs, torch.Tensor):\n"
            "            outputs = outputs.sequences\n"
            "        prompt_len = input_ids_2d.shape[-1]\n"
            "        if outputs.shape[-1] > prompt_len:\n"
            "            outputs = outputs[:, prompt_len:]\n"
            "        return outputs",
        )
        chat_changed = True
    if '.to(dtype=input_embeds.dtype, device=input_embeds.device)' not in chat:
        chat = chat.replace(
            'input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)',
            'input_embeds[selected] = vit_embeds.reshape(-1, C).to(dtype=input_embeds.dtype, device=input_embeds.device)\n'
            '            print(f"视觉token={int(selected.sum())} vit mean={float(vit_embeds.float().mean()):.4f} '
            'std={float(vit_embeds.float().std()):.4f} nan={int(torch.isnan(vit_embeds).sum())}", flush=True)',
        )
        chat_changed = True
    if chat_changed:
        with open(chat_path, 'w', encoding='utf-8') as f:
            f.write(chat)

    lm_path = os.path.join(model_dir, 'modeling_internlm2.py')
    with open(lm_path, encoding='utf-8') as f:
        lm_src = f.read()
    lm_changed = False
    if 'GenerationMixin' not in lm_src:
        lm_src = lm_src.replace(
            'from transformers.modeling_utils import PreTrainedModel',
            'from transformers.modeling_utils import PreTrainedModel\nfrom transformers.generation import GenerationMixin',
        )
        lm_src = lm_src.replace(
            'class InternLM2ForCausalLM(InternLM2PreTrainedModel):',
            'class InternLM2ForCausalLM(InternLM2PreTrainedModel, GenerationMixin):',
        )
        lm_changed = True
    if '_internlm2_cache_seq_len' not in lm_src:
        # transformers 5.x 默认用 DynamicCache，InternLM2 仍按旧 tuple 下标取 KV，会报
        # TypeError: 'DynamicCache' object is not subscriptable
        lm_src = lm_src.replace(
            "logger = logging.get_logger(__name__)\n",
            "logger = logging.get_logger(__name__)\n"
            "\n"
            "def _internlm2_cache_seq_len(past_key_values):\n"
            "    if past_key_values is None:\n"
            "        return 0\n"
            "    if hasattr(past_key_values, 'get_seq_length'):\n"
            "        try:\n"
            "            return int(past_key_values.get_seq_length() or 0)\n"
            "        except Exception:\n"
            "            return 0\n"
            "    try:\n"
            "        return int(past_key_values[0][0].shape[2])\n"
            "    except Exception:\n"
            "        return 0\n"
            "\n"
            "def _internlm2_layer_kv(past_key_values, idx):\n"
            "    if past_key_values is None:\n"
            "        return None\n"
            "    if hasattr(past_key_values, 'layers'):\n"
            "        if idx >= len(past_key_values.layers):\n"
            "            return None\n"
            "        layer = past_key_values.layers[idx]\n"
            "        if getattr(layer, 'keys', None) is None:\n"
            "            return None\n"
            "        if hasattr(layer, 'get_seq_length') and layer.get_seq_length() == 0:\n"
            "            return None\n"
            "        return (layer.keys, layer.values)\n"
            "    return past_key_values[idx]\n"
            "\n"
            "def _internlm2_write_cache(orig_cache, next_decoder_cache):\n"
            "    if next_decoder_cache is None:\n"
            "        return orig_cache\n"
            "    if orig_cache is not None and hasattr(orig_cache, 'layers'):\n"
            "        for idx, kv in enumerate(next_decoder_cache):\n"
            "            if kv is None:\n"
            "                continue\n"
            "            k, v = kv[0], kv[1]\n"
            "            while len(orig_cache.layers) <= idx:\n"
            "                cls = getattr(orig_cache, 'layer_class_to_replicate', None)\n"
            "                orig_cache.layers.append(cls() if cls is not None else type(orig_cache.layers[0])())\n"
            "            layer = orig_cache.layers[idx]\n"
            "            if not getattr(layer, 'is_initialized', False) and hasattr(layer, 'lazy_initialization'):\n"
            "                layer.lazy_initialization(k, v)\n"
            "            layer.keys = k\n"
            "            layer.values = v\n"
            "            layer.is_initialized = True\n"
            "        return orig_cache\n"
            "    return next_decoder_cache\n"
            "\n",
        )
        lm_src = lm_src.replace(
            "        seq_length_with_past = seq_length\n"
            "        past_key_values_length = 0\n"
            "        if past_key_values is not None:\n"
            "            past_key_values_length = past_key_values[0][0].shape[2]\n"
            "            seq_length_with_past = seq_length_with_past + past_key_values_length\n",
            "        seq_length_with_past = seq_length\n"
            "        _orig_past_key_values = past_key_values\n"
            "        past_key_values_length = _internlm2_cache_seq_len(past_key_values)\n"
            "        seq_length_with_past = seq_length_with_past + past_key_values_length\n",
        )
        lm_src = lm_src.replace(
            "            past_key_value = past_key_values[idx] if past_key_values is not None else None\n",
            "            past_key_value = _internlm2_layer_kv(past_key_values, idx)\n",
        )
        lm_src = lm_src.replace(
            "        next_cache = next_decoder_cache if use_cache else None\n",
            "        next_cache = _internlm2_write_cache(_orig_past_key_values, next_decoder_cache) if use_cache else None\n",
        )
        lm_src = lm_src.replace(
            "        if past_key_values is not None:\n"
            "            past_length = past_key_values[0][0].shape[2]\n",
            "        past_length = _internlm2_cache_seq_len(past_key_values)\n"
            "        if past_length == 0:\n"
            "            past_key_values = None\n"
            "        if past_key_values is not None:\n",
        )
        lm_changed = True
    if 'You cannot specify both input_ids and inputs_embeds at the same time' in lm_src:
        lm_src = lm_src.replace(
            "        if input_ids is not None and inputs_embeds is not None:\n"
            "            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')\n",
            "        if input_ids is not None and inputs_embeds is not None:\n"
            "            input_ids = None\n",
        )
        lm_changed = True
    if 'neg = -1e4 if dtype' not in lm_src:
        _before_mask = lm_src
        lm_src = lm_src.replace(
            "    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)\n"
            "    mask_cond = torch.arange(mask.size(-1), device=device)\n"
            "    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)\n"
            "    mask = mask.to(dtype)\n",
            "    neg = -1e4 if dtype in (torch.float16, torch.bfloat16) else torch.finfo(dtype).min\n"
            "    mask = torch.full((tgt_len, tgt_len), neg, dtype=dtype, device=device)\n"
            "    mask_cond = torch.arange(mask.size(-1), device=device)\n"
            "    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)\n",
        )
        lm_src = lm_src.replace(
            "    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)\n",
            "    neg = -1e4 if dtype in (torch.float16, torch.bfloat16) else torch.finfo(dtype).min\n"
            "    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), neg)\n",
        )
        if lm_src == _before_mask:
            print('警告: InternLM2 注意力 mask 替换未命中原文', flush=True)
        else:
            print('已将 InternLM2 注意力 mask 从 -inf 改为 -1e4（避免 bf16 下 softmax 崩溃）', flush=True)
        lm_changed = True
    if 'scaled_dot_product_attention' not in lm_src:
        lm_src = lm_src.replace(
            "        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)\n"
            "\n"
            "        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):\n"
            "            raise ValueError(\n"
            "                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'\n"
            "                f' {attn_weights.size()}'\n"
            "            )\n"
            "\n"
            "        if attention_mask is not None:\n"
            "            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):\n"
            "                raise ValueError(\n"
            "                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'\n"
            "                )\n"
            "            attn_weights = attn_weights + attention_mask\n"
            "\n"
            "        # upcast attention to fp32\n"
            "        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)\n"
            "        attn_output = torch.matmul(attn_weights, value_states)\n",
            "        attn_output = torch.nn.functional.scaled_dot_product_attention(\n"
            "            query_states, key_states, value_states,\n"
            "            attn_mask=None, dropout_p=0.0, is_causal=(q_len > 1),\n"
            "        )\n"
            "        attn_weights = None\n",
        )
        print('已将 InternLM2 eager 注意力改为 SDPA is_causal', flush=True)
        lm_changed = True
    if lm_changed:
        with open(lm_path, 'w', encoding='utf-8') as f:
            f.write(lm_src)

    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'modules', 'transformers_modules')
    if os.path.isdir(cache_dir):
        for root, _, files in os.walk(cache_dir):
            if 'configuration_internvl_chat.py' in files:
                shutil.copy2(src_path, os.path.join(root, 'configuration_internvl_chat.py'))
            if 'modeling_intern_vit.py' in files:
                shutil.copy2(vit_path, os.path.join(root, 'modeling_intern_vit.py'))
            if 'modeling_internvl_chat_hico2.py' in files:
                shutil.copy2(os.path.join(model_dir, 'modeling_internvl_chat_hico2.py'), os.path.join(root, 'modeling_internvl_chat_hico2.py'))
            if 'modeling_internlm2.py' in files:
                shutil.copy2(os.path.join(model_dir, 'modeling_internlm2.py'), os.path.join(root, 'modeling_internlm2.py'))


# 模型配置：强制 GPU
model_path = model_dir
patch_internvl_config(model_path)
if not torch.cuda.is_available():
    raise RuntimeError(
        '未检测到 GPU。nvidia-smi 无法连接 NVIDIA 驱动（Driver Not Loaded）。'
        '请安装或重新启用 NVIDIA 驱动后，用 conda pytorch 环境重跑。本脚本强制走 GPU，不会回退 CPU。'
    )
print(f'加载模型到 GPU: {model_path}  device={torch.cuda.get_device_name(0)}')
print(f'加载前显存: 已用 {torch.cuda.memory_allocated()/1024**3:.2f} GB / 峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
torch.cuda.empty_cache()

from transformers.modeling_utils import PreTrainedModel

_orig_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized


def _mark_tied_weights_as_initialized(self, loading_info):
    if not getattr(self, 'all_tied_weights_keys', None):
        try:
            self.post_init()
        except Exception:
            pass
        if not getattr(self, 'all_tied_weights_keys', None):
            self.all_tied_weights_keys = {}
    return _orig_mark_tied(self, loading_info)


PreTrainedModel.mark_tied_weights_as_initialized = _mark_tied_weights_as_initialized

# transformers 5 的 AutoTokenizer 会把 InternLM2 的 sentencepiece 错转成 TikToken。
# 必须直接用官方 InternLM2Tokenizer，才会自动加 <s>，特殊符号 ID 也才和权重一致。
import sys
if model_path not in sys.path:
    sys.path.insert(0, model_path)
from tokenization_internlm2 import InternLM2Tokenizer
tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
print(
    f'分词器 {type(tokenizer).__name__} len={len(tokenizer)} bos={tokenizer.bos_token_id} '
    f'add_bos={tokenizer.add_bos_token} im_start={tokenizer.convert_tokens_to_ids("<|im_start|>")} '
    f'IMG_CONTEXT={tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")}',
    flush=True,
)
# #region agent log
_dbg("A", "video-understand.py:tokenizer", "tokenizer special ids", {
    "cls": type(tokenizer).__name__,
    "add_bos": bool(tokenizer.add_bos_token),
    "bos": tokenizer.bos_token_id,
    "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
    "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    "img_ctx": tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"),
    "vocab": len(tokenizer),
})
# #endregion
_load = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    output_loading_info=True,
)
if isinstance(_load, tuple):
    model, _loading_info = _load
else:
    model, _loading_info = _load, {}
if next(model.parameters()).device.type != 'cuda':
    model = model.to('cuda')
model.eval()
torch.cuda.empty_cache()
# InternLM2 的 inv_freq/cos/sin 是 persistent=False，low_cpu_mem_usage 加载后全是 0。
_rope0 = model.language_model.model.layers[0].attention.rotary_emb
# #region agent log
_dbg("G", "video-understand.py:rope_before", "cos before rebuild", {
    "cos_std": float(_rope0.cos_cached.float().std()),
    "cos_mean": float(_rope0.cos_cached.float().mean()),
    "inv_std": float(_rope0.inv_freq.float().std()),
    "cos_shape": list(_rope0.cos_cached.shape),
})
# #endregion
_dev = next(model.parameters()).device
for _layer in model.language_model.model.layers:
    _r = _layer.attention.rotary_emb
    _dim = _r.dim
    _inv = 1.0 / (_r.base ** (torch.arange(0, _dim, 2, device=_dev, dtype=torch.float32) / _dim))
    _r.register_buffer('inv_freq', _inv, persistent=False)
    _r._set_cos_sin_cache(seq_len=_r.max_position_embeddings, device=_dev, dtype=torch.float32)
print(
    f'已重建 RoPE  inv_std={_rope0.inv_freq.float().std().item():.4f} '
    f'cos_std={_rope0.cos_cached.float().std().item():.4f} '
    f'mean={_rope0.cos_cached.float().mean().item():.4f}',
    flush=True,
)
# #region agent log
_dbg("G", "video-understand.py:rope_after", "cos after rebuild", {
    "runId": "post-fix",
    "inv_std": float(_rope0.inv_freq.float().std()),
    "cos_std": float(_rope0.cos_cached.float().std()),
    "cos_mean": float(_rope0.cos_cached.float().mean()),
})
# #endregion
_attn0 = model.language_model.model.layers[0].attention
print(f'当前注意力: {_attn0.__class__.__name__}  impl={model.language_model.model.config.attn_implementation}', flush=True)
if _attn0.__class__.__name__ == 'InternLM2FlashAttention2':
    _eager_cls = _attn0.__class__.__mro__[1]
    for _layer in model.language_model.model.layers:
        _layer.attention.__class__ = _eager_cls
    model.language_model.model.config.attn_implementation = 'eager'
    print('已将 FlashAttention2 切换为 eager，避免 Windows 上生成直接收束到结束符', flush=True)
_emb = model.language_model.get_input_embeddings().weight
_head = model.language_model.get_output_embeddings().weight
print(
    f'wqkv.std={model.language_model.model.layers[0].attention.wqkv.weight.std().item():.4f} '
    f'output.std={model.language_model.output.weight.std().item():.4f} '
    f'mlp1.std={model.mlp1[1].weight.std().item():.4f}',
    flush=True,
)
# #region agent log
_miss = [k for k in _loading_info.get('missing_keys', []) if 'language_model' in k or k.endswith('output.weight') or 'tok_embeddings' in k]
_attn_src = inspect.getsource(_attn0.forward)
_dbg("B", "video-understand.py:attn", "attention implementation", {
    "attn_cls": _attn0.__class__.__name__,
    "impl": str(model.language_model.model.config.attn_implementation),
    "has_sdpa": "scaled_dot_product_attention" in _attn_src,
    "src_head": _attn_src.replace("\n", " ")[:240],
})
_dbg("C", "video-understand.py:weights", "load and lm_head", {
    "missing_lm": _miss[:12],
    "missing_lm_n": len(_miss),
    "unexpected_n": len(_loading_info.get('unexpected_keys', [])),
    "embed_std": float(_emb.float().std()),
    "head_std": float(_head.float().std()),
    "same_storage": int(_emb.data_ptr() == _head.data_ptr()),
    "max_diff": float((_emb.float() - _head.float()).abs().max()) if _emb.shape == _head.shape else None,
})
_attn_once = [False]
_attn_orig = _attn0.forward
_rope = _attn0.rotary_emb
_rope_orig = _rope.forward
_rope_once = [False]


def _rope_wrap(x, seq_len=None):
    out = _rope_orig(x, seq_len)
    if not _rope_once[0]:
        _rope_once[0] = True
        cos, sin = out
        _dbg("G", "video-understand.py:rope", "rotary cache", {
            "rope_cls": type(_rope).__name__,
            "x_shape": list(x.shape),
            "seq_len": int(seq_len) if seq_len is not None else None,
            "cos_shape": list(cos.shape),
            "cos_std": float(cos.float().std()),
            "base": float(getattr(_rope, "base", -1.0)),
            "max_pos": int(getattr(_rope, "max_position_embeddings", -1)),
        })
    return out


def _attn_wrap(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    if (not _attn_once[0]) and hidden_states.dim() == 3 and hidden_states.shape[1] > 4:
        _attn_once[0] = True
        am = attention_mask
        data = {
            "q": int(hidden_states.shape[1]),
            "hid_last_std": float(hidden_states[0, -1].float().std()),
            "hid_last_norm": float(hidden_states[0, -1].float().norm()),
            "pos_head": position_ids[0, :4].tolist() if position_ids is not None else None,
            "pos_tail": position_ids[0, -4:].tolist() if position_ids is not None else None,
            "mask_dim": None if am is None else list(am.shape),
        }
        if am is not None and am.dim() == 4:
            data["mask_last_min"] = float(am[0, 0, -1].float().min())
            data["mask_last_max"] = float(am[0, 0, -1].float().max())
            data["mask_last_keep"] = int((am[0, 0, -1] == 0).sum())
        _dbg("D", "video-understand.py:attn_wrap", "layer0 prefill mask/pos", data)
    return _attn_orig(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        **kwargs,
    )


_attn0.forward = _attn_wrap
_attn0.rotary_emb.forward = _rope_wrap
# #endregion
with torch.no_grad():
    _probe = tokenizer('<|im_start|>user\n1+1等于多少？<|im_end|>\n<|im_start|>assistant\n', return_tensors='pt')
    print(
        '探测 prompt 头/尾:',
        [(int(i), tokenizer.decode([int(i)])) for i in _probe['input_ids'][0, :4]],
        [(int(i), tokenizer.decode([int(i)])) for i in _probe['input_ids'][0, -6:]],
        flush=True,
    )
    _probe = {k: v.to(model.device) for k, v in _probe.items()}
    _pout = model.language_model(
        input_ids=_probe['input_ids'],
        attention_mask=_probe['attention_mask'],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    _plogits = _pout.logits[0, -1].float()
    _pv, _pi = torch.topk(_plogits, 8)
    print(
        '纯文本探测 1+1 top8:',
        [(int(i), round(float(v), 2), tokenizer.decode([int(i)])) for i, v in zip(_pi, _pv)],
        flush=True,
    )
    # #region agent log
    _h = _pout.hidden_states[-1][0, -1].float()
    _dbg("E", "video-understand.py:text_probe", "1+1 last logits", {
        "seq": int(_probe['input_ids'].shape[-1]),
        "mask_sum": int(_probe['attention_mask'].sum()),
        "h_norm": float(_h.norm()),
        "h_std": float(_h.std()),
        "logits_finite": bool(torch.isfinite(_plogits).all()),
        "top8": [(int(i), round(float(v), 2), tokenizer.decode([int(i)])) for i, v in zip(_pi, _pv)],
        "ids_head": _probe['input_ids'][0, :6].tolist(),
        "ids_tail": _probe['input_ids'][0, -6:].tolist(),
    })
    model.language_model.to(dtype=torch.float16)
    _p16 = model.language_model(
        input_ids=_probe['input_ids'],
        attention_mask=_probe['attention_mask'],
        use_cache=False,
        return_dict=True,
    )
    _l16 = _p16.logits[0, -1].float()
    _v16, _i16 = torch.topk(_l16, 8)
    _top16 = [(int(i), round(float(v), 2), tokenizer.decode([int(i)])) for i, v in zip(_i16, _v16)]
    print('纯文本探测 fp16 top8:', _top16, flush=True)
    _dbg("F", "video-understand.py:fp16_probe", "1+1 last logits fp16", {
        "top8": _top16,
        "logits_finite": bool(torch.isfinite(_l16).all()),
    })
    model.language_model.to(dtype=torch.bfloat16)
    # #endregion

from transformers.generation import GenerationMixin
if not hasattr(model.language_model, 'generate'):
    model.language_model.__class__ = type(
        model.language_model.__class__.__name__,
        (model.language_model.__class__, GenerationMixin),
        {},
    )


def _greedy_generate(
    self,
    input_ids=None,
    inputs_embeds=None,
    attention_mask=None,
    max_new_tokens=256,
    eos_token_id=None,
    use_cache=True,
    generation_config=None,
    output_hidden_states=None,
    **kwargs,
):
    """真 greedy：不再强行跳过结束符。强迫输出会得到「正浙浙浙」这种胡话。"""
    if generation_config is not None:
        max_new_tokens = getattr(generation_config, 'max_new_tokens', None) or max_new_tokens or 256
        if eos_token_id is None:
            eos_token_id = getattr(generation_config, 'eos_token_id', None)
    if isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_token_id}
    elif eos_token_id is not None:
        eos_ids = {int(eos_token_id)}
    else:
        eos_ids = set()
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(int(im_end_id))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
    if input_ids is not None:
        print(
            'generate 输入头/尾:',
            [(int(i), tokenizer.decode([int(i)])) for i in input_ids[0, :4]],
            [(int(i), tokenizer.decode([int(i)])) for i in input_ids[0, -6:]],
            f'len={input_ids.shape[-1]}',
            flush=True,
        )
    generated = []
    past = None
    cur_embeds = inputs_embeds
    cur_ids = None if inputs_embeds is not None else input_ids
    mask = attention_mask

    for step in range(int(max_new_tokens)):
        out = self(
            input_ids=cur_ids,
            inputs_embeds=cur_embeds,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :].float()
        if step == 0:
            if cur_embeds is not None:
                _e = cur_embeds.float()
                print(
                    f'inputs_embeds mean={_e.mean().item():.4f} std={_e.std().item():.4f} '
                    f'nan={int(torch.isnan(cur_embeds).sum())}',
                    flush=True,
                )
            topv, topi = torch.topk(logits[0], 8)
            print(
                '首步 top8:',
                [(int(i), round(float(v), 2), tokenizer.decode([int(i)])) for i, v in zip(topi, topv)],
                flush=True,
            )
            # #region agent log
            _dbg("E", "video-understand.py:greedy", "first token", {
                "has_embeds": cur_embeds is not None,
                "seq": int(mask.shape[-1]) if mask is not None else None,
                "top8": [(int(i), round(float(v), 2), tokenizer.decode([int(i)])) for i, v in zip(topi, topv)],
                "argmax": int(logits[0].argmax().item()),
            })
            # #endregion
        tid = int(logits[0].argmax().item())
        if tid in eos_ids:
            print(f'第 {step} 步结束 token={tid} {tokenizer.decode([tid])!r}', flush=True)
            break
        generated.append(tid)
        if step < 12 or step % 32 == 0:
            print(f'  +{step}: {tid} {tokenizer.decode([tid])!r}', flush=True)
        past = out.past_key_values
        cur_embeds = None
        cur_ids = torch.tensor([[tid]], dtype=torch.long, device=device)
        if mask is not None:
            mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)

    raw = tokenizer.decode(generated, skip_special_tokens=False) if generated else ''
    print(f'贪心生成 {len(generated)} tokens  raw={raw[:240]!r}', flush=True)
    if generated:
        return torch.tensor([generated], dtype=torch.long, device=device)
    return torch.zeros((1, 0), dtype=torch.long, device=device)


import types
model.language_model.generate = types.MethodType(_greedy_generate, model.language_model)
print(
    '模型加载完成，language_model.generate =',
    hasattr(model.language_model, 'generate'),
    f' 显存 {torch.cuda.memory_allocated()/1024**3:.2f} GB',
)
with torch.no_grad():
    _p = tokenizer('<|im_start|>user\n1+1等于多少？<|im_end|>\n<|im_start|>assistant\n', return_tensors='pt')
    _p = {k: v.to(model.device) for k, v in _p.items()}
    _gen = model.language_model.generate(
        input_ids=_p['input_ids'],
        attention_mask=_p['attention_mask'],
        max_new_tokens=24,
    )
    print('纯文本生成:', tokenizer.decode(_gen[0].tolist(), skip_special_tokens=False), flush=True)
if os.environ.get('IV25_DEBUG_PROBE_ONLY') == '1':
    print('IV25_DEBUG_PROBE_ONLY=1，跳过视频推理', flush=True)
    raise SystemExit(0)

# ImageNet 数据集的均值和标准差
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    构建图像转换pipeline
    
    参数:
        input_size: 输入图像大小
    
    返回:
        transform: 转换pipeline
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    寻找最接近原始图像宽高比的目标比例
    
    参数:
        aspect_ratio: 原始图像的宽高比
        target_ratios: 目标比例列表
        width: 原始图像宽度
        height: 原始图像高度
        image_size: 目标图像大小
        
    返回:
        best_ratio: 最佳比例
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    动态预处理图像，根据宽高比将图像分割成多个块
    
    参数:
        image: 原始图像
        min_num: 最小块数
        max_num: 最大块数
        image_size: 目标图像大小
        use_thumbnail: 是否使用缩略图
        
    返回:
        processed_images: 处理后的图像列表
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算现有图像宽高比
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 寻找最接近目标的宽高比
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, 
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    """
    加载并处理图像
    
    参数:
        image: 输入图像
        input_size: 输入大小
        max_num: 最大块数
        
    返回:
        pixel_values: 处理后的图像张量
    """
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    获取视频帧索引
    
    参数:
        bound: 时间边界 [开始时间, 结束时间]
        fps: 视频帧率
        max_frame: 最大帧数
        first_idx: 第一帧索引
        num_segments: 分段数量
        
    返回:
        frame_indices: 帧索引数组
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
    """
    根据视频时长计算帧数
    
    参数:
        duration: 视频时长（秒）
        
    返回:
        num_frames: 计算出的帧数
    """
    local_num_frames = 4        
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
    
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)

    return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    """
    加载并处理视频
    
    参数:
        video_path: 视频路径
        bound: 时间边界
        input_size: 输入大小
        max_num: 最大块数
        num_segments: 分段数量
        get_frame_by_duration: 是否根据时长获取帧数
        
    返回:
        pixel_values: 处理后的视频帧张量
        num_patches_list: 每帧的块数列表
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    print(f'抽取视频帧: {len(frame_indices)} 帧, fps={fps:.1f}, 总帧数={max_frame + 1}')
    sampled = vr.get_frames(frame_indices)
    for i, frame in enumerate(sampled):
        img = Image.fromarray(frame.asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        if (i + 1) % 8 == 0 or (i + 1) == len(sampled):
            print(f'  已处理 {i + 1}/{len(sampled)} 帧')
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# 评估设置
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    max_new_tokens=256,
    num_beams=1,
)
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car.mp4")
# 16GB 显卡上 8B bf16 权重已占约 14GB，128 帧会在生成时 OOM；演示用 16 帧
num_segments = 16


with torch.no_grad():
  # 加载视频并处理
  print('开始抽帧...')
  pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
  pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
  video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
  print(f'视频张量 {tuple(pixel_values.shape)}  显存 {torch.cuda.memory_allocated()/1024**3:.2f} GB')
  
  # 单轮对话：视频详细描述
  question1 = "Describe this video in detail."
  question = video_prefix + question1
  print('推理中: Describe this video in detail. ...', flush=True)
  output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
  print('【回答】', output1 if output1 else '(空)', flush=True)
  
  # 多轮对话：询问视频中的人数
  question2 = "How many people appear in the video?"
  print('推理中: How many people appear in the video? ...', flush=True)
  output2, chat_history = model.chat(tokenizer, pixel_values, question2, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
  print('【回答】', output2 if output2 else '(空)', flush=True)


# In[4]:


# video_prefix


# In[3]:


with torch.no_grad():
  # 单轮对话：询问车辆损伤部位（中文）
  question1 = "车的哪个部位损伤了？"
  question = video_prefix + question1
  print('推理中: 车的哪个部位损伤了？ ...', flush=True)
  output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
  print('【回答】', output1 if output1 else '(空)', flush=True)
  
  # 多轮对话：询问车辆碰撞位置（中文）
  question2 = "车撞到哪里了？"
  print('推理中: 车撞到哪里了？ ...', flush=True)
  output2, chat_history = model.chat(tokenizer, pixel_values, question2, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
  print('【回答】', output2 if output2 else '(空)', flush=True)

