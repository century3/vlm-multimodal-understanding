#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-vl-max", #qwen-vl-plus
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是什么"},
                {"type": "image_url","image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}}
            ]
        }
        ]
    )
print(completion.model_dump_json())


# In[2]:


messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "框出图中轮毂的位置"},
                {"type": "image_url","image_url": {"url": "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"}}
            ]
        }
    ]

completion = client.chat.completions.create(
    model="qwen-vl-max-2024-08-09",
    messages=messages
    )
print(completion.model_dump_json())


# In[3]:


#help(completion)


# In[3]:


messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "图中轮毂的位置在哪里"},
                {"type": "image_url","image_url": {"url": "https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg"}}
            ]
        })
print(messages)


# In[4]:


completion = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=messages
)     

print(completion.model_dump_json())


# In[5]:


completion


# In[9]:


# 这里需要用到 autodl，调用Qwen-VL本地的模型
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model/Qwen/Qwen-VL-Chat", trust_remote_code=True)

query = tokenizer.from_list_format([
    #{'image': 'https://easycar.oss-cn-beijing.aliyuncs.com/2.jpg'}, # Either a local path or an url
    {'image': 'https://easycar.oss-cn-beijing.aliyuncs.com/car_undistorted.jpg'},
    {'text': '这是什么?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中展示了一辆蓝色的特斯拉Model3轿车的尾部，从图中可以观察到车辆的尾部有部分凹陷和划痕，但无法确定是否为车辆被追尾。

response, history = model.chat(tokenizer, '框出图中轮毂的位置', history=history)
print(response)
# <ref>轮毂</ref><box>(154,553),(310,880)</box>
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save('wheel.jpg')
else:
    print("no box")
"""


# In[14]:


"""
#response = '<ref>轮毂</ref><box>(100,460),(250,880)</box>'
#response = '<ref>轮毂</ref><box>(1,1),(1000,1000)</box>'
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save('wheel.jpg')
else:
    print("no box")
"""


# In[3]:


"""
response, history = model.chat(tokenizer, '框出图中凹陷和划痕的位置', history=history)
print(response)
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save('car_damage.jpg')
else:
    print("no box")
"""


# In[2]:


# import gradio
# gradio.__version__
# import transformers 
# transformers.__version__
# import torch
# torch.__version__


# In[3]:


"""
# 第1轮对话
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
    {'text': '这是什么?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。
"""


# In[4]:


"""
# 第1轮对话
query = tokenizer.from_list_format([
    {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/1.jpg'}, 
    {'text': '这是什么？'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
"""


# In[3]:


"""
# 第2轮对话
response, history = model.chat(tokenizer, '框出图中包装有缺陷的位置', history=history)
print(response)
# <ref>击掌</ref><box>(536,509),(588,602)</box>
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save('2.jpg')
else:
    print("no box")
"""


# In[9]:


"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图像
img = mpimg.imread('2.jpg')
# 显示图像
plt.imshow(img)
# 添加标题
plt.title('2.jpg')
# 隐藏坐标轴
plt.axis('off')
# 显示图像
plt.show()
"""


# In[10]:


"""
# 读取图像
img = mpimg.imread('3.jpg')
# 显示图像
plt.imshow(img)
# 隐藏坐标轴
plt.axis('off')
# 显示图像
plt.show()

# 第1轮对话
query = tokenizer.from_list_format([
    {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/3.jpg'}, 
    {'text': '厦门航空在哪个区？'}, 
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)

# 第1轮对话
query = tokenizer.from_list_format([
    {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/3.jpg'}, 
    {'text': 'A区有哪些航空公司？'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)

# 第2轮对话
"""


# In[11]:


"""
# 读取图像
img = mpimg.imread('4.jpg')
# 显示图像
plt.imshow(img)
# 隐藏坐标轴
plt.axis('off')
# 显示图像
plt.show()

# 第1轮对话
query = tokenizer.from_list_format([
    {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/4.jpg'}, 
    {'text': '机场巴士在第几层？'}, 
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
"""


# In[12]:


"""
# 读取图像
img = mpimg.imread('5.png')
# 显示图像
plt.imshow(img)
# 隐藏坐标轴
plt.axis('off')
# 显示图像
plt.show()

# 第1轮对话
query = tokenizer.from_list_format([
    {'image': 'https://vl-image.oss-cn-shanghai.aliyuncs.com/5.png'}, 
    {'text': '头等舱免费行李额是多少KG？'}, 
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
"""

