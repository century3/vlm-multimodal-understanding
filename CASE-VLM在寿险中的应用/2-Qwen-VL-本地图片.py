#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
from dashscope import MultiModalConversation
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

local_file_path = 'file://1-Chinese-document-extraction.jpg'
messages = [{
    'role': 'system',
    'content': [{
        'text': 'You are a helpful assistant.'
    }]
}, {
    'role':
    'user',
    'content': [
        {
            'image': local_file_path
        },
        {
            'text': '图片里有什么东西?'
        },
    ]
}]
response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
print(response)


# In[2]:


print(response.output.choices[0].message.content[0]['text'])

