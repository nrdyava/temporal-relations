#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import csv
import json
import glob
import shutil
import argparse
import numpy as np
import pickle as pk
import torch
import transformers
from transformers import pipeline
from tqdm import tqdm
#from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login

from torch.utils.data import Dataset, DataLoader

cache_dir = '/home/nrdyava/main/hf_home'
os.environ['HF_HOME'] = cache_dir

"""
allinone_path = os.path.dirname(os.getcwd())
sys.path.append(allinone_path)

from AllInOne.datasets import ActivityNetDataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
"""
#cache_dir = '/dvmm-filer3a/users/nrdyava/hf_home'
#os.environ['HF_HOME'] = cache_dir


# In[ ]:





# In[2]:


HF_TOKEN = "<hf-token>"
login(token=HF_TOKEN)


# In[ ]:





# In[3]:


system_content = """In this task, you will be given two texts which summarize two different events. 
The two events may be from the same context, but may also be unrelated or have a weak relationship.
Your task is to determine if the two events have a likely temporal relation, meaning one procedes the other.

Each problem has all of the following information:
- A summary of the two events, each in a different line"""

user_prompt_1 = """Event 1: The man in the grey and white shirt enters the enclosed squash court picks up some of the balls and proceeds to load the squash cannon serving machines .
Event 2: A man wearing a white and grey shirt serves in a practice squash session and another man wearing a purple shirt returns the serves in an enclosed squash court .

Do the two events summarized above have a temporal relationship? Please choose between Yes and No. Please answer in json format with explanation."""

assistant_response_1 = """{"answer": "Yes", "explanation": "The serving machine must be loaded with balls before the man can practice serving."}"""

user_prompt_2 = """Event 1: Sumo wrestlers lift up legs and then crouch .
Event 2: Sumo wrestlers eat food in the dojo .

Do the two events summarized above have a temporal relationship? Please choose between Yes and No. Please answer in json format with explanation."""

assistant_response_2 = """{"answer": "No", "explanation": "It is ambiguous which event comes first as neither must necessarily precede the other."}"""

INSTRUCTION = "Do the two events summarized above have a temporal relationship? Please choose between Yes and No. Please answer in json format with explanation."


# In[ ]:





# In[ ]:





# In[4]:


def run_llama_subset(pipeline, terminators):
    fname = '../data_viz/samples_200.csv'
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    llama_outputs = []

    for sample in data[1:]:
        idx = sample[0]
        user_prompt_3 = 'Event 1: ' + sample[1] + '\nEvent 2: ' + sample[2] + '\n\n' + INSTRUCTION

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt_1},
            {"role": "assistant", "content": assistant_response_1},
            {"role": "user", "content": user_prompt_2},
            {"role": "assistant", "content": assistant_response_2},
            {"role": "user", "content": user_prompt_3}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=3000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        answer_json_str1 = outputs[0]["generated_text"][-1]['content']

        try:
            answer_json = json.loads(answer_json_str1)
            answer = answer_json['answer']
            explanation = answer_json['explanation']
        except:
            try:
                last_inst_idx = output.rfind('{')
                answer_json_str2 = output[last_inst_idx:]
                answer_json = json.loads(answer_json_str2)
                answer = answer_json['answer']
                explanation = answer_json['explanation']
            except:
                answer = 'unclear'
                explanation = 'none'

        sample[-1] = answer
        sample.append(explanation)
        # llama_outputs.append({
        #     'idx': idx,
        #     'answer': answer,
        #     'explanation': explanation,
        # })

    # with open('samples+llama.json', 'w') as f:
    #     json.dump(llama_outputs, f, indent=4)
    with open(fname+'+llama3', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    return


# In[ ]:





# In[ ]:





# In[5]:


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# In[ ]:





# In[ ]:


run_llama_subset(pipeline, terminators)


# In[ ]:




