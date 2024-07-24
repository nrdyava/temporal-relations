import os
import json
import glob
import torch
import shutil
import argparse
import numpy as np
import pickle as pk
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
#from AllInOne.datasets import ActivityNetDataset
from transformers import LlamaForCausalLM, LlamaTokenizer

# text zeroshot
PROMPT="""<s>[INST] <<SYS>>
In this task, you will be given two texts which summarize two different events. 
Your task is to determine if whether or not the first event starts or ends before or after the second event.

Each problem has all of the following information:
- A summary of the two events, each in a different line
- A relationship token (starts or ends) which indicates whether to compare the start or end times of the two events respectively

<</SYS>>
Event 1: He adds oil to the pan while talking to the camera . He then stir fries the chopped vegetables .
Event 2: He is shown putting on an apron and then turning the stove top to high heat .

Does event 1 start before or after event 2? Please choose between before and after. Please answer in json format with explanation.
[/INST]{"answer": "after", "explanation": "The man must first turn on the heat before being able to stir fry the vegetables."}</s>
<s>[INST]"""
# ------------------------------------------------------------------- # 
# multimodal
# PROMPT="""<s>[INST] <<SYS>>
# In this task, you will be given two texts which summarize two different events. 
# Your task is to determine if whether or not the first event starts or ends before or after the second event.

# Each problem has all of the following information:
# - A summary of the two events, each in a different line
# - A relationship token (starts or ends) which indicates whether to compare the start or end times of the two events respectively

# For each sample, please choose between before and after. Please answer in json format with explanation. 
# An example of a properly formatted answer: {"answer": "after", "explanation": "The man must first turn on the heat before being able to stir fry the vegetables."}

# <</SYS>>
# <s>[INST]"""

def get_instruction(comp):
    if comp[-1] == 's':
        comp = comp[:-1]
    instruction = f"Does event 1 {comp} before or after event 2? Please choose between before and after. Please answer in json format with explanation."
    return instruction

def load_model(modal_ckpt, torch_device):
    tokenizer = LlamaTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_ckpt)
    model = model.to(torch.float16)
    model = model.to(torch_device)
    return model, tokenizer

def load_dataloader():
    dataset = ActivityNetDataset(
        split='test', 
        num_frames=3, 
        image_size=224, 
        max_text_len=40, 
        masking_type='image', 
        masking_prob=0.0,
        transform=True,
        use_git=False,
    )
    dataloader = DataLoader(
        dataset, 
        shuffle=False, 
        num_workers=15, 
        batch_size=1,
    )
    return dataloader

def run_llama(model, tokenizer, dataloader, torch_device, filter=False):
    
    if filter:
        print('filtering')
        with open('relationships/has_relationship.pkl', 'rb') as f:
            has_relationship = pk.load(f)
            
    batch_size = 1
    llama_outputs = []
    for data in tqdm(dataloader):
        batch = []
        batch_outputs = []
        for idx in range(batch_size):
            if filter:
                key = (data['text_1'][idx][0], data['text_2'][idx][0])
                if has_relationship[key] == 0:
                    print('skipping')
                    continue
            prompt = PROMPT + \
                    '\nEvent 1: ' + data['text_1'][idx][0] + \
                    '\nEvent 2: ' + data['text_2'][idx][0] + \
                    '\n\n' + get_instruction(data['comp'][idx]) + '\n[/INST]'
            batch.append(prompt)
            batch_outputs.append({
                'event1': data['text_1'][idx][0],
                'event2': data['text_2'][idx][0],
                'comp': data['comp'][idx],
                'label': data['label'][idx],
            })
        # for idx in range(len(data['text_1'])):
        #     prompt = PROMPT + \
        #             '\nEvent 1: ' + data['text_1'][0] + \
        #             '\nEvent 2: ' + data['caption_2'] + \
        #             '\n\n' + get_instruction(data['comp']) + '\n[/INST]'
        #     batch.append(prompt)
        #     batch_outputs.append({
        #         'event1': data['text_1'][0],
        #         'event2': data['caption_2'],
        #         'comp': data['comp'],
        #         'label': data['label'],
        #     })
        # print(batch)

        if len(batch) == 0:
            continue
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(torch_device)
        generate_ids = model.generate(inputs.input_ids, max_length=2048)
        output = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )

        # print(output)
        for idx, out in enumerate(output):
            prompt_len = len(batch[idx])
            answer_json_str1 = out[prompt_len:]
            batch_outputs[idx]['raw_answer'] = answer_json_str1
            try:
                answer_json = json.loads(answer_json_str1)
                answer = answer_json['answer']
                explanation = answer_json['explanation']
            except:
                try:
                    last_inst_idx = out.rfind('{')
                    answer_json_str2 = out[last_inst_idx:]
                    answer_json = json.loads(answer_json_str2)
                    answer = answer_json['answer']
                    explanation = answer_json['explanation']
                except:
                    answer = 'unclear'
                    explanation = 'unknown'
            batch_outputs[idx]['answer'] = answer
            batch_outputs[idx]['explanation'] = explanation

        llama_outputs += batch_outputs
        # if len(llama_outputs) > 10:
        #     break

    with open('llama_predictions/llama_zeroshot_filtered.json', 'w') as f:
        json.dump(llama_outputs, f, indent=4)
    return

def eval_llama():
    with open('../llama_predictions/llama_blip_zeroshot_t1v2.json', 'r') as f:
        data = json.load(f)
    
    #with open('../llama_predictions/llama_zeroshot.json', 'r') as f:
    #    data = json.load(f)

    acc, unclear = [], []
    for sample in data:
        acc.append(int(sample['label'] == sample['answer']))
        unclear.append(int(sample['answer'] == 'unclear'))
    acc = np.array(acc)
    unclear = np.array(unclear)
    print('Samples:', len(acc))
    print('Accuracy:', acc.mean())
    print('Unclear rate:', unclear.mean())

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_ckpt", dest="model_ckpt", type=str, metavar='<str>', help="File path to checkpoint")
    # args = parser.parse_args()
    
    # dataloader = load_dataloader()
    # # with open('blip2_captions_12frames.json', 'r') as f:
    # #     dataloader = json.load(f)
    # model_ckpt = args.model_ckpt
    # torch_device = 'cuda:0'
    # model, tokenizer = load_model(model_ckpt, torch_device)
    # run_llama(model, tokenizer, dataloader, torch_device, filter=True)

    eval_llama()
