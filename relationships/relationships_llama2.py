import sys
import os
import csv
import json
import glob
import torch
import shutil
import argparse
import numpy as np
import pickle as pk
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

allinone_path = os.path.dirname(os.getcwd())
sys.path.append(allinone_path)

from AllInOne.datasets import ActivityNetDataset
from transformers import LlamaForCausalLM, LlamaTokenizer

PROMPT="""<s>[INST] <<SYS>>
In this task, you will be given two texts which summarize two different events. 
The two events may be from the same context, but may also be unrelated or have a weak relationship.
Your task is to determine if the two events have a likely temporal relation, meaning one procedes the other.

Each problem has all of the following information:
- A summary of the two events, each in a different line

<</SYS>>
Event 1: The man in the grey and white shirt enters the enclosed squash court picks up some of the balls and proceeds to load the squash cannon serving machines .
Event 2: A man wearing a white and grey shirt serves in a practice squash session and another man wearing a purple shirt returns the serves in an enclosed squash court .

Do the two events summarized above have a temporal relationship? Please choose between Yes and No. Please answer in json format with explanation.
[/INST]{"answer": "Yes", "explanation": "The serving machine must be loaded with balls before the man can practice serving."}</s>
<s>[INST]

Event 1: Sumo wrestlers lift up legs and then crouch .
Event 2: Sumo wrestlers eat food in the dojo .

Do the two events summarized above have a temporal relationship? Please choose between Yes and No. Please answer in json format with explanation.
[/INST]{"answer": "No", "explanation": "It is ambiguous which event comes first as neither must necessarily precede the other."}</s>
<s>[INST]"""
INSTRUCTION = "Do the two events summarized above have a temporal relationship? Please choose between Yes and No. Please answer in json format with explanation."

def load_model(modal_ckpt, torch_device):
    tokenizer = LlamaTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_ckpt)
    model = model.to(torch.float16)
    model = model.to(torch_device)
    # model = None
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

def run_llama(model, tokenizer, dataloader, torch_device):
    
    llama_outputs = []
    for data in tqdm(dataloader):
        batch = []
        batch_outputs = []
        for idx in range(len(data['text_1'][0])):
            prompt = PROMPT + \
                    '\nEvent 1: ' + data['text_1'][0][idx] + \
                    '\nEvent 2: ' + data['text_2'][0][idx] + \
                    '\n\n' + INSTRUCTION + '\n[/INST]'
            batch.append(prompt)
            batch_outputs.append({
                'event1': data['text_1'][0][idx],
                'event2': data['text_2'][0][idx],
            })
        # print(batch)

        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(torch_device)
        generate_ids = model.generate(inputs.input_ids, max_length=3000)
        output = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )

        # print(output)
        for idx, out in enumerate(output):
            prompt_len = len(batch[idx])
            answer_json_str1 = out[prompt_len:]
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
            batch_outputs[idx]['answer'] = answer
            batch_outputs[idx]['explanation'] = explanation
        # print(batch_outputs)

        llama_outputs += batch_outputs

    with open('relationships.json', 'w') as f:
        json.dump(llama_outputs, f, indent=4)
    return

def run_llama_subset(model, tokenizer, torch_device):
    fname = '../data_viz/samples_200.csv'
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
   
    # data = data[1:]
    llama_outputs = []
    for sample in data[1:]:
        idx = sample[0]
        prompt = PROMPT + \
                '\nEvent 1: ' + sample[1] + \
                '\nEvent 2: ' + sample[2] + \
                '\n\n' + INSTRUCTION + '\n[/INST]'
        batch = [prompt]

        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(torch_device)
        generate_ids = model.generate(inputs.input_ids, max_length=3000)
        output = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )[0]

        prompt_len = len(batch[0])
        answer_json_str1 = output[prompt_len:]
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
    with open(fname+'+llama2', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    return


def eval_relationships():
    with open('relationships.json', 'r') as f:
        data = json.load(f)
    
    def answer_to_binary(answer):
        if answer == 'Yes' or answer == 'yes':
            return 1
        elif answer == 'No' or answer == 'no':
            return 0
        return None

    relationships = [answer_to_binary(sample['answer']) for sample in data]
    filtered_relationships = [r for r in relationships if r is not None]
    filtered_relationships = np.array(filtered_relationships)
    N = len(relationships)
    F = len(filtered_relationships)
    good_samples = np.sum(filtered_relationships)
    bad_samples = F - good_samples
    unk_samples = N - F
    print('Number of good samples:', good_samples)
    print('Number of bad samples:', bad_samples)
    print('Number of inconclusive samples:', unk_samples)
    print('Percentage of good samples:', np.round(100 * good_samples / N, 2))

def clean_relationships():
    with open('relationships/relationships.json', 'r') as f:
        relationships = json.load(f)
        print(len(relationships))
    has_relationship = {}
    for r in relationships:
        events = (r['event1'], r['event2'])
        label = not (r['answer'] in ['no', 'No'])
        has_relationship[events] = label
    with open('has_relationship.pkl', 'wb') as f:
        pk.dump(has_relationship, f)
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_ckpt", dest="model_ckpt", type=str, metavar='<str>', help="File path to checkpoint")
    # args = parser.parse_args()
    
    dataloader = load_dataloader()
    # model_ckpt = args.model_ckpt
    model_ckpt = "/dvmm-filer3a/users/hammad/llama/llama-2-13b-chat-hf/"
    torch_device = 'cuda:0'
    model, tokenizer = load_model(model_ckpt, torch_device)
    # run_llama(model, tokenizer, dataloader, torch_device)
    run_llama_subset(model, tokenizer, torch_device)

    # eval_relationships()
    #clean_relationships()