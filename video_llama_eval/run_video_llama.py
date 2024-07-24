import torch
import argparse
import numpy as np
from tqdm import tqdm

import os
import sys
import json
import copy

from torch.utils.data import Dataset, DataLoader

sys.path.append("/dvmm-filer3a/users/nrdyava/temporal-relations")
from video_llama_eval.video_llama.models.video_llama_wrapper import build_videollama_wrapper
from AllInOne.datamodules.multitask_datamodule import MTDataModule
import pytorch_lightning as pl

import time
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model-type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def run_video_llama(_config):
    print('Initializing Chat')
    args = parse_args()
    model = build_videollama_wrapper(args)

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    dm = MTDataModule(_config, dist=False)
    dm.prepare_data()
    dm.setup(stage='test')
    dataloader = dm.test_dataloader()

    print()
    print(len(dataloader))
    batch = next(iter(dataloader))
    for k, v in batch.items():
        print(k, np.array(v).shape)
    print()
    
    try:
        with open('video_llama_output_v1t2_15frames.json', 'w') as f:
            saved_output = json.load(f)
        done_keys = set([item['video_key'] for item in saved_output])
    except:
        saved_output = []
        done_keys = {}

    video_name = 'video.mp4'
    with torch.no_grad():
        for batch in tqdm(dataloader):
            video_key = batch['video_key'][0]
            if video_key in done_keys:
                continue
            raw_idx = batch['raw_index'][0]
            label = batch['label'][0]
            comp = batch['comp'][0]
            img1 = batch['image_1'][0][0]
            img2 = batch['image_2'][0][0]
            text1 = batch['text_1'][0]
            text2 = batch['text_2'][0]

            frames = []
            # for f in img1:
            #     frame = np.transpose(f.numpy(), (2, 1, 0))
            #     frames.append(frame)
            for f in img2:
                frame = np.transpose(f.numpy(), (2, 1, 0))
                frames.append(frame)
            clip = ImageSequenceClip(frames, fps=6)
            clip.write_videofile(video_name, logger=None)
            time.sleep(0.1)

            llm_message = model(video_name, text1, comp)
            # llm_message = model(video_name, text2, comp)

            try:
                answer_json = json.loads(llm_message)
                answer = answer_json['answer']
                explanation = answer_json['explanation']
            except:
                try:
                    last_inst_idx = llm_message.rfind('{')
                    llm_message2 = llm_message[last_inst_idx:]
                    answer_json = json.loads(llm_message2)
                    answer = answer_json['answer']
                    explanation = answer_json['explanation']
                except:
                    answer = 'unclear'
                    explanation = 'none'

            saved_output.append(
                {
                    'video_key': str(video_key),
                    'raw_idx': int(raw_idx),
                    'comp': str(comp),
                    'label': str(label),
                    'output': str(llm_message),
                    'answer': answer,
                    'explanation': explanation,
                    'raw_answer': llm_message,
                }
            )

            with open('video_llama_output_t1v2_15frames.json', 'w') as f:
                json.dump(saved_output, f)

def eval_video_llama_output():
    text1 = True
    flip_label = {'after': 'before', 'before': 'after'}
    print('Evaluating')
    with open('video_llama_output_t1v2_15frames.json', 'r') as f:
        output = json.load(f)
    
    cor, unk = 0, 0
    ta, tb, fa, fb = 0, 0, 0, 0
    for item in tqdm(output):
        label = item['label']
        if text1:
            label = flip_label[label]
        if item['answer'] != 'unclear':
            p = item['answer']
        else:
            video_llama_output = item['output']
            afters = video_llama_output.count('after')
            befores = video_llama_output.count('before')
            if afters > befores:
                p = 'after'
            elif afters < befores:
                p = 'before'
            else:
                unk += 1
                continue
        if p == 'after':
            if label == 'after':
                cor += 1
                ta += 1
            else:
                fa += 1
        else:
            if label == 'before':
                cor += 1
                tb += 1
            else:
                fb += 1
    tot = len(output)
    print(f'Correct: {cor} / {tot}')
    print(f'Incorrect: {tot-cor-unk} / {tot}')
    print(f'Inconclusive: {unk} / {tot}')
    print('*'*30)
    print('Accuracy:', np.round(100*cor/tot, 2))
    print('Incorrect rate:', np.round(100*(tot-cor-unk)/tot, 2))
    print('Inconclusive rate:', np.round(100*unk/tot, 2))
    print('*'*30)
    print('True afters:', ta)
    print('True befores:', tb)
    print('False afters', fa)
    print('False befores', fb)

if __name__ == '__main__':
    config_activity_net = {'exp_name': 'mask_one_modality', 'seed': 0, 
                       'datasets': ['activitynet'], 
                       'loss_names': {'itm': 0, 'itc': 0, 'mlm': 0, 'mpp': 0, 'ind_itc': 1, 
                                      'vcop': 0, 'vqa': 0, 'openend_vqa': 0, 'mc_vqa': 0, 'nlvr2': 0, 
                                      'irtr': 0, 'multiple_choice': 0, 'vcr_q2a': 0, 'zs_classify': 0}, 
                       'batch_size': 1, 'linear_evaluation': False, 'train_transform_keys': ['pixelbert_randaug'], 
                       'val_transform_keys': ['pixelbert'], 'image_size': 224, 'patch_size': 16, 'max_image_len': -1, 
                       'draw_false_image': 1, 'image_only': False, 'num_frames': 15, 'vqav2_label_size': 3129, 
                       'msrvttqa_label_size': 1501, 'max_text_len': 40, 'tokenizer': 'bert-base-uncased', 
                       'vocab_size': 30522, 'whole_word_masking': False, 'mlm_prob': 0.15, 'draw_false_text': 10, 
                       'draw_options_text': 0, 'vit': 'vit_base_patch16_224', 'hidden_size': 768, 'num_heads': 12, 
                       'num_layers': 12, 'mlp_ratio': 4,'drop_rate': 0.1, 'shared_embedding_dim': 512, 
                       'save_checkpoints_interval': 5, 'optim_type': 'adamw', 'learning_rate': 0.0, 'weight_decay': 0.1, 
                       'decay_power': 1, 'max_epoch': 25, 'max_steps': None, 'warmup_steps': 0.1, 'end_lr': 0, 'lr_mult': 1,
                       'backend': 'others', 'get_recall_metric': False, 'get_itc_recall_metric': False, 'get_ind_recall_metric': False, 
                       'retrieval_views': 3, 'resume_from': None, 'fast_dev_run': False, 'val_check_interval': 1.0, 'test_only': True, 
                       'data_root': 'DataSet', 'log_dir': 'result', 'per_gpu_batchsize': 1, 
                       'num_gpus': 1, 'num_nodes': 1,  'num_workers': 8, 'precision': 16, 'load_path': ''}
    run_video_llama(config_activity_net)
    # eval_video_llama_output()