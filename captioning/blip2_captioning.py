import os
import json
import glob
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from AllInOne.datasets import ActivityNetDataset
from lavis.models import load_model_and_preprocess

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def load_dataloader(split):
    dataset = ActivityNetDataset(
        split=split, 
        num_frames=12, 
        image_size=224, 
        max_text_len=40, 
        masking_type='image', 
        masking_prob=0.0,
        transform=False,
        use_git=False,
    )
    dataloader = DataLoader(
        dataset, 
        shuffle=False, 
        num_workers=15, 
        batch_size=1,
    )
    return dataloader

def combine_captions(captions):
    caption = f'The video shows {captions[0]}.'
    for c in captions[1:]:
        caption += f' Then it shows {c}.'
    return caption

if __name__ == '__main__':
    num_frames = 12
    # loads BLIP-2 pre-trained model
    device = 'cuda:0'
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", 
        model_type="large_coco",
        # name="blip2_t5",
        # model_type="pretrain_flant5xl",
        is_eval=True, 
        device=device,
    )
    # model = model.to(torch.float32)
    samples = []
    processor = vis_processors["eval"]
    dataloader = load_dataloader('test')
    for data in tqdm(dataloader):
        img1_frames = data['image_1'][0].permute(1,0,3,4,2).numpy()
        img2_frames = data['image_2'][0].permute(1,0,3,4,2).numpy()
        frames = []
        for frame1, frame2 in zip(img1_frames, img2_frames):
            im1 = Image.fromarray(frame1[0])
            im2 = Image.fromarray(frame2[0])
            image1 = processor(im1).unsqueeze(0).to(device)
            image2 = processor(im2).unsqueeze(0).to(device)
            frames.append(image1)
            frames.append(image2)
        frames = torch.cat(frames, dim=0)
        # frames = torch.cat(frames, dim=0).to(torch.float32)
        output = model.generate({
            "image": frames, 
            # "prompt": "Question: Describe the event in the image Answer:"
        })
        img1_captions = output[:num_frames]
        img2_captions = output[num_frames:]
        caption1 = combine_captions(img1_captions)
        caption2 = combine_captions(img2_captions)
        samples.append({
            'text_1': data['text_1'][0],
            'text_2': data['text_2'][0],
            'caption_1': caption1,
            'caption_2': caption2,
            'comp': data['comp'][0],
            'label': data['label'][0],
        })
        # if len(samples) == 10:
        #     break
        # print(output)
    
    with open('blip2_captions_12frames.json', 'w') as f:
        json.dump(samples, f, indent=4)
    