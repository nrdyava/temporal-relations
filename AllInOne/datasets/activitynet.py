import os
import json
import time
import torch
import random
import numpy as np
from AllInOne.transforms.videoaug import VideoTransform
from .activitynet_utils import read_frames_from_img_dir, collate
# from .video_base_dataset import read_frames_from_img_dir


class ActivityNetDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split, 
        num_frames, 
        image_size, 
        max_text_len, 
        masking_type='image', 
        masking_prob=0.0,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.masking_type = masking_type
        self.masking_prob = masking_prob
        dest = '/dvmm-filer3a/users/kevin/all-in-one-main/video_dest.json'
        self.dest_dict = json.load(open(dest, 'r'))
        if split == "train":
            names = ["activitynet_train"]
        elif split == "val":
            names = ["activitynet_val"]
        elif split == "test":
            names = ["activitynet_test"]
        self._load_metadata()
        self.video_transform = VideoTransform(
            mode=self.split, 
            crop_size=self.image_size,
        )
        super().__init__()

    def _load_metadata(self):
        metadata_dir = './metadata/'
        split_files = {
            'train': 'train.json',
            'val': 'val_18k.json',
            'test': 'test_19k.json',
        }
        target_split_fp = split_files[self.split]
        metadata = json.load(open(os.path.join(metadata_dir, target_split_fp), 'r'))
        meta = []
        for key, value in metadata.items():
            for event_2 in value:
                meta.append({
                    'video_key': key,
                    'event_1': event_2['sent1']['sent'],
                    'time_stamp_1': event_2['vid_event1'],
                    'event_2': event_2['sent2'][0]['sent'],
                    'time_stamp_2': event_2['vid_event2'],
                    'comp': event_2['comp'], 
                    'label': event_2['label']
                    })
        random.shuffle(meta)
        self.metadata = meta

    def get_raw_video(self, sample):
        if self.split == 'train':
            rel_fp = sample['video_key'][:-2]
        else:
            rel_fp = sample['video_key'][:-3]
        time_stamp_1 = sample['time_stamp_1']
        time_stamp_2 = sample['time_stamp_2']
        imgs_1, _ = read_frames_from_img_dir(
            rel_fp, 
            self.num_frames, 
            mode=self.split, 
            time_stamp=time_stamp_1,
            dest_dict=self.dest_dict,
        )
        imgs_2, _ = read_frames_from_img_dir( 
            rel_fp, 
            self.num_frames, 
            mode = self.split, 
            time_stamp=time_stamp_2, 
            dest_dict=self.dest_dict,
        )
        return imgs_1, imgs_2, time_stamp_1, time_stamp_2, rel_fp

    def get_video(self, index, sample, image_key="image"):
        imgs_1, imgs_2, time_stamp_1, time_stamp_2, rel_fp = self.get_raw_video(sample)
        imgs_1 = imgs_1.permute(0,2,1,3)  # to cthw
        imgs_2 = imgs_2.permute(0,2,1,3)  # to cthw
        imgs_tensor_1 = [self.video_transform(imgs_1).permute(1,0,2,3)]  # to tchw
        imgs_tensor_2 = [self.video_transform(imgs_2).permute(1,0,2,3)]  # to tchw

        return {
            "image_1": imgs_tensor_1,
            "image_2": imgs_tensor_2,
            "time_stamp_1": time_stamp_1,
            "time_stamp_2": time_stamp_2,
            "video_key": rel_fp,
            "raw_index": index,
        }    
    
    def get_text(self, raw_index, sample):
        text_1 = sample['event_1']
        text_2 = sample['event_2']
        encoding_1 = self.tokenizer(
            text_1,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        encoding_2 = self.tokenizer(
            text_2,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text_1": (text_1, encoding_1),
            "text_2": (text_2, encoding_2),
            "raw_index": raw_index,
        }

    def get_relation(self, index, sample):
        relation = {
            'comp': sample['comp'],
            'label': sample['label']
        }
        return relation
    
    def get_suite(self, index):
        sample = self.metadata[index]
        ret = dict()
        p = torch.rand(2)
        mask = p[0] < self.masking_prob
        if self.masking_type == "image":
            text_1_mask = False
            text_2_mask = False
            image_1_mask = mask
            image_2_mask = mask
        elif self.masking_type == "one_modality":
            text_1_mask = mask
            text_2_mask = not mask
            image_1_mask = not mask
            image_2_mask = mask
        ret.update(self.get_video(index, sample))
        ret.update({
            'text_1_mask': int(text_1_mask),
            'text_2_mask': int(text_2_mask),
            'image_1_mask': int(image_1_mask),
            'image_2_mask': int(image_2_mask),
        })
        ret.update(self.get_text(index, sample))
        ret.update(self.get_relation(index, sample))
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        res = self.get_suite(index)
        return res
    
    def collate(self, batch, mlm_collator):
        return collate(batch, mlm_collator)