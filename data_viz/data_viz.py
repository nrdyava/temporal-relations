import os
import cv2
import yaml
import copy
import torch
import numpy as np
import pytorch_lightning as pl
from AllInOne.modules import base_vision_transformer as vit
from AllInOne.datamodules.multitask_datamodule import MTDataModule

def main(config_path):
    torch.manual_seed(0)
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    dm = MTDataModule(_config, dist=False)
    dm.prepare_data()
    dm.setup('train')

    c = 1
    idx = 2
    loader = dm.train_dataloader()
    for batch in loader:
        if c < idx:
            c += 1
            continue
        img_1 = batch['image_1'][0][0][0].detach().cpu().numpy()
        img_2 = batch['image_2'][0][0][0].detach().cpu().numpy()
        print('img1 shape:', img_1.shape)
        print('img2 shape:', img_2.shape)
        print('text1:', batch['text_1'][0])
        print('text2:', batch['text_2'][0])
        cv2.imwrite('img1.jpg', np.transpose(img_1, (1,2,0)))
        cv2.imwrite('img2.jpg', np.transpose(img_2, (1,2,0)))
        break

if __name__ == '__main__':
    main('config.yaml')