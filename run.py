import os
from train import main

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TOKENIZERS_PARALLELISM'] = "true"
    
main('config_clip.yaml')