import os
from train import main

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TOKENIZERS_PARALLELISM'] = "true"
    
main('configs/config.yaml')