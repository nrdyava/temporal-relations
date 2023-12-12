import os
from train import main

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"
    
main('config.yaml')