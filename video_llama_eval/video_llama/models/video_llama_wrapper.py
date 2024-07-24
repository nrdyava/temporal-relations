import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

class VideoLlamaWrapper(torch.nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.cfg = cfg
        self.model_config = cfg.model_cfg
        self.model_config.device_8bit = args.gpu_id
        self.model_cls = registry.get_model_class(self.model_config.arch)
        self.model = self.model_cls.from_config(self.model_config).to('cuda:{}'.format(args.gpu_id))
        self.model.eval()
        self.vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(self.vis_processor_cfg)
        self.chat = Chat(self.model, self.vis_processor, device='cuda:{}'.format(args.gpu_id))
        self.chat_state = None

    def forward(self, video_path, text, comp, **kwargs):
        img_list = self.upload_video(video_path)
        msg = f"""
<s>[INST] <<SYS>> 
In this task, you will be given a video and a text which depict two different events.
Your task is to determine if the event in the video starts or ends before or after the event described in the text.

Each problem has all of the following information:
    - A video of one of the events
    - A text summary of the other event
    - A relationship token (starts or ends) which indicates whether to compare the start or end times of the two events respectively

Please choose between before and after. Please answer in json format with explanation.
An example of a properly formatted answer is the following:
{{"answer": "after", "explanation": "The man must first turn on the heat before being able to stir fry the vegetables."}}
<</SYS>>
<s>[INST]
Text event: "{text}"
Does the event in the video {comp} before or after the event described in the text? Please choose between before and after. Please answer in json format with explanation.
"""
        self.chat.ask(msg, self.chat_state)

        llm_message = self.chat.answer(
            conv=self.chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1,
            max_new_tokens=300,
            max_length=2000
        )[0]
        return llm_message

    def upload_video(self, video):
        self.chat_state = default_conversation.copy()
        self.chat_state.system =  "You are able to understand the visual content that the user provides. Follow the provided instructions carefully."
        return self.chat.upload_video_without_audio(video, self.chat_state, [])

def build_videollama_wrapper(args):
    cfg = Config(args)
    model = VideoLlamaWrapper(cfg, args)
    return model