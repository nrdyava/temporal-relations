import os
import cv2
import math
import time
import torch
import decord
import numpy as np
from decord import cpu, gpu

try:
    import clip
except:
    pass


def sample_frames(num_frames, time_stamp, max_idx, sample):
    np.random.seed(0)
    start = math.ceil(time_stamp[0])
    end = math.floor(time_stamp[1])
    if end > max_idx:
        end = max_idx
    if start > max_idx:
        start = max_idx
    if start > end:
        end = start
    if sample == 'random':
        frames = np.random.choice(range(start, end+1), size=num_frames)
        # frames = np.sort(frames)
    else:
        frames = np.linspace(start, end, num_frames, dtype=int)
    return frames

def read_frames_from_img_dir(
    video_id, 
    num_frames, 
    mode='train', 
    time_stamp=None, 
    dest_dict=None,
    use_clip=False,
    use_git=False,
    event=-1,
):
    root = dest_dict[video_id[2:]]
    video_path = os.path.join(root, video_id[2:])
    if use_git or use_clip:
        sample = 'uniform'
    else:
        sample = 'random' if mode in ['train', 'val'] else 'uniform'
    decord.bridge.set_bridge('torch')
    
    retrieved_idxs = False
    while not retrieved_idxs:
        try:
            if 'frames_rest' not in video_path:
                idxs = [int(file[4:-4]) for file in os.listdir(video_path) if file.startswith('img')]
            elif 'frames_rest' in video_path:
                idxs = [int((i[:-4])) for i in os.listdir(video_path)]
            frame_idxs = sample_frames(num_frames, time_stamp, max(idxs), sample)
            retrieved_idxs = True
        except:
            print(video_path)
            print('getting idxs resting')
            time.sleep(20)

    frames = []
    for i in frame_idxs:
        if 'frames_rest' not in video_path:
            s = str(i+1)
            while len(s) != 5:
                s = '0'+ s
            video_name = f'img_{s}.jpg'
        elif 'frames_rest'  in video_path:
            video_name = str(i) + '.jpg'
        
        retrieved_frame = False
        while not retrieved_frame:
            try:
                frame = cv2.imread(os.path.join(video_path, video_name))
                frame = torch.from_numpy(frame).byte()   # hwc
                frame = frame.permute(2, 0, 1)    # chw
                frames.append(frame)
                retrieved_frame = True
            except:
                print(frame)
                print(frame_idxs, time_stamp, os.path.join(video_path, video_name))
                print('getting frames resting')
                time.sleep(20)
  
    frames = torch.stack(frames)          # tchw
    frames = frames.permute(1, 0, 2, 3)   # cthw
    return frames, frame_idxs

def collate(batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    img_keys = [k for k in list(dict_batch.keys()) if "image" in k and "mask" not in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert len(size) == 4
    if len(img_keys) != 0:
        max_height = max([i[2] for i in img_sizes])
        max_width = max([i[3] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])
        num_frames = img[0][0].shape[0]

        new_images = [
            torch.zeros(batch_size, num_frames, 3, max_height, max_width)
            for _ in range(view_size)
        ]
        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is not None:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, :, : orig.shape[-2], : orig.shape[-1]] = orig
        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k and "mask" not in k and "clip" not in k]
    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        try:
            flatten_mlms = mlm_collator(flatten_encodings)
        except:
            tmp = []
            for i in flatten_encodings:
                if 'token_type_ids' not in i.keys():
                    i['token_type_ids'] = [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ]
                    tmp.append(i)
                else:
                    tmp.append(i)
            flatten_mlms = mlm_collator(tmp)
        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                try:
                    input_ids[_i, : len(_input_ids)] = _input_ids
                except:
                    print()
                attention_mask[_i, : len(_attention_mask)] = _attention_mask
            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask

    return dict_batch

def clip_tokenizer(ret):
    text_1_clip = clip.tokenize(ret["text_1"][0], truncate=True)
    text_2_clip = clip.tokenize(ret["text_2"][0], truncate=True)

    ret.update({
        'text_1_clip': text_1_clip,
        'text_2_clip': text_2_clip,
    })
    return ret


def git_encoder(ret, git_processor):
    def process_text(text, padding):
        encoding = git_processor(text=text, padding=padding, return_tensors="pt")
        return encoding["input_ids"], encoding["attention_mask"]

    def process_imgs(imgs, padding):
        encoding = git_processor(images=imgs, padding="max_length", return_tensors="pt")
        return encoding["pixel_values"]
    
    def pad(x, shape):
        assert x.shape[1] <= shape[1]
        out = torch.zeros(shape)
        out[:, :x.shape[1]] = x
        return out
    
    assert ret['text_1_mask'] == ret['image_2_mask']
    assert ret['image_1_mask'] == ret['text_2_mask']
    text_event = 1 + ret['text_1_mask']
    if text_event == 2:
        flip_label = {'after': 'before', 'before': 'after'}
        ret['label'] = flip_label[ret['label']]
    ret['question'] = f"Does the event in the text {ret['comp']} before or after the event shown in the video?"
    ret['question_label'] = ret['question'] + ret['label']

    if text_event == 1:
        imgs = ret['image_2'][0]
        text = 'Text event:' + ret['text_1'][0]
        text_ids, text_masks = process_text(text, "do_not_pad")
        img_ids = process_imgs([imgs[0], imgs[1], imgs[2]], "max_length")
    else:
        imgs = ret['image_1'][0]
        text = 'Text event:' + ret['text_2'][0]
        text_ids, text_masks = process_text(text, "do_not_pad")
        img_ids = process_imgs([imgs[0], imgs[1], imgs[2]], "max_length")

    question_ids, question_masks = process_text(ret['question'], "do_not_pad")
    question_label_ids, _ = process_text(ret['question_label'], "do_not_pad")
    text_ids[0][-1] = 99

    text_ids = text_ids.int()
    text_masks = text_masks.int()
    question_ids = question_ids.int()
    question_masks = question_masks.int()
    question_label_ids = question_label_ids.int()

    question_length = question_masks.shape[1] - 1
    text_length = text_ids.shape[1]
    eos_idx = question_length + text_length
    
    output_shape = (1, 512)
    input_ids = torch.cat([text_ids, question_ids[:, 1:]], dim=1)
    input_masks = torch.cat([text_masks, question_masks[:, 1:]], dim=1)
    label_ids = torch.cat([text_ids, question_label_ids[:, 1:]], dim=1).long()
    need_predict = torch.Tensor([eos_idx-1, eos_idx])

    input_ids = pad(input_ids, output_shape).int()
    input_masks = pad(input_masks, output_shape).int()
    label_ids = pad(label_ids, output_shape).long()
    img_ids = img_ids.float()
    need_predict = need_predict.int()

    ret.update({
        'input_ids': input_ids,
        'input_masks': input_masks,
        'label_ids': label_ids,
        'img_ids': img_ids,
        'need_predict': need_predict,
    })
    return ret