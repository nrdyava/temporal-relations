import pytorchvideo.transforms as video_transforms

# input: (C, T, H, W) output: (C, T, H, W)
def VideoTransform(mode='train', crop_size=224):
    return video_transforms.create_video_transform(
        mode=mode, 
        min_size=int(crop_size*1.2),
        max_size=int(crop_size*1.5),
        crop_size=crop_size,
        aug_type='randaug',
        num_samples=None,
    )