import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision import tv_tensors
from torchvision import transforms
from torchvision.transforms import v2
import random
from vocab import STOI
from torch.utils.data import DataLoader

# clip level spatial transformation
clip_spatial_transform = v2.Compose([
    # 224x224 crop
    # randomly cropping the input with a cropping size 
    # between [0.08, 1.00] of the original image.
    # Jitter aspect ratio between 3/4 to 4/3.
    v2.RandomResizedCrop(
        size=(224, 224),
        scale=(0.08, 1.00),
        ratio=(3.0 / 4.0, 4.0 / 3.0)
    ),

    # random horizontal flip with probability 0.5
    v2.RandomHorizontalFlip(p=0.5),

    # Rand Augment with probability of 0.5,
    # for 4 layers, 
    # of maximum magnitude 7
    v2.RandomApply([
        v2.RandAugment(num_ops=4, magnitude=7)
    ], p=0.5),

    # convert to float32 and rescale to [0.0, 1.0].
    # For the above transformations, they are more
    # efficient in uint8 (uint8 is more efficient than float32)
    v2.ToDtype(torch.float32, scale=True),

    # we need to normalize (ImageNet Normalization) before RandomErasing
    # otherwise the zeroed out pixels might no longer be 0 if we do
    # random erasing first
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Random Erasing with probability 0.25
    v2.RandomErasing(p=0.25)
])


class MixupCutmixAugmentation:
    def __init__(self, num_classes = 400):
        # Mixup with alpha=0.8
        self.mixup = v2.MixUp(alpha=0.8, num_classes=num_classes)
        # CutMix, default alpha
        self.cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
        
    def augment(self, videos, labels):
        # videos: (B, T, C, H, W)
        # labels: (B,) 

        B = videos.shape[0]
        half = B // 2
        
        # wrap the tensors as Video so v2 knows to treat the temporal dimension
        # Note we have to slice the batch into two halves first before wrapping as Video,
        # as it seems like slicing does not preserve the Video wrapper
        first_half_videos = tv_tensors.Video(videos[:half]) # (B/2, T, C, H, W)
        second_half_videos = tv_tensors.Video(videos[half:]) # (B/2, T, C, H, W)

        # apply MixUp to the first half of the batch
        # labels_m is converted into soft labels of shape (B/2, num_classes)
        # similar to one-hot but with soft probabilities at the position 
        # of the two mixed classes
        videos_m, labels_m = self.mixup(first_half_videos, labels[:half])
        
        # apply CutMix to the second half of the batch
        # labels_c is converted into soft labels of shape (B/2, num_classes)
        videos_c, labels_c = self.cutmix(second_half_videos, labels[half:])
        
        # recombine
        mixed_videos = torch.cat([videos_m, videos_c], dim=0)
        mixed_labels = torch.cat([labels_m, labels_c], dim=0)
        
        return mixed_videos, mixed_labels # (B, T, C, H, W), (B, num_classes)


class KineticsCSVDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_dir,
        frames_per_clip=16,
        temporal_stride = 4,
    ):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.frames_per_clip = frames_per_clip
        self.temporal_stride = temporal_stride
        self.transform = clip_spatial_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # video_path = os.path.join(self.video_dir, row[0])
        video_path = row['path']
        label = row['label']

        label = STOI[label]

        # video: (T, H, W, C), uint8
        # Kinetics should be mostly 10 seconds video, 30 fps, so 300 frames per clip
        # (there might be exceptions)
        video, _, _ = read_video(video_path)

        # temporal sampling
        T = video.shape[0]
        
        clip_length = self.frames_per_clip * self.temporal_stride # 64

        if T < clip_length:
            # pad by repeating the last frame
            num_pad = clip_length - T
            pad_frames = video[-1:].repeat(num_pad, 1, 1, 1) # (num_pad, H, W, C)
            video = torch.cat([video, pad_frames], dim=0) # (clip_length, H, W, C)

        # randint is inclusive on both ends
        starting_idx = random.randint(0, T-clip_length) # a random starting point for the clip
        indices = torch.arange(starting_idx, starting_idx + clip_length, step=self.temporal_stride)
        video = video[indices] # (16, H, W, C)

        # rearrange to match v2 and tv_tensor's expectation
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

        # we need to wrapped the tensor as a Video so v2 knows it needs to 
        # treat the temporal dimension consistently. 
        # Same transformation applied across all frames in an input clip
        video = tv_tensors.Video(video)
        video = self.transform(video)

        return video, label 
    

def get_loader(csv_path, video_dir, batch_size=8, num_workers=4):
    dataset = KineticsCSVDataset(csv_path=csv_path,
                                 video_dir=video_dir)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader