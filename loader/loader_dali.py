import os
import pandas as pd
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.auto_aug import rand_augment
from vocab import STOI 

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# DALI utilizes NVDEC (NVIDIA's hardware decoder) to read, decode, and sample the videos 
# directly on the GPU, completely bypassing the CPU overhead.

def apply_augmentations(video):
    # video data aug on gpu, cpu generates a few random number
    # random Resized Crop, 
    video = fn.random_resized_crop(
        video,
        size=[224, 224],
        random_area=[0.08, 1.0],
        random_aspect_ratio=[0.75, 1.333]
    )

    # random Horizontal Flip (p=0.5)
    coin_flip = fn.random.coin_flip(probability=0.5)
    video = fn.flip(video, horizontal=coin_flip)

    # RandAugment (p=0.5, 4 layers, magnitude 7)
    # apply RandAugment given 0.5 probability
    apply_ra = fn.random.coin_flip(probability=0.5, dtype=types.BOOL)
    if apply_ra:
        video = rand_augment.rand_augment(video, shape=[224, 224], n=4, m=7)

    # (1/255.0) scaling and normalization in one step.
    video = fn.crop_mirror_normalize(
        video,
        dtype=types.FLOAT,
        output_layout="FHWC",
        mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    )

    # random erasing (p=0.25)
    apply_erasing = fn.random.coin_flip(probability=0.25, dtype=types.BOOL)
    if apply_erasing:

        # top left corner of the erasure rectangle is uniformly sampled from 
        # the image
        # shape=[2] means we want to sample 2 values (for x and y coordinates)
        anchor = fn.random.uniform(range=[0.0, 1.0], shape=[2])

        # size of the erased region is between 2% to 33% of the original image
        shape  = fn.random.uniform(range=[0.02, 0.33], shape=[2])
        
        video = fn.erase(
            video,
            anchor=anchor,
            shape=shape,
            axes=[1, 2], # erase in the H and W dimensions
            normalized=True, # the anchor and shape are (ratios) normalized to [0.0, 1.0], rather than absolute pixel counts
            fill_value=0.0 # Will fill with 0, black
        )

    # permute from (T, H, W, C) to FCHW which aligns with PyTorch's (T, C, H, W) expectation.
    video = fn.transpose(video, perm=[0, 3, 1, 2])

    return video


# one issue with fn.readers.video is it will aggregate and put all possible (16 frames 
# stride 4) non-overlapping sequences in a bucket, shuffle it if random_shuffle, 
# and draw from that bucket. So we can get multiple examples from the same
# video clip and the epoch size will be a lot larger than what we expect.
# This won't match the kinetics epoch logic by itself.

# the argument 'step' defaults to the temporal span of the clip (in our case, 16×4=64 frames). 
# For a standard 10-second, 300-frame Kinetics video, DALI will automatically extract ~4 sequential, 
# non-overlapping clips per video. random_shuffle=True just tosses all of these generated clips into 
# a shuffle bucket. This will make a single epoch 4x larger.

# this flag allows you to use functional control flow (like if statements and else blocks) directly 
# inside the GPU-accelerated data pipeline, enabling you to apply certain transformations 
# (like RandAugment and Random Erasing)
@pipeline_def(enable_conditionals=True) # pipeline_def decorator automatically injects batch_size, num_threads, and device_id as required keyword arguments for the pipeline
def kinetics_video_pipeline(filenames, labels, sequence_length=16, temporal_stride=1):
    # video reader & decoder (executes entirely on GPU), (T, H, W, C)
    video, label = fn.readers.video(
        device="gpu",
        filenames=filenames,
        labels=labels,
        sequence_length=sequence_length,
        stride=temporal_stride,
        random_shuffle=True,
        initial_fill = 256, # size of the buffer that is used for shuffling, pre-loads this many sequences into a buffer before shuffling begins, larger values gives better randomness but more memory
        pad_sequences=True, # if the video is shorter than the required clip length, it will pad by 0
        name="loader",
        step=10000,
        # additional_decode_surfaces=16, # decode surfaces are pre-allocated GPU memory buffers used to hold decoded video frames before they are processed by the rest of the pipeline
        # prefetch_queue_depth=16, # Specifies the number of batches to be prefetched by the internal Loader
        # read_ahead=True, # allows the reader to read and decode the next batch of video sequences while the current batch is being processed by the GPU, overlapping I/O and computation to improve throughput
    )

    # 2 repeated augmentation reptitions
    aug1 = apply_augmentations(video)
    aug2 = apply_augmentations(video)

    return aug1, aug2, label


def create_dali_loader(filenames, labels, batch_size, num_threads, device_id=0):
    # instantiate the loading pipeline
    pipe = kinetics_video_pipeline(
        filenames=filenames,
        labels=labels,
        batch_size=batch_size,
        num_threads=num_threads, # controls the size of the CPU thread pool dedicated to this pipeline, DALI still uses CPU threads for orchestrating instructions, try 4-8
        device_id=device_id, # index of the GPU
        # prefetch_queue_depth={"cpu_size": 8, "gpu_size": 8}, #  pipeline to use separated queues executor, with buffer queue size 4 for cpu stage and 8 for mixed and gpu stages
        prefetch_queue_depth=4,
    )
    # our setting for prefetch_queue_depth allows the CPU stage to buffer its results independently of the GPU stage, 
    # which is more effective at hiding the spiky stats. 
    
    # build the graph
    pipe.build()

    # wrap in PyTorch Iterator
    dali_loader = DALIGenericIterator(
        pipe,
        output_map=["aug1", "aug2", "label"], # output_map must match the order of variables returned in @pipeline_def
        reader_name="loader", # reader_name="loader" syncs the iterator's epoch size with fn.readers.video
        auto_reset=True, # automatically reset the iterator at the end of an epoch
        last_batch_policy=LastBatchPolicy.DROP
    )
    
    return dali_loader


def build_file_lists(csv_path, video_dir):
    # DALI needs a list of absolute paths and integer labels
    df = pd.read_csv(csv_path)
    paths  = [os.path.join(video_dir, p) for p in df['path']]
    labels = [STOI[l] for l in df['label']]
    return paths, labels


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


# Assuming dali_loader is created
# for i, data in enumerate(dali_loader):
#     # DALI outputs a list of dicts. If single GPU, grab index 0.
#     batch = data[0]
    
#     # Extract tensors (already on GPU)
#     v1 = batch["view1"] # Shape: (B, T, C, 224, 224)
#     v2 = batch["view2"] # Shape: (B, T, C, 224, 224)
#     y  = batch["label"].squeeze(-1) # Shape: (B,)
    

