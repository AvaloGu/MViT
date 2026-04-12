import os
import pandas as pd
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.auto_aug import rand_augment
from nvidia.dali import math as dali_math
from vocab import STOI 

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

import glob

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

# A DALI pipeline is basically a declarative computation graph. Instead of executing operations immediately, you describe a 
# data-processing workflow as a graph of nodes (DataNodes) connected by operators. Each node represents a tensor flowing through 
# the pipeline, and each fn.* call adds an operation (decode, resize, etc.) to that graph rather than running it right away. 
# When you call pipe.run(), DALI compiles and executes the whole graph efficiently—pipelining CPU/GPU work, parallelizing operations, 
# and prefetching data so it keeps the training loop fast.
# Every fn.* operation outputs a DataNode, which represents a TensorList in the pipeline graph (one tensor per sample in the batch)
# In other words, a DataNode doesn’t hold one tensor, it holds a whole batch. In DALI, a TensorList is basically a list of tensors, 
# one per sample in the batch, so if your batch size is 32, that DataNode represents 32 tensors flowing together through the graph. 
# Every operation (fn.*) processes the entire batch at once, and outputs another batch-shaped TensorList.

# this flag allows you to use functional control flow (like if statements and else blocks) directly 
# inside the GPU-accelerated data pipeline, enabling you to apply certain transformations 
# (like RandAugment and Random Erasing)
@pipeline_def(enable_conditionals=True) # pipeline_def decorator automatically injects batch_size, num_threads, and device_id as required keyword arguments for the pipeline
def kinetics_webdataset_pipeline(tar_files, index_files, sequence_length=16, temporal_stride=4):
    # read from WebDataset
    # DALI extracts the specific extensions into separate variables
    video_bytes, label_bytes, nframes_bytes = fn.readers.webdataset(
        paths=tar_files,
        index_paths=index_files,
        ext=["mp4", "lbl", "nframes"],
        random_shuffle=True,
        initial_fill=256,
        name="loader"
    )

    num_frames = fn.reinterpret(nframes_bytes, dtype=types.INT64, shape=[1])

    clip_length = sequence_length * temporal_stride # 64
    clip_len_tensor = fn.constant(idata=clip_length, dtype=types.INT64, device="cpu")

    max_start = dali_math.max(num_frames - clip_len_tensor, 0)

    # temporal sampling
    temp_rand_ratio = fn.random.uniform(range=[0.0, 1.0])
    start_frame = fn.cast(
        temp_rand_ratio * fn.cast(max_start, dtype=types.FLOAT),
        dtype=types.INT64
    )

    # Decode the video dynamically using NVDEC
    # device="mixed" means the CPU handles the byte-stream, but the GPU handles the decoding
    video = fn.decoders.video(
        video_bytes,
        device="mixed", 
        sequence_length=sequence_length,
        stride=temporal_stride,
        pad_mode='repeat', # Repeat the last valid frame for padding
        start_frame=start_frame,
    )
    
    # parse the label back to an INT32 tensor
    # fn.reinterpret casts the raw bytes we packed in Step 1 back into numbers
    label = fn.reinterpret(label_bytes, dtype=types.INT64, shape=[1])

    # apply GPU augmentations, 2 repeated augmentation repetition
    aug1 = apply_augmentations(video)
    aug2 = apply_augmentations(video)

    return aug1, aug2, label


def create_dali_loader(webdataset_dir, batch_size, num_threads, device_id=0):
    # gather the generated .tar and .idx files
    tar_files = sorted(glob.glob(os.path.join(webdataset_dir, "*.tar"))) # glob.glob returns a list of file paths matching the specified pattern
    index_files = sorted(glob.glob(os.path.join(webdataset_dir, "*.idx"))) # glob is used for pattern matching on file paths
    
    assert len(tar_files) > 0, "No .tar files found!"
    assert len(tar_files) == len(index_files), "Mismatch between .tar and .idx files!"

    # Instantiate the WebDataset pipeline
    pipe = kinetics_webdataset_pipeline(
        tar_files=tar_files,
        index_files=index_files,
        batch_size=batch_size,
        num_threads=num_threads, 
        device_id=device_id,
        prefetch_queue_depth={"cpu_size": 4, "gpu_size": 8}
    )
    
    pipe.build()

    # Wrap in PyTorch Iterator
    dali_loader = DALIGenericIterator(
        pipe,
        output_map=["aug1", "aug2", "label"],
        reader_name="loader",
        auto_reset=True
    )
    
    return dali_loader


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
    

