import pandas as pd
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.auto_aug import rand_augment
from vocab import STOI 

# DALI utilizes NVDEC (NVIDIA's hardware decoder) to read, decode, and sample the videos 
# directly on the GPU, completely bypassing the CPU overhead.

# one issue with fn.readers.video is it will aggregate and put all possible (16 frames 
# stride 4) non-overlapping sequences in a bucket, shuffle it if random_shuffle, 
# and draw from that bucket. So we can get multiple examples from the same
# video clip and the epoch size will be a lot larger than what we expect.
# This won't match the kinetics epoch logic by itself.

# the argument 'step' defaults to the temporal span of the clip (in our case, 16×4=64 frames). 
# For a standard 10-second, 300-frame Kinetics video, DALI will automatically extract ~4 sequential, 
# non-overlapping clips per video. random_shuffle=True just tosses all of these generated clips into 
# a shuffle bucket.

# this flag allows you to use functional control flow (like if statements and else blocks) directly 
# inside the GPU-accelerated data pipeline, enabling you to apply certain transformations 
# (like RandAugment and Random Erasing)
@pipeline_def(enable_conditionals=True)
def kinetics_video_pipeline(filenames, labels, sequence_length=16, temporal_stride=4):
    # video reader & decoder (executes entirely on GPU)
    video, label = fn.readers.video(
        device="gpu",
        filenames=filenames,
        labels=labels,
        sequence_length=sequence_length,
        stride=temporal_stride,
        random_shuffle=True,
        pad_last_frame=True, # if the video is shorter than the required clip length, it will pad by repeating the last frame
        name="Reader"
    )

    # random Resized Crop
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
        # rand_augment requires knowing the spatial shape dynamically
        shape = fn.peek_image_shape(video) 
        video = rand_augment.rand_augment(video, shape=shape, n=4, m=7)

    # (1/255.0) scaling and normalization in one step.
    # permute to FCHW which aligns with PyTorch's (T, C, H, W) expectation.
    video = fn.crop_mirror_normalize(
        video,
        dtype=types.FLOAT,
        output_layout="FCHW",
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
        
        video = fn.erasing(
            video,
            anchor=anchor,
            shape=shape,
            normalized=True, # the anchor and shape are (ratios) normalized to [0.0, 1.0], rather than absolute pixel counts
            fill_value=0.0 # Will fill with 0, black
        )

    return video, label