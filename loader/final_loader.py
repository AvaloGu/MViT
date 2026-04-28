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

# fn.readers.video handles file reading and MP4 demux on CPU (internally), 
# then hands the bitstream to NVDEC for GPU decode. The output tensor lives on GPU.
# Multiple reader instances run in parallel to distribute work across NVDEC engines.

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



def simple_augmentations(video):
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
    # apply_ra = fn.random.coin_flip(probability=0.5, dtype=types.BOOL)
    # if apply_ra:
    #     video = rand_augment.rand_augment(video, shape=[224, 224], n=4, m=7)

     # (1/255.0) scaling and normalization in one step.
    # video = fn.crop_mirror_normalize(
    #     video,
    #     dtype=types.FLOAT,
    #     output_layout="FHWC",
    #     mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
    #     std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    # )

    # random erasing (p=0.25)
    # apply_erasing = fn.random.coin_flip(probability=0.25, dtype=types.BOOL)
    # if apply_erasing:

    #     # top left corner of the erasure rectangle is uniformly sampled from 
    #     # the image
    #     # shape=[2] means we want to sample 2 values (for x and y coordinates)
    #     anchor = fn.random.uniform(range=[0.0, 1.0], shape=[2])

    #     # size of the erased region is between 2% to 33% of the original image
    #     shape  = fn.random.uniform(range=[0.02, 0.33], shape=[2])
        
    #     video = fn.erase(
    #         video,
    #         anchor=anchor,
    #         shape=shape,
    #         axes=[1, 2], # erase in the H and W dimensions
    #         normalized=True, # the anchor and shape are (ratios) normalized to [0.0, 1.0], rather than absolute pixel counts
    #         fill_value=0.0 # Will fill with 0, black
    #     )

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
def kinetics_video_pipeline(filenames, labels, sequence_length=16, temporal_stride=4, n_readers=4):
    # video reader & decoder (executes entirely on GPU), (T, H, W, C)
    aug1_outputs, aug2_outputs, label_outputs = [], [], []

    # for loop is non-blocking, DALI pipeline sees graph, not for loop
    for i in range(n_readers): # Run multiple reader instances in parallel, utilize the available dec cores
        video, label = fn.readers.video(
            device="gpu",
            filenames=filenames,
            labels=labels,
            sequence_length=sequence_length,
            stride=temporal_stride,
            random_shuffle=True,
            initial_fill = 256, # size of the buffer that is used for shuffling, pre-loads this many sequences into a buffer before shuffling begins, larger values gives better randomness but more memory
            pad_sequences=True, # if the video is shorter than the required clip length, it will pad by 0
            step=32,
            additional_decode_surfaces=8, # decode surfaces are pre-allocated GPU memory buffers used to hold decoded video frames before they are processed by the rest of the pipeline
            prefetch_queue_depth=8, # Specifies the number of batches to be prefetched by the internal Loader
            read_ahead=True, # allows the reader to read and decode the next batch of video sequences while the current batch is being processed by the GPU, overlapping I/O and computation to improve throughput
            num_shards=n_readers,
            shard_id=i, 
            name=f"loader_{i}",
        )

        # 2 repeated augmentation reptitions
        aug1_outputs.append(simple_augmentations(video))
        aug2_outputs.append(simple_augmentations(video))
        # aug1_outputs.append(video)
        # aug2_outputs.append(video)
        label_outputs.append(label)

    # return flat tuple, concatenate on PyTorch side
    return (*aug1_outputs, *aug2_outputs, *label_outputs)


def create_dali_loader(filenames, labels, batch_size, num_threads, n_readers=4, device_id=0):
    # instantiate the loading pipeline

    # each reader produces batch_size/n_readers samples; combined gives full batch
    assert batch_size % n_readers == 0, \
        f"batch_size ({batch_size}) must be divisible by n_readers ({n_readers})"
    per_reader_batch = batch_size // n_readers

    pipe = kinetics_video_pipeline(
        filenames=filenames,
        labels=labels,
        n_readers=n_readers,
        batch_size=per_reader_batch,
        num_threads=num_threads, # controls the size of the CPU thread pool dedicated to this pipeline, DALI still uses CPU threads for orchestrating instructions, try 4-8
        device_id=device_id, # index of the GPU
        # prefetch_queue_depth={"cpu_size": 8, "gpu_size": 8}, #  pipeline to use separated queues executor, with buffer queue size 4 for cpu stage and 8 for mixed and gpu stages
        prefetch_queue_depth=8,
    )
    # our setting for prefetch_queue_depth allows the CPU stage to buffer its results independently of the GPU stage, 
    # which is more effective at hiding the spiky stats. 
    
    # build the graph
    pipe.build()

    output_map = (
        [f"aug1_{i}" for i in range(n_readers)] +
        [f"aug2_{i}" for i in range(n_readers)] +
        [f"label_{i}" for i in range(n_readers)]
    )

    # wrap in PyTorch Iterator
    dali_loader = DALIGenericIterator(
        pipe,
        output_map=output_map, # output_map must match the order of variables returned in @pipeline_def
        reader_name="loader_0", # reader_name="loader" syncs the iterator's epoch size with the first fn.readers.video
        auto_reset=True, # automatically reset the iterator at the end of an epoch
        last_batch_policy=LastBatchPolicy.DROP
    )
    
    return dali_loader


def combine_multi_reader_batch(data, n_readers):
    """Concatenate outputs from N readers into a single batch."""
    import torch
    batch = data[0]
    aug1 = torch.cat([batch[f"aug1_{i}"] for i in range(n_readers)], dim=0)
    aug2 = torch.cat([batch[f"aug2_{i}"] for i in range(n_readers)], dim=0)
    labels = torch.cat([batch[f"label_{i}"] for i in range(n_readers)], dim=0).squeeze(-1).long()
    return aug1, aug2, labels


def build_file_lists(csv_path, video_dir):
    # DALI needs a list of absolute paths and integer labels
    df = pd.read_csv(csv_path)
    paths  = [os.path.join(video_dir, p) for p in df['path']]
    labels = [STOI[l] for l in df['label']]
    return paths, labels

# clip level spatial transformation
clip_spatial_transform = v2.Compose([
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


def clip_spatial_aug(x):
    x = tv_tensors.Video(x)
    x = clip_spatial_transform(x) # augment on gpu

    return x


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
    


# test the speed of the loader

# the DALI pipeline output — TensorListGPU objects, 
# DALIGenericIterator Wraps each output in a PyTorch tensor (requires CUDA stream synchronization). 
# wraps 12 outputs (4 aug1 + 4 aug2 + 4 labels) into dict form every iteration

# import time
# csv_path = "/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv"
# video_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/val"

# paths, labels = build_file_lists(csv_path=csv_path, video_dir=video_dir)
# train_loader = create_dali_loader(filenames=paths, labels=labels, batch_size=32, num_threads=14, n_readers=16, device_id=0)


# t0 = time.time()

# for i, data in enumerate(train_loader): 

#     # aug1, aug2, label = combine_multi_reader_batch(data, 4) # (B/2, T, C, 224, 224), (B/2,)

#     if i >= 50:
#         break

# torch.cuda.synchronize()
# elapsed = time.time() - t0

# print(f"n_readers={8}: {elapsed:.2f}s")

# test the speed without DALI iterator
# for n_readers in [16]:
#     effective_batch = 32
#     per_reader_batch = effective_batch // n_readers
    
#     pipe = kinetics_video_pipeline(
#         filenames=paths,
#         labels=labels,
#         n_readers=n_readers,
#         batch_size=per_reader_batch,
#         num_threads=16,
#         device_id=0,
#     )
    
#     pipe.build()
    
#     n_iters = 50
#     t0 = time.time()
#     for _ in range(n_iters):
#         pipe.run()
#     torch.cuda.synchronize()
#     elapsed = time.time() - t0
    
#     total_samples = n_iters * effective_batch
#     print(f"n_readers={n_readers}: {elapsed:.2f}s, "
#           f"{total_samples/elapsed:.1f} samples/sec "
#           f"({elapsed/total_samples*1000:.2f} ms/sample)")


# DALIGenericIterator.__next__() does something like:
# Call pipe.run() to get outputs
# Wait for GPU to finish (sync point to safely wrap in PyTorch tensors)
# Wrap outputs in PyTorch tensors
# Build output dict
# Return to user

# the key issue with this loader is not the DALI Iterator, but with heavy data augmentations.
# if we spawn 8 parallel loadering pipeline, the augmentations will become extremely slow. 
# Each time augmentation is performed on 4 samples, and there are 8 of those. The stalls the pipeline
# specifically rand_augment is extremely slow.
