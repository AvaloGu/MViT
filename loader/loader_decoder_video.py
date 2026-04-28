import os
import torch
import numpy as np
import pandas as pd
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dali_math
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.auto_aug import rand_augment
from vocab import STOI


def decode_clip_with_temporal_sampling(video_bytes, num_frames, 
                                        sequence_length=16, temporal_stride=4):
    """Random temporal sampling + NVDEC decode. Returns (T, H, W, C) on GPU."""
    clip_length = sequence_length * temporal_stride
    clip_len_tensor = fn.constant(idata=clip_length, dtype=types.INT64, device="cpu")
    
    # max_start = max(num_frames - clip_length, 0)
    # max_start = dali_math.max(num_frames - clip_len_tensor, 0)
    
    # # uniform in [0, max_start]
    # temp_rand = fn.random.uniform(range=[0.0, 1.0])
    # start_frame = fn.cast(
    #     temp_rand * fn.cast(max_start, dtype=types.FLOAT),
    #     dtype=types.INT32,
    # )
    # start_frame = fn.squeeze(start_frame, axes=[0])
    
    video = fn.decoders.video(
        video_bytes,
        device="mixed",
        sequence_length=sequence_length,
        stride=temporal_stride,
        start_frame=0,
        pad_mode="repeat",
    )
    return video


def apply_augmentations(video):
    video = fn.random_resized_crop(
        video, size=[224, 224],
        random_area=[0.08, 1.0],
        random_aspect_ratio=[0.75, 1.333]
    )
    coin_flip = fn.random.coin_flip(probability=0.5)
    video = fn.flip(video, horizontal=coin_flip)
    
    apply_ra = fn.random.coin_flip(probability=0.5, dtype=types.BOOL)
    if apply_ra:
        video = rand_augment.rand_augment(video, shape=[224, 224], n=4, m=7)
    
    video = fn.crop_mirror_normalize(
        video, dtype=types.FLOAT, output_layout="FHWC",
        mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    )
    
    apply_erasing = fn.random.coin_flip(probability=0.25, dtype=types.BOOL)
    if apply_erasing:
        anchor = fn.random.uniform(range=[0.0, 1.0], shape=[2])
        shape = fn.random.uniform(range=[0.02, 0.33], shape=[2])
        video = fn.erase(
            video, anchor=anchor, shape=shape,
            axes=[1, 2], normalized=True, fill_value=0.0
        )
    
    video = fn.transpose(video, perm=[0, 3, 1, 2])
    return video


PACK_STRIDE = 10000  # must exceed max label; labels are 0-399, work around that fn.readers.file only accpets integer labels


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
    # coin_flip = fn.random.coin_flip(probability=0.5)
    # video = fn.flip(video, horizontal=coin_flip)

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


@pipeline_def(enable_conditionals=True)
def kinetics_multi_reader_pipeline(
    filenames, labels_plus_nframes,
    n_readers=4,
    sequence_length=16, temporal_stride=4,
):
    aug1_outputs, aug2_outputs, label_outputs = [], [], []
    
    for i in range(n_readers):
        # Each reader gets its own shard of files
        # labels_plus_nframes is shape (N, 2): [class_label, num_frames] per sample
        video_bytes, combined_label = fn.readers.file(
            files=filenames,
            labels=labels_plus_nframes,
            num_shards=n_readers,
            shard_id=i,
            random_shuffle=True,
            initial_fill=256,
            read_ahead=True,
            prefetch_queue_depth=4,
            name=f"file_reader_{i}",
        )
        
        # Unpack: packed = num_frames * PACK_STRIDE + label
        # combined_label is int32 scalar (shape (1,) per sample in batch)
        # integer division gives num_frames, modulo gives label
        num_frames_float = fn.cast(combined_label, dtype=types.FLOAT) / PACK_STRIDE
        num_frames = fn.cast(num_frames_float, dtype=types.INT64)  # truncation = floor division
        class_label = combined_label - num_frames * PACK_STRIDE
        
        # Each branch has its own decoder — this is what parallelizes NVDEC
        video = decode_clip_with_temporal_sampling(
            video_bytes, num_frames,
            sequence_length=sequence_length,
            temporal_stride=temporal_stride,
        )
        
        # Two augmented views per sample (repeated aug)
        aug1_outputs.append(simple_augmentations(video))
        aug2_outputs.append(simple_augmentations(video))
        label_outputs.append(class_label)
    
    # Return flat tuple; concatenate on PyTorch side
    return (*aug1_outputs, *aug2_outputs, *label_outputs)


def _probe_num_frames(path):
    """
    Read the frame count from a video container. Tries decord (fast, no full
    decode), then falls back to opencv. Raises if neither is available.
    """
    # try:
    #     import decord
    #     vr = decord.VideoReader(path, num_threads=1)
    #     return len(vr)
    # except ImportError:
    #     pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    except ImportError as e:
        raise RuntimeError(
            "Need `decord` or `opencv-python` to probe frame counts. "
            "Alternatively, add a `num_frames` column to your CSV."
        ) from e


def build_file_lists(csv_path, video_dir, num_frames_col="num_frames"):
    df = pd.read_csv(csv_path)
    paths = [os.path.join(video_dir, p) for p in df["path"]]
    labels = [STOI[l] for l in df["label"]]
    
    if num_frames_col in df.columns:
        num_frames = df[num_frames_col].astype(int).tolist()
    else:
        num_frames = [_probe_num_frames(p) for p in paths]
        df["num_frames"] = num_frames
        df.to_csv(csv_path, index=False)
    
    # Pack num_frames and label into a single int
    # Labels 0-399, so stride of 10000 is safe
    # This is a work around that fn.readers.file only accpets list of integer as labels
    packed = [nf * PACK_STRIDE + lbl for nf, lbl in zip(num_frames, labels)]
    
    # Sanity check
    assert max(labels) < PACK_STRIDE, \
        f"Label {max(labels)} >= PACK_STRIDE {PACK_STRIDE}; pick a larger stride"
    
    return paths, packed


def create_dali_loader(paths, labels_plus_nframes, batch_size, num_threads,
                       n_readers=4, device_id=0):
    # Each reader produces batch_size/n_readers samples; combined gives full batch
    assert batch_size % n_readers == 0, \
        f"batch_size ({batch_size}) must be divisible by n_readers ({n_readers})"
    per_reader_batch = batch_size // n_readers
    
    pipe = kinetics_multi_reader_pipeline(
        filenames=paths,
        labels_plus_nframes=labels_plus_nframes,
        n_readers=n_readers,
        batch_size=per_reader_batch,
        num_threads=num_threads,
        device_id=device_id,
        # prefetch_queue_depth={"cpu_size": 4, "gpu_size": 4},
    )
    pipe.build()
    
    # Output map: aug1_0..aug1_{N-1}, aug2_0..aug2_{N-1}, label_0..label_{N-1}
    output_map = (
        [f"aug1_{i}" for i in range(n_readers)] +
        [f"aug2_{i}" for i in range(n_readers)] +
        [f"label_{i}" for i in range(n_readers)]
    )
    
    dali_loader = DALIGenericIterator(
        pipe,
        output_map=output_map,
        reader_name="file_reader_0",  # sync to first reader's epoch
        auto_reset=True,
    )
    
    return dali_loader, n_readers


def combine_multi_reader_batch(data, n_readers):
    """Concatenate outputs from N readers into a single batch."""
    import torch
    batch = data[0]
    aug1 = torch.cat([batch[f"aug1_{i}"] for i in range(n_readers)], dim=0)
    aug2 = torch.cat([batch[f"aug2_{i}"] for i in range(n_readers)], dim=0)
    labels = torch.cat([batch[f"label_{i}"] for i in range(n_readers)], dim=0).squeeze(-1)
    return aug1, aug2, labels


# test the speed of the loader

import time
csv_path = "/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv"
video_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/val"

paths, labels_plus_nframes = build_file_lists(csv_path=csv_path, video_dir=video_dir)

for n_readers in [1, 2, 4]:
    effective_batch = 32
    per_reader_batch = effective_batch // n_readers
    
    pipe = kinetics_multi_reader_pipeline(
        filenames=paths,
        labels_plus_nframes=labels_plus_nframes,
        n_readers=n_readers,
        batch_size=per_reader_batch,
        num_threads=16,
        device_id=0,
        # prefetch_queue_depth={"cpu_size": 4, "gpu_size": 4},
    )
    
    pipe.build()
    _ = pipe.run()  # warmup
    torch.cuda.synchronize()
    
    n_iters = 50
    t0 = time.time()
    for _ in range(n_iters):
        pipe.run()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    total_samples = n_iters * effective_batch
    print(f"n_readers={n_readers}: {elapsed:.2f}s, "
          f"{total_samples/elapsed:.1f} samples/sec "
          f"({elapsed/total_samples*1000:.2f} ms/sample)")

    # fn.decoders.video is unusable
    # n_readers=1: 106.85s, 15.0 samples/sec (66.78 ms/sample)
    # n_readers=2: 110.99s, 14.4 samples/sec (69.37 ms/sample)
    # n_readers=4: 111.74s, 14.3 samples/sec (69.83 ms/sample)

    # on cpu
    # n_readers=1: 54.48s, 29.4 samples/sec (34.05 ms/sample)
    # n_readers=2: 69.38s, 23.1 samples/sec (43.36 ms/sample)
    # n_readers=4: 108.10s, 14.8 samples/sec (67.56 ms/sample)