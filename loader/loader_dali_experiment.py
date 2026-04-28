import os
import numpy as np
import pandas as pd
import nvidia.dali.fn as fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.auto_aug import rand_augment
from vocab import STOI

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# DALI utilizes NVDEC (NVIDIA's hardware decoder) to decode the videos
# directly on the GPU. With external_source + parallel=True, the raw file
# I/O happens in Python worker processes (py_num_workers), completely
# decoupling disk reads from the GPU decode/augment pipeline.


# ---------------------------------------------------------------------------
# External source callable
# ---------------------------------------------------------------------------
# DALI's external_source with parallel=True requires a picklable callable that
# accepts a SampleInfo and returns the per-sample tensors. Each call runs in a
# separate Python worker process (py_num_workers), so pure-Python work (file
# I/O, numpy) scales across cores without GIL contention.
#
# We return three arrays per sample:
#   - raw video bytes (uint8)  -> fed to fn.decoders.video (device="mixed")
#   - num_frames (int64)       -> used in-graph to compute a random start_frame
#   - label (int32)            -> classification target
#
# num_frames is probed once up front (see build_file_lists) to avoid re-opening
# the container on every worker call. If your CSV already has a frame count
# column, pass it in directly and skip the probe.
class VideoFileSource:
    def __init__(self, filenames, labels, num_frames, batch_size, shuffle=True, seed=42):
        assert len(filenames) == len(labels) == len(num_frames)
        self.filenames = list(filenames)
        self.labels = np.asarray(labels, dtype=np.int32)
        self.num_frames = np.asarray(num_frames, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n = len(self.filenames)
        # full epoch length in samples; DALI will call us (epoch_size) times
        # per epoch when configured with size=self.n in external_source.
        self.epoch_size = self.n

    def __call__(self, sample_info):
        # sample_info.idx_in_epoch is the sample index within the current epoch
        # sample_info.epoch_idx is the current epoch number (useful for shuffling)
        if sample_info.iteration >= self.epoch_size // self.batch_size:
            # signal end of epoch; DALIGenericIterator's auto_reset handles this
            raise StopIteration

        # per-epoch deterministic shuffle: every worker computes the same
        # permutation because it's a pure function of (seed, epoch_idx)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + sample_info.epoch_idx)
            perm = rng.permutation(self.n)
            real_idx = perm[sample_info.idx_in_epoch]
        else:
            real_idx = sample_info.idx_in_epoch

        path = self.filenames[real_idx]
        with open(path, "rb") as f:
            video_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        # external_source expects numpy arrays; scalars must be 0-d or 1-d arrays
        num_frames = np.asarray(self.num_frames[real_idx], dtype=np.int64)
        label = np.asarray(self.labels[real_idx], dtype=np.int32)
        return video_bytes, num_frames, label


# ---------------------------------------------------------------------------
# GPU augmentation stack
# ---------------------------------------------------------------------------
def apply_augmentations(video):
    # video data aug on gpu, cpu generates a few random numbers
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
        # top-left corner of the erase rectangle, uniformly sampled
        anchor = fn.random.uniform(range=[0.0, 1.0], shape=[2])
        # size of the erased region: 2%-33% of the image
        shape = fn.random.uniform(range=[0.02, 0.33], shape=[2])

        video = fn.erase(
            video,
            anchor=anchor,
            shape=shape,
            axes=[1, 2],        # erase in H and W
            normalized=True,    # anchor/shape are ratios in [0, 1]
            fill_value=0.0
        )

    # permute (T, H, W, C) -> (T, C, H, W) to match PyTorch's expectation
    video = fn.transpose(video, perm=[0, 3, 1, 2])
    return video


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
# enable_conditionals=True lets us use Python `if` on DALI booleans inside the
# graph (needed for the RandAugment / Random Erasing probability gates).
@pipeline_def(enable_conditionals=True)
def kinetics_video_pipeline(
    source,
    sequence_length=16,
    temporal_stride=4,
    py_num_workers_enabled=True,
):
    # external_source runs `source` in parallel Python worker processes when
    # parallel=True. The pipeline's py_num_workers argument (set in
    # create_dali_loader) controls how many workers spin up.
    video_bytes, num_frames, label = fn.external_source(
        source=source,
        num_outputs=3,
        batch=False,           # source returns one sample per call
        parallel=py_num_workers_enabled,
        dtype=[types.UINT8, types.INT64, types.INT32],
        layout=["", "", ""],   # scalars/1-D, no named layout
    )

    # ---- temporal sampling: pick a random start frame in-graph ----
    clip_length = sequence_length * temporal_stride  # 64 by default
    clip_len_tensor = fn.constant(idata=clip_length, dtype=types.INT64, device="cpu")

    # max_start = max(num_frames - clip_length, 0); clamps short clips to 0
    # so decoders.video's pad_mode='repeat' can handle the tail.
    max_start = dali_math.max(num_frames - clip_len_tensor, 0)

    # uniform ratio in [0, 1) -> integer start frame in [0, max_start]
    temp_rand_ratio = fn.random.uniform(range=[0.0, 1.0])
    start_frame = fn.cast(
        temp_rand_ratio * fn.cast(max_start, dtype=types.FLOAT),
        dtype=types.INT32,
    )
    # fn.random.uniform returns shape [1]; decoders.video wants a scalar
    start_frame = fn.squeeze(start_frame, axes=[0])

    # ---- decode on GPU via NVDEC ----
    # device="mixed": CPU demuxes the container, GPU runs NVDEC.
    # Output layout is (T, H, W, C).
    video = fn.decoders.video(
        video_bytes,
        device="mixed",
        sequence_length=sequence_length,
        stride=temporal_stride,
        start_frame=start_frame,
        pad_mode="repeat",     # repeat last valid frame if the clip is short
    )

    # two augmented views for repeated augmentation
    aug1 = apply_augmentations(video)
    aug2 = apply_augmentations(video)

    # labels need to be on GPU if you want .to(device) to be a no-op downstream
    return aug1, aug2, label.gpu()


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------
def create_dali_loader(
    filenames,
    labels,
    num_frames,
    batch_size,
    num_threads=4,
    py_num_workers=4,
    device_id=0,
    sequence_length=16,
    temporal_stride=4,
    shuffle=True,
    seed=42,
):
    """
    Args:
        filenames:      list of absolute video paths
        labels:         list of integer class labels
        num_frames:     list of frame counts (len == len(filenames))
        batch_size:     samples per batch
        num_threads:    DALI CPU thread pool (pipeline orchestration), try 4-8
        py_num_workers: Python worker processes for external_source I/O.
                        Set to 0 to run in-process (single-threaded, for debug).
        device_id:      GPU index
    """
    source = VideoFileSource(
        filenames=filenames,
        labels=labels,
        num_frames=num_frames,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    # py_num_workers > 0 requires parallel=True in external_source
    parallel_source = py_num_workers > 0

    pipe = kinetics_video_pipeline(
        source=source,
        sequence_length=sequence_length,
        temporal_stride=temporal_stride,
        py_num_workers_enabled=parallel_source,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        py_num_workers=py_num_workers,
        # py_start_method="spawn" is safer with CUDA already initialized in the
        # parent; "fork" is faster but can deadlock if CUDA was touched pre-fork.
        py_start_method="spawn",
        prefetch_queue_depth=4,
    )

    # build() is called implicitly on first run in recent DALI, but calling it
    # here surfaces pipeline errors at loader-construction time.
    pipe.build()

    dali_loader = DALIGenericIterator(
        pipe,
        output_map=["aug1", "aug2", "label"],
        size=source.epoch_size,  # tell the iterator how long an epoch is
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP,  # or PARTIAL / FILL
    )
    return dali_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def build_file_lists(csv_path, video_dir, num_frames_col=None):
    """
    DALI needs a list of absolute paths + integer labels + frame counts.

    If your CSV has a precomputed frame count column, pass its name as
    `num_frames_col` and we'll use it directly (much faster than probing).
    Otherwise we probe each file once up front.
    """
    df = pd.read_csv(csv_path)
    paths = [os.path.join(video_dir, p) for p in df["path"]]
    labels = [STOI[l] for l in df["label"]]

    if num_frames_col is not None and num_frames_col in df.columns:
        num_frames = df[num_frames_col].astype(int).tolist()
    else:
        # one-time probe; consider caching this to disk for large datasets
        num_frames = [_probe_num_frames(p) for p in paths]

    return paths, labels, num_frames


# ---------------------------------------------------------------------------
# MixUp / CutMix (unchanged)
# ---------------------------------------------------------------------------
class MixupCutmixAugmentation:
    def __init__(self, num_classes=400):
        self.mixup = v2.MixUp(alpha=0.8, num_classes=num_classes)
        self.cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)

    def augment(self, videos, labels):
        # videos: (B, T, C, H, W)
        # labels: (B,)
        B = videos.shape[0]
        half = B // 2

        # slice the batch first, then wrap as Video (slicing drops the wrapper)
        first_half_videos = tv_tensors.Video(videos[:half])
        second_half_videos = tv_tensors.Video(videos[half:])

        videos_m, labels_m = self.mixup(first_half_videos, labels[:half])
        videos_c, labels_c = self.cutmix(second_half_videos, labels[half:])

        mixed_videos = torch.cat([videos_m, videos_c], dim=0)
        mixed_labels = torch.cat([labels_m, labels_c], dim=0)
        return mixed_videos, mixed_labels


# Example usage:
# paths, labels, num_frames = build_file_lists(csv_path, video_dir,
#                                              num_frames_col="num_frames")
# loader = create_dali_loader(paths, labels, num_frames,
#                             batch_size=32, py_num_workers=6)
# for data in loader:
#     batch = data[0]
#     v1 = batch["aug1"]                      # (B, T, C, 224, 224)
#     v2 = batch["aug2"]                      # (B, T, C, 224, 224)
#     y  = batch["label"].squeeze(-1).long()  # (B,)