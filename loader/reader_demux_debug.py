from loader.loader_dali import build_file_lists

csv_path = "/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv"
video_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/val"
webdataset_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/webdataset"

filenames, labels = build_file_lists(csv_path, video_dir)

import time
import numpy as np

# Simulate what the reader does: open, read all bytes, close
# t0 = time.time()
# total_bytes = 0
# for i in range(2000):
#     with open(filenames[i % len(filenames)], 'rb') as f:
#         data = f.read()
#         total_bytes += len(data)
# elapsed = time.time() - t0
# print(f"Pure read: {elapsed:.1f} files/sec, {total_bytes/elapsed/1e9:.2f} GB/s")

# Option A: Pure demux, no decode, Demuxing is reading and parsing the container to extract encoded video packets
import av  # pyav
import random

t0 = time.time()
for fn in filenames:
    container = av.open(fn)
    stream = container.streams.video[0]
    # seek to random point
    target = 0 # random.randint(0, stream.frames - 64)
    container.seek(target, stream=stream)
    packets = 0
    for packet in container.demux(stream): # A "packet" here is the compressed bitstream for one frame 
        packets += 1
        if packets > 20: break  # enough packets for 16 frames
    container.close()
print(f"Demux only: {(time.time()-t0):.1f} sec")


# -------------------------------------------------------------
# full pipeline (demux + NVDEC decode, no augmentation)
# -------------------------------------------------------------

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import torch

@pipeline_def
def minimal_video_pipe(filenames, labels, sequence_length=16, stride=1):
    video, label = fn.readers.video(
        device="gpu",
        filenames=filenames,
        labels=labels,
        sequence_length=sequence_length,
        stride=stride,
        step=10000,
        random_shuffle=False,
        initial_fill=1,
        pad_sequences=True,
        additional_decode_surfaces=8,
        read_ahead=True,
        prefetch_queue_depth=4,
        name="loader",
    )
    return video, label

BATCH_SIZE = 32  # tune as needed

pipe = minimal_video_pipe(
    filenames=filenames,
    labels=labels,
    batch_size=BATCH_SIZE,
    num_threads=4,
    device_id=0,
)
pipe.build()

# Warmup — first batch includes pipeline initialization overhead
_ = pipe.run()
torch.cuda.synchronize()

n_batches = len(filenames) // BATCH_SIZE
t0 = time.time()
for _ in range(n_batches):
    out = pipe.run()
torch.cuda.synchronize()
elapsed = time.time() - t0
total_samples = n_batches * BATCH_SIZE
print(f"DALI full pipeline: {elapsed:.2f} sec "
      f"({total_samples/elapsed:.1f} samples/sec, "
      f"{elapsed/total_samples*1000:.1f} ms/sample)")

# Demux only: 8.2 sec
# DALI full pipeline: 21.34 sec (93.0 samples/sec, 10.8 ms/sample)
# with stride=1, DALI full pipeline: 7.88 sec
# With stride=1, you're doing 4x less decode work per clip. 2.8x difference in terms of epoch time, dec % 6-8%
# This is actually a GOP-adjacent problem, You're paying for sequential 
# decoding through the P/B-frame chain from frame 0 to frame 60.
