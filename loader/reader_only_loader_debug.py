import os
import time
import pandas as pd
import numpy as np
import torch

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from vocab import STOI  


PACK_STRIDE = 10000


def build_file_lists(csv_path, video_dir, num_frames_col="num_frames"):
    df = pd.read_csv(csv_path)
    paths = [os.path.join(video_dir, p) for p in df["path"]]
    labels = [STOI[l] for l in df["label"]]
    num_frames = df[num_frames_col].astype(int).tolist()
    packed = [nf * PACK_STRIDE + lbl for nf, lbl in zip(num_frames, labels)]
    return paths, packed


@pipeline_def
def bare_file_reader_pipe(filenames, packed_labels):
    """Just read MP4 bytes — nothing else.
    
    This isolates fn.readers.file's throughput from decode, demux,
    augmentation, and everything downstream.
    """
    video_bytes, packed = fn.readers.file(
        files=filenames,
        labels=packed_labels,
        random_shuffle=True,
        initial_fill=256,
        read_ahead=True,
        prefetch_queue_depth=4,
        name="file_reader",
    )
    return video_bytes, packed


@pipeline_def
def bare_file_reader_multi_pipe(filenames, packed_labels, n_readers=4):
    """Same as above but with N parallel readers, testing if reading scales."""
    outputs = []
    for i in range(n_readers):
        video_bytes, packed = fn.readers.file(
            files=filenames,
            labels=packed_labels,
            num_shards=n_readers,
            shard_id=i,
            random_shuffle=True,
            initial_fill=256,
            read_ahead=True,
            prefetch_queue_depth=4,
            name=f"file_reader_{i}",
        )
        outputs.append(video_bytes)
        outputs.append(packed)
    return tuple(outputs)


def benchmark(pipe, label, batch_size, n_iters=50, warmup=5):
    pipe.build()
    
    # Warmup
    for _ in range(warmup):
        pipe.run()
    torch.cuda.synchronize()
    
    # Measure
    t0 = time.time()
    total_bytes = 0
    for _ in range(n_iters):
        outputs = pipe.run()
        # Count bytes read — inspect the first output (video_bytes)
        # outputs[0] is a TensorListCPU of byte tensors
        try:
            for tensor in outputs[0]:
                total_bytes += tensor.shape()[0]  # each byte tensor is 1D
        except Exception:
            pass  # skip byte counting if inspection fails
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    total_samples = n_iters * batch_size
    throughput_samples = total_samples / elapsed
    throughput_mbps = (total_bytes / elapsed) / 1e6 if total_bytes else 0
    
    print(f"{label}:")
    print(f"  {elapsed:.2f}s for {total_samples} samples "
          f"({throughput_samples:.1f} samples/sec, "
          f"{elapsed/total_samples*1000:.2f} ms/sample)")
    if total_bytes:
        print(f"  Total bytes read: {total_bytes/1e9:.2f} GB "
              f"({throughput_mbps:.1f} MB/s)")
    print()


if __name__ == "__main__":
    paths, packed = build_file_lists(
        csv_path="/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv",
        video_dir="/blue/uf-dsi/rongguan.gu/MViT/k400/val",
    )
    
    BATCH_SIZE = 32
    
    # Test 1: Single reader
    pipe = bare_file_reader_pipe(
        filenames=paths,
        packed_labels=packed,
        batch_size=BATCH_SIZE,
        num_threads=8,
        device_id=0,
    )
    benchmark(pipe, "Single reader (bytes only)", BATCH_SIZE)
    
    # Test 2: Multi reader (4 parallel)
    pipe = bare_file_reader_multi_pipe(
        filenames=paths,
        packed_labels=packed,
        n_readers=4,
        batch_size=BATCH_SIZE // 4,  # each reader produces 1/4 batch
        num_threads=8,
        device_id=0,
    )
    benchmark(pipe, "4 parallel readers (bytes only)", BATCH_SIZE)

    # The bottle neck is not reader, reader is fast. 
    # Something is just wrong with fn.decoders.video, much slower, maybe cuz its stateless 
    # fn.readers.video is a stateful operation, it keeps file handles open, maintains NVDEC contexts
    # although it seems like fn.readers.video also demux on cpu and decode on gpu, it proabably fused 
    # operations and maintained a single integrated op that keeps state. 
    # Unlike the decoupled path (fn.readers.file + fn.decoders.video) 
    
    # Single reader (bytes only):
    # 1.20s for 1600 samples (1335.0 samples/sec, 0.75 ms/sample)
    # Total bytes read: 2.49 GB (2081.2 MB/s)

    # 4 parallel readers (bytes only):
    # 0.52s for 1600 samples (3092.7 samples/sec, 0.32 ms/sample)
    # Total bytes read: 0.66 GB (1270.6 MB/s)