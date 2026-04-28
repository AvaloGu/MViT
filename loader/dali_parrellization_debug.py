from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import torch

from loader_dali import build_file_lists

import time

csv_path = "/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv"
video_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/val"
webdataset_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/webdataset"

filenames, labels = build_file_lists(csv_path, video_dir)

@pipeline_def
def single_reader_pipe(filenames, labels):
    video, label = fn.readers.video( # combined read + demux + decode operation
        device="gpu",
        filenames=filenames,
        labels=labels,
        sequence_length=16,
        stride=4,
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

@pipeline_def
def multi_reader_pipe(filenames, labels, n_readers=2): # Run multiple reader instances in parallel
    outputs = []
    for i in range(n_readers):
        video, label = fn.readers.video( 
            device="gpu",
            filenames=filenames,
            labels=labels,
            sequence_length=16,
            stride=4,
            step=32,
            random_shuffle=False,
            initial_fill=1,
            pad_sequences=True,
            additional_decode_surfaces=8,
            read_ahead=True,
            prefetch_queue_depth=4,
            num_shards=n_readers,
            shard_id=i,
            name=f"loader_{i}",
        )
        outputs.append(video)
        outputs.append(label)
    return tuple(outputs)

# Benchmark both with same effective batch size
# Single reader: batch_size=32
# 2 readers: each produces batch_size=16, combined = 32

for n_readers in [8]:
    effective_batch = 32
    per_reader_batch = effective_batch // n_readers
    
    if n_readers == 1:
        pipe = single_reader_pipe(
            filenames=filenames, labels=labels,
            batch_size=effective_batch,
            num_threads=8, device_id=0,
        )
    else:
        pipe = multi_reader_pipe(
            filenames=filenames, labels=labels,
            n_readers=n_readers,
            batch_size=per_reader_batch,
            num_threads=8, device_id=0,
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

    
    # n_readers=1: 17.38s, 92.1 samples/sec (10.86 ms/sample), dec 12-13%
    # n_readers=2: 10.00s, 160.0 samples/sec (6.25 ms/sample), dec 21-25%
    # n_readers=4: 5.99s, 267.0 samples/sec (3.74 ms/sample), dec 35-44%

    # the bottleneck is per-reader serialization and the fix is parallelism.
    # each fn.readers.video instance is a single-threaded pipeline that serializes demux, 
    # per-file setup (MP4 parsing, seek, NVDEC context init), and decode submission, keeping the 
    # decoder idle most of the time between short decode bursts. 
    # Throughput is scaling almost linearly with reader count and decoder utilization is scaling proportionally, 
    # which is exactly the symptom of a pipeline that was submission-bound rather than hardware-bound, i.e. NVDEC had plenty of capacity, 
    # it just wasn't being fed fast enough by a single reader.

    # A single fn.readers.video is architected around a single file-reading thread
    # fn.readers.video has no built-in parallelization knob. Everything inside a single reader instance is 
    # serialized: one demux thread, one submission stream, one output queue. 
    # The parameters that sound like they might help (num_threads at pipeline level, additional_decode_surfaces, 
    # prefetch_queue_depth) don't parallelize the demux/submission path.
