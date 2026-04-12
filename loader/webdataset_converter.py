import os
import cv2  # pip install opencv-python
import pandas as pd
import webdataset as wds
import numpy as np
from vocab import STOI

def convert_to_webdataset(csv_path, video_dir, output_dir, prefix="kinetics", max_size=1e9):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # wds.ShardWriter handles splitting into multiple .tar files (e.g., 1GB each, about 1e9 bytes)
    # pattern we use as filename template where %06d is replaced by a zero-padded integer
    # so shard files will look like data-000000.tar, data-000001.tar, etc.
    pattern = os.path.join(output_dir, f"{prefix}-%06d.tar") 
    
    with wds.ShardWriter(pattern, maxsize=max_size) as sink:
        # open webdataset shard writer, maxsize controls how large each shard file can grow before a new one is started
        for idx, row in df.iterrows():
            video_path = os.path.join(video_dir, row['path'])
            label_int = STOI[row['label']]
            
            # 1. Read raw video bytes
            with open(video_path, "rb") as f: # read binary
                video_bytes = f.read()

            # 2. Extract frame count via OpenCV
            cap = cv2.VideoCapture(video_path) # VideoCapture can read video metadata without loading the whole video into memory
            # cap.get() is a method of the VideoCapture object in OpenCV that lets you query properties of the video
            # CAP_PROP_FRAME_COUNT is a constant
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # 3. Convert label to raw int32 bytes for easy DALI parsing later
            label_bytes = np.int64(label_int).tobytes()
            nframes_bytes = np.int64(nframes).tobytes()
            
            # 4. Write to the tar file (extensions dictate how we extract them later)
            sample = {
                "__key__": f"{idx:08d}",
                "mp4": video_bytes,
                "lbl": label_bytes, # custom extension for the label bytes
                "nframes": nframes_bytes
            }
            sink.write(sample)
            
    print("WebDataset generation complete!")

# Run this once:
if __name__ == "__main__":
    csv_path = "/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv"
    video_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/val"
    output_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/webdataset"
    convert_to_webdataset(csv_path, video_dir, output_dir)

    # Because DALI needs to randomly shuffle the data across epochs, it requires index files to map out where each sample lives inside the .tar archives.
    # DALI provides a built-in command-line tool for this called wds2idx. Run this in your terminal inside the directory containing your new .tar files
    # Loop through all tar files and generate an .idx file for each
    # wds2idx is a tool from NVIDIA DALI that scans a WebDataset .tar shard and builds an index file listing where each sample is located inside the tar.
    # data-000001.tar is the actual dataset shard, containing your samples (e.g. videos, labels) packed together.
    # data-000001.idx is a small index file that stores byte offsets and metadata pointing to where each sample lives inside the tar, so loaders 
    # (like DALI) can jump directly to items without scanning the whole file.
    # for f in *.tar; do wds2idx "$f" "${f%.tar}.idx"; done