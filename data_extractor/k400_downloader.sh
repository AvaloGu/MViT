#!/bin/bash

# Download directories vars
root_dl="k400"
root_dl_targz="k400_targz"

# Make root directories
# [] is the test command, -d $root_dl checks if this directory already exist
# && is AND
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download train tars, will resume
curr_dl=${root_dl_targz}/train
url=https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
# p stands for parent, create this entire path, and if any folder (parents) along the path is missing
# create them, if folders already exists, it just moves on. "create the parent directories as needed"
[ ! -d $curr_dl ] && mkdir -p $curr_dl 
# wget is a command-line tool to download files over HTTP/HTTPS/FTP.
# -c resumes a partially downloaded file. If the download was interrupted, wget continues from where it stopped
# it will also skip the file if it alread exist.
# if the file is already fully downloaded, it won’t re-download it
# -i tells wget to read URLs from a file, where $url should be a text file, where: each line is a URL to download
# -P $curr_dl (directory prefix), Saves downloaded files into the directory $curr_dl
wget -c -i $url -P $curr_dl

# Download validation tars, will resume
curr_dl=${root_dl_targz}/val
url=https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download test tars, will resume
curr_dl=${root_dl_targz}/test
url=https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download replacement tars, will resume
curr_dl=${root_dl_targz}/replacement
url=https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c $url -P $curr_dl

# Download annotations csv files
curr_dl=${root_dl}/annotations
url_tr=https://s3.amazonaws.com/kinetics/400/annotations/train.csv
url_v=https://s3.amazonaws.com/kinetics/400/annotations/val.csv
url_t=https://s3.amazonaws.com/kinetics/400/annotations/test.csv
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c $url_tr -P $curr_dl
wget -c $url_v -P $curr_dl
wget -c $url_t -P $curr_dl

# Download readme
url=http://s3.amazonaws.com/kinetics/400/readme.md
wget -c $url -P $root_dl

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k400_extractor.sh"
