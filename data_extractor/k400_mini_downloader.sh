#!/usr/bin/env bash

# Download directories vars
root_dl="k400"
root_dl_targz="k400_targz"

# Make root directories
# [] is the test command, -d $root_dl checks if this directory already exist
# && is AND
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download validation tars, will resume
curr_dl=${root_dl_targz}/val
url=https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# Only download the first 2 tarballs from the list
# curl fetches the contents of a URL, 
# -s is the silent mode, make sure only the urls are outputted to stdout
# pipe to head, grab the first 2 lines
# pipe to wget,
# -i -, read URLs from stdin, - is a Unix convention meaning “standard input”
curl -s $url | head -n 2 | wget -c -i - -P $curr_dl

# Download annotations csv files
curr_dl=${root_dl}/annotations
url_v=https://s3.amazonaws.com/kinetics/400/annotations/val.csv
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c $url_v -P $curr_dl

# Download readme
url=http://s3.amazonaws.com/kinetics/400/readme.md
wget -c $url -P $root_dl

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k400_extractor.sh"
