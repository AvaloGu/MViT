#!/bin/bash

# Download directories vars
root_dl="k400"
root_dl_targz="k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl

# Extract train
curr_dl=$root_dl_targz/train
curr_extract=$root_dl/train
[ ! -d $curr_extract ] && mkdir -p $curr_extract
# $(...) is command substitution, it captures the output of a command and stores it in a variable.
tar_list=$(ls $curr_dl)
for f in $tar_list
do
# It checks whether a filename looks like a tarball, and if so, extracts it.
# [...] test command, [[...]] is the newer version that allows for much more powerful string comparisons,
# such as pattern matching.
# echo Extracting $curr_dl/$f to $curr_extract just prints a status message to the screen
# tar zxf $curr_dl/$f -C $curr_extract is where extraction happens
# tar command to extract the archive
# z flag is filter through gzip (.gz), i.e. pass the data through gzip to decompress it
# x flag is to extract
# in other words, zx together means “gunzip it and untar it.”
# f flag means next argument is the file name
# -C $curr_extract means change directory before extracting, files go into $curr_extract, not the current directory
# -C is local and only tar is affected, no side effect
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extract validation
curr_dl=$root_dl_targz/val
curr_extract=$root_dl/val
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extract test
curr_dl=$root_dl_targz/test
curr_extract=$root_dl/test
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extract replacement
curr_dl=$root_dl_targz/replacement
curr_extract=$root_dl/replacement
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tgz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extraction complete
echo -e "\nExtractions complete!"