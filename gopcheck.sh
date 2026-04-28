for f in $(ls ./k400/val/*.mp4 | shuf -n 20); do
    echo "=== $f ==="
    ffprobe -v error -select_streams v:0 \
        -show_entries frame=pict_type \
        -of csv=p=0 "$f" | tr ',' '\n' | sort | uniq -c
    echo
done

# GOP (Group of Pictures): the frames from one I-frame up to (but not including) the next I-frame
# Video compression exploits the fact that consecutive frames are usually very similar. 
# Instead of storing each frame independently, most frames are stored as differences from other frames.
# I-frames are complete, standalone image. Stored like a JPEG — no references to other frames. Can be decoded entirely on its own.
# Every video must start with an I-frame, because decode has to start from something complete.
# P-frame (Predicted): Stored as differences from the previous I-frame or P-frame. size of 1/5 to 1/10 of an I-frame
# B-frames (Bidirectionally predicted): stored (interpolated) as differences from both past and future frames. More compressed than P
# our input videos have very few I-frames, so seeking a random frame is extremely slow. 
