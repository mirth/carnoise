#!/bin/bash

SAMPLE_RATE=16000

dir=$1

# fetch_clip(videoID, startTime, endTime)
fetch_clip() {
  echo "Fetching $1"
  outname="$1"
  if [ -f "$dir/${outname}.wav" ]; then
    echo "Already have it."
    return
  fi

  youtube-dl https://youtube.com/watch?v=$1 \
    --quiet --extract-audio --audio-format wav \
    --output "$dir/$outname.%(ext)s"
  if [ $? -eq 0 ]; then
    # If we don't pipe `yes`, ffmpeg seems to steal a
    # character from stdin. I have no idea why.
    yes | ffmpeg -loglevel quiet -i "$dir/$outname.wav" -ar $SAMPLE_RATE "$dir/${outname}_out.wav"
    mv "$dir/${outname}_out.wav" "$dir/$outname.wav"
    #gzip "$dir/$outname.wav"
  else
    # Give the user a chance to Ctrl+C.
    sleep 1
  fi
}

while read line
do
  fetch_clip $(echo "$line" | cut -d, -f2)
done
