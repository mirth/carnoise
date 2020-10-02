input_dir=$1
output_dir=$2


mkdir ${output_dir}
for wavfile in ${input_dir}/*.wav
do
  filename=$(basename -- "$wavfile")
  echo $filename
  ffmpeg -i "${wavfile}" -f segment -segment_time 5 -c copy "${output_dir}/${filename%%.*}_%03d.wav"
done
