#!/bin/bash

checkpoint=/host_data/van/LDA/model/finedance/lightning_logs/version_0/checkpoints/epoch=9-step=21480.ckpt
dest_dir=/host_data/van/LDA/results/finedance

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=/host_data/van/LDA/data/finedance/feat
wav_dir=/host_data/van/LDA/data/finedance/music_wav
basenames=$(cat "${data_dir}/test_list.txt")

start=0
seed=150
fps=24
trim_s=0
length_s=10
trim=$((trim_s*fps))
length=$((length_s*fps))
fixed_seed=false
gpu="cuda:0"
render_video=true

for wavfile in $basenames; 
do
	start=0
	style=$(echo $wavfile | awk -F "_" '{print $2}') #Coherent style parsed from file-name
	for postfix in 0 1 2 3 4 5 6 7 8 9 10 11
	do
		input_file=${wavfile}.audio29_${fps}fps.pkl
		
		output_file=${wavfile}_${postfix}_${style}
		
		echo "start=${start}, len=${length}, postfix=${postfix}, seed=${seed}"
		python synthesize.py --checkpoints="${checkpoint}" --data_dirs="${data_dir}" --input_files="${input_file}" --styles="${style}" --start=${start} --end=${length} --seed=${seed} --postfix=${postfix} --trim=${trim} --dest_dir=${dest_dir} --gpu=${gpu} --video=${render_video} --outfile=${output_file}
		if [ "$fixed_seed" != "true" ]; then
			seed=$((seed+1))
		fi 
		echo seed=$seed
		python utils/cut_wav.py ${wav_dir}/${wavfile}.wav $(((start+trim)/fps)) $(((start+length-trim)/fps)) ${postfix} ${dest_dir}
		if [ "$render_video" == "true" ]; then
			ffmpeg -y -i ${dest_dir}/${output_file}.mp4 -i ${dest_dir}/${wavfile}_${postfix}.wav ${dest_dir}/${output_file}_audio.mp4
			rm ${dest_dir}/${output_file}.mp4
		fi
		
		start=$((start+length))
	done
done
