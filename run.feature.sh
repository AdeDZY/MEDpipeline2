#!/bin/bash

opensmile_path=/home/ubuntu/tools/openSMILE-2.1.0/bin/linux_x64_standalone_static
speech_tools_path=/home/ubuntu/tools/speech_tools/bin
ffmpeg_path=/home/ubuntu/tools/ffmpeg-2.2.4
export PATH=$opensmile_path:$speech_tools_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

video_path=../video   # path to the directory containing all the videos. In this example setup, we are linking all the videos to "../video"

#for line in $(cat "list/all.video"); 
#do
#    ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t 30 -r 15 -vf scale=224x224,setdar=1:1 video2/${line}.mp4
#    ffmpeg -ss 0 -i video/${line}.mp4 -t 30  -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0 keyframes2/${line}_%03d.jpg
#done


for video in $(cat "list/all.video"); 
do
    echo ${video}
    i=0
    for line in $(ls ~/hw2/keyframes/${video}_*);
    do
        let i=i+1
        ~/hw2/scripts/extractSift ${line} > ./sift_features/${video}_$(printf "%03d" $i).sift
    done
done 

    
