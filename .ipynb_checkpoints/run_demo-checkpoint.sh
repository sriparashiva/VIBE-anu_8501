#!/bin/bash
eval "$(/d/sw/miniconda3/4.8.3/condabin/conda shell.bash hook)"
conda activate /data/anu_8501/anu_u7034818/envs/vibe-env

 

export http_proxy="http://proxy.per.dug.com:3128"
export https_proxy="http://proxy.per.dug.com:3128"
python demo.py --vid_file sample_video.mp4 --output_folder output/
