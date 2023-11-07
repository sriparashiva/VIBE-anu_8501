#!/bin/bash
module add singularity

export http_proxy="http://proxy.per.dug.com:3128"
export https_proxy="http://proxy.per.dug.com:3128"

singularity exec --nv -B /data/anu_8501 -B /d/sw -B /usr/share/glvnd /data/anu_8501/.jupyterhub/jl-tf-23.01-tf2-py3-protobuf.sif ./run_demo.sh
