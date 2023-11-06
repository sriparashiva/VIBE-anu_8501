#!/bin/bash
export PATH="/d/sw/slurm/latest/bin:/d/sw/slurm/scripts:/d/sw/openmpi/latest-icc/bin:/d/sw/singularity/3.7.3/bin:/d/sw/singularity/3.7.3/usr/bin:/d/sw/singularity/3.7.3/usr/sbin:/d/sw/java64/jdk-14.0.2/bin:/d/sw/cuda/11.2/cuda-toolkit/nvvm/bin:/d/sw/cuda/11.2/cuda-toolkit/bin:/d/sw/intel/2020.3.036/compilers_and_libraries_2020.4.304/linux/bin/intel64:/d/sw/ucx/1.10.1/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/d/sw/bin:.:/d/sw/scripts:/d/sw/hpc/bin"

export http_proxy="http://proxy.per.dug.com:3128"
export https_proxy="http://proxy.per.dug.com:3128"
export HTTP_PROXY="http://proxy.per.dug.com:3128"
export HTTPS_PROXY="http://proxy.per.dug.com:3128"

module add miniconda
eval "$(conda shell.bash hook)"
conda activate /data/anu_8501/anu_u7034818/envs/vibe-env
./scripts/prepare_data.sh
