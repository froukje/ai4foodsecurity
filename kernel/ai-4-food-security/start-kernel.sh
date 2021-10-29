#!/bin/bash

IMAGE_ROOT=/work/ka1176/shared_data/singularity/images

IMAGE_PATH=${IMAGE_ROOT}/ai-4-food-security-mistral_latest.sif

source /etc/profile
module purge

#. /sw/spack-rhel6/spack/share/spack/setup-env.sh       # if singularity is not found, try explicitly adding the spack software tree
module load singularity

# make some global filesystem namespaces available in container:
export SINGULARITY_BINDPATH="/scratch, /work, /mnt/lustre01, /mnt/lustre02"


# as noted in: https://www.dkrz.de/up/systems/mistral/singularity#CACHE-and-TMP-directories
# redirect singularity temporary data to scratch to avoid home from overflowing
mkdir -p /scratch/k/$USER/singularity/{cache,tmp}
export SINGULARITY_TMPDIR=/scratch/k/$USER/singularity/tmp
export SINGULARITY_CACHEDIR=/scratch/k/$USER/singularity/cache


#singularity shell --nv --cleanenv ${IMAGE_PATH}  # uncomment to troubleshoot
singularity exec --nv --cleanenv ${IMAGE_PATH} python -m ipykernel_launcher -f "$1"
# --nv   to expose GPUs, see: https://sylabs.io/guides/3.5/user-guide/gpu.html
