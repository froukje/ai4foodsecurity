## Instructions

Singularity is the way to use docker containers on HPC systems. Documentation for using singularity on mistral:
https://www.dkrz.de/up/systems/mistral/singularity

As explained in the documentation, it is not possible to directly create the image directly on mistral as it requires sudo. I created the docker image on my laptop (ubuntu 20.04 running via wsl 2)

The Dockerfile was written following https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45. The base image is centos 8.4.2105, the same OS as you find on the vader nodes.

If you want to make any persistent changes, to the conda environment or by adding files to the container, you need to execute all of the following steps. If you want to use the existing image, jump to https://gitlab.dkrz.de/aim/ai4foodsecurity/-/blob/main/Instructions.md#run-the-singularity-container and follow instructions from there.

## Updating the Docker image

### Create the docker image

1. Check out this git repository
2. Go to the subdirectory Docker `cd docker`
3. Optional: edit the Dockerfile / the environment yaml file (TODO more detail here) and add your changes to git
4. Use the parent directory (i.e. the git root directory) as the docker build context. Execute 

`sudo docker image build -f Dockerfile-development -t ai-4-food-security ..`

5. Check if the image was created correctly with `sudo docker image ls`. You should see an image with the repository name `ai-4-food-security` in the list (about 10 GB).
6. For debugging the conda environment, run the container locally: `sudo docker run ai-4-food-security`

### Transfer the image to dockerhub

You now have a local docker image. To transfer it to mistral, I used dockerhub: `hub.docker.com`. Register a user account (`$USER_DOCKERHUB`) and do the following:

1. `sudo docker tag ai-4-food-security $USER_DOCKERHUB/ai-4-food-security` 
2. `sudo docker push $USER_DOCKERHUB/ai-4-food-security`

This may take some time.

### Transfer the image to mistral

Login to mistral. Checkout the git repository there as well:

1. Checkout the git repository: `git clone git@gitlab.dkrz.de:aim/ai4foodsecurity.git`
2. `cd ai4foodsecurity/images`

Create the singularity image:

1. Start an interactive session on any node with internet access (I used a node from the gpu partition on `mistral.dkrz.de`)
2. Activate singularity module: `module load singularity`
3. Pull your docker image: `singularity pull docker://$USER_DOCKERHUB/ai-4-food-security`

Again, this takes some time. The process will create a file `ai-4-food-security_latest.sif` that is the singularity image.

## Run the singularity container

### Interactive session

Create an allocation for an interactive job on any of the amd nodes

1. `ssh trial.dkrz.de`
2. `salloc --partition=amd --time=04:00:00 --exclusive -A ka1176`
3. `ssh vader{N}` (use `squeue` to see where your interactive job is running)
4. Get started: 

``` { .bash }
# activate singularity module
module load singularity
# start the container
singularity shell --nv --bind /scratch/k/$USER/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/:/work /work/ka1176/shared_data/singularity/images/ai-4-food-security_latest.sif

# --nv for activating the NVIDIA GPUs
# --bind /scratch/k/$USER/singularity/cache:/home/jovyan/.cache: I had problems with some folders that are protected in the singularity container, and where my programs wanted to write to, mostly caches. I used --bind to direct them to my scratch directory.
# --bind /mnt/lustre02/work/ka1176/:/swork: Makes our project directory on /work/ka1176 available within the singularity container as /swork
```

6. Activate the conda environment (unfortunately Docker and Singularity do not seem to be fully compatible here, the shell is not initialized correctly at first): 

``` { .bash }
conda init 
source .bashrc
conda activate ai4foodsecurity
```

To check whether the environment works as intended: 

```
python # opens interactive python shell
>>> import torch
>>> torch.cuda.is_available() # should return True
```



### From a script

TODO

### Starting NNI trials

TODO
