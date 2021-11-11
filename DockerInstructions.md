## Instructions

Singularity is the way to use docker containers on HPC systems. Documentation for using singularity on mistral:
https://www.dkrz.de/up/systems/mistral/singularity

As explained in the documentation, it is not possible to directly create the image directly on mistral as it requires sudo. I created the docker image on my laptop (ubuntu 20.04 running via wsl 2)

The Dockerfile was written following https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45. The base image is centos 8.4.2105, the same OS as you find on the vader nodes.

If you want to make any persistent changes, to the conda environment or by adding files to the container, you need to execute all of the following steps. If you want to use the existing image, jump to https://gitlab.dkrz.de/aim/ai4foodsecurity/-/blob/main/DockerInstructions.md#run-the-singularity-container and follow instructions from there.

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
singularity shell --nv --bind /scratch/k/$USER/singularity/cache:/miniconda3/envs/ai4foodsecurity/nni --bind /scratch/{k,b}/$USER/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/:/work /work/ka1176/shared_data/singularity/images/ai-4-food-security_latest.sif

# --nv for activating the NVIDIA GPUs
# --bind /scratch/k/$USER/singularity/cache:/home/jovyan/.cache: I had problems with some folders that are protected in the singularity container, and where my programs wanted to write to, mostly caches. I used --bind to direct them to my scratch directory.
# --bind /scratch/k/$USER/singularity/cache:/miniconda3/envs/ai4foodsecurity/nni: for nnictl output
# --bind /mnt/lustre02/work/ka1176/:/work: Makes our project directory on /work/ka1176 available within the singularity container as /work
```

Note that this does not make the home directory visible within the container. The `trial.dkrz.de` home directory is not the same as the `
mistral.dkrz.de` home directory.

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

Please adapt the following script, taken from `/work/ka1176/caroline/jobs/ai4food/prepare_data/submit_planet_5day_2.sh`.

``` { .bash }
#!/bin/bash
#SBATCH -J extract
#SBATCH -p amd
#SBATCH -A ka1176
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --nodelist=vader3

hostname

module load /sw/spack-amd/spack/modules/linux-centos8-zen2/singularity/3.7.0-gcc-10.2.0

# we will bind the folder /work/ka1176 to /swork in the singularity container
gitdir_c=/work/ka1176/caroline/gitlab/ai4foodsecurity  # gitlab dir (change this to gitlab directory as it would appear in the container)
scriptdir_c=/work/ka1176/caroline/jobs/ai4food/prepare_data # script dir (change this to current directory as it would appear in the container)

# create run script for the job
echo "echo 'HELLO BOX'" > singularity_run.sh
echo "gitdir=$gitdir_c" >> singularity_run.sh
echo "conda init" >> singularity_run.sh
echo "source ~/.bashrc" >> singularity_run.sh
echo "conda activate ai4foodsecurity" >> singularity_run.sh
echo "echo \$gitdir" >> singularity_run.sh
echo "cd \$gitdir/scripts" >> singularity_run.sh
echo "python prepare_planet_data.py --n-processes 128 --five-day True --train-set 2" >> singularity_run.sh

# execute the singularity container
singularity exec --bind /mnt/lustre02/work/:/work /work/ka1176/shared_data/singularity/images/ai-4-food-security_latest.sif /bin/bash $scriptdir_c/singularity_run.sh
```

### Starting NNI trials

TODO
