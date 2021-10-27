## How to use Jupyterhub for this challenge

### Start the Jupyterhub session

Log into https://jupyterhub.dkrz.de. Start a Jupyter session with the following settings:

1. From "Spawner Options", select "Advanced"
2. Click on "advanced slurm options"
3. Enter the server options:
    * Account: ka1176
    * Partition: shared / compute / gpu as required
    * Reservation: none
    * Time: as required
    * Number of cores: as required
    * Memory: 32000 
    * User interface: select JupyterLab

### First-time user: install kernel

Skip this step if you have the kernel installed already.

In JupyterLab, open a terminal.

```
cd $GIT_ROOTDIR # this is where you checked out your git repository
cd kernel
. setup_environment.sh
```

Refresh your browser window (`F5`) and you should have the kernel `ai-4-food-security` available. 


### Select the kernel

Select the kernel `ai-4-food-security` and get started with your Jupyter Notebook!

## Advanced: Setup kernel

Create the Docker image from `Dockerfile-mistral` analogous to the directions given in DockerInstructions.md.

```
# on your laptop

sudo docker tag ai-4-food-security-mistral cadkrz/ai-4-food-security-mistral
sudo docker push cadkrz/ai-4-food-security-mistral
```

```
# on a node in the amd partition (trial.dkrz.de)
module load singularity
cd /work/ka1176/shared_data/singularity/images/
singularity pull -F docker://cadkrz/ai-4-food-security-mistral
```

Proceed with setup as described above (first-time user).
