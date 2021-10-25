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
    * Number of cores: depends on task
    * Memory: 32000 
    * User interface: select JupyterLab

### First-time user: install kernel

Skip this step if you have the kernel installed already.

TODO: Instructions how to start the kernel

### Select the kernel

Select the kernel `ai-4-food-security` and get started with your Jupyter Notebook!
