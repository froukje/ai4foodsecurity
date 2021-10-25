## How to use Jupyterhub for this challenge

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

