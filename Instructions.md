## Instructions

Dockerfile following https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45. The base image is centos 8.4.2105, the same OS as you find on the vader nodes.

How to create the Docker image:

1. Check out this git repository
2. Go to the subdirectory Docker `cd docker`
3. Use the parent directory (i.e. the git root directory) as the docker build context. Execute 


`sudo docker image build -f Dockerfile-development -t ai-4-food-security ..`
