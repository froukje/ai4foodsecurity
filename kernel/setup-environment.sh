#!/bin/bash

# Adapted from MarDATA training setup

# Add the training environment to user kernel. To see changes take affect do the follwoing:
# 1) Users can run this script once from the standard jupyterlab
# 2) Users should refresh jupyter hub in the browser once so kernels show up
mkdir -p ~/.local/share/jupyter/kernels/
cd ~/.local/share/jupyter/kernels/

# Add symlinks to kernel specifications (this allows to make most changes without requiring users to change anything)

rm -f ai-4-food-security
ln -s /work/ka1176/caroline/gitlab/ai4foodsecurity/kernel/ai-4-food-security


echo "You are all set:"
echo " * We added some juypter kernels to your environment (see ~/.local/share/jupyter/kernels)"
echo ""
echo "Have fun!"

# For the curious user, if you want to setup your own environemnt, have a look at:
# https://jupyterhub.gitlab-pages.dkrz.de/jupyterhub-docs/kernels.html
