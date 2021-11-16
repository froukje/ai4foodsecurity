## Preprocessing

Commands for running preprocessing in interactive jobs - adapt arguments as needed. Corresponding job submission scripts in `/work/ka1176/caroline/jobs/ai4food/prepare_data`.

Calling the main preprocessing module with the most important parameters:

```
python preprocess_planet.py --n-processes <Number of processors for parallel preprocessing (default: 64)>
                            --region [germany, south-africa]
                            --data-source [planet, planet-5, sentinel-1, sentinel-2]
                            --min-area-to-ignore <minimum crop field size (default: 1000)>
                            --t-spatial-encoder <Apply image transform>
                            --t-image-size <If image transform: image size (default: 32)>
                            --t-random-extraction <Apply random extraction (default: 0)>
                            --t-normalize <Apply z-score transform (default: False)>
```

First, the raw data is divided into crop fields and is saved as `fid_{fieldID}.npz` files. In the next step, the transform is applied. There are currently 3 transforms, based on the DENETHOR paper.

```
Image transform with N x N pixel cropped image:     --t-spatial-encoder --t-image-size N
Random extraction transform with M selected pixels: --t-random-extraction M
Spatial average transform: specify none of the arguments that start with --t                         
```

Example for creating data from the South-Africa region, using Sentinel-2 data, and creating random cropped images:

```
python preprocessor.py --region south-africa --data-source sentinel-2 --n-processes 64 --t-spatial-encoder --target-sub-dir default --overwrite
```

Example for creating data from the Germany region, using Planet data, and creating spatially averaged data:

```
python preprocessor.py --region germany -data-source planet --n-processes 64 --target-sub-dir averaged --overwrite
```

Example for creating data from the Germany region, using Planet-5 data, and creating randomly extracted data:

```
python preprocessor.py --region germany -data-source planet-5 --n-processes 64 --t-random-extraction 64 --target-sub-dir extracted --overwrite
```

## Training

The main training script can be found here: ai4foodsecurity/ai4food/training.py. The script reads the preprocessed data (.h5) and returns predictions in a .json file.

There are several parameters that can be changed. When running from the container the data path has to be changed, e.g. for the planet data:

`python3 training.py --dev-data-dir '/swork/shared_data/2021-ai4food/dev_data/south-africa/planet/default'`

If `--save-preds` is set to `True` the predictions are saved to a .json file. Depending on whether `--split` is set to `train` or `test` the predictions are made on the validation or test set, repectively.

## Hyperparametertuning using nni

In order to use nni, the flag `--nni` has to be set.

*Note:* In the image `..._latest.sif` the current nni version is 2.4, this leads to some problems, so I downgraded it to 2.3 and built a new image`..._nni.sif`, it is also stored in `shared_data/singularity/images` and has to be used, when nni is used - this can be converted in the `.._latest.sif` version in future.

Allocate a node on vader, e.g.: `salloc -A k20200 -n 1 -p amd --exclusive`

Run the following script:
`
#SBATCH --mem=0 # use entire memory of node
#SBATCH --exclusive # do not share node
#SBATCH --time=12:00:00 # limit of total run time
#SBACTH --mail-type=FAIL
#SBATCH --account=k20200
#SBATCH --nodelist vader1

hostname

module load /sw/spack-amd/spack/modules/linux-centos8-zen2/singularity/3.7.0-gcc-10.2.0

gitdir_c=/swork/frauke/ai4foodsecurity/nni  # gitlab dir (change this to gitlab directory as it would appear in the container)
scriptdir_c=/swork/frauke/ai4foodsecurity/jobs # script dir (change this to current directory as it would appear in the container)

echo "echo 'HELLO BOX'" > singularity_run_nni.sh
echo "gitdir=$gitdir_c" >> singularity_run_nni.sh
echo "conda init" >> singularity_run_nni.sh
echo "source .bashrc" >> singularity_run_nni.sh
echo "conda activate ai4foodsecurity" >> singularity_run_nni.sh
echo "export NNI_OUTPUT_DIR=\$gitdir" >> singularity_run_nni.sh # this is supposed to change the dir of the exp, but it's not working!!
echo "port=$((8080 + $RANDOM % 10000))" >> singularity_run_nni.sh
echo "nnictl create -c \$gitdir/config.yml --port \$port || nnictl create -c \$gitdir/config.yml --port \$port || nnictl create -c \$gitdir/config.yml --port \$port || nnictl create -c \$gitdir/config.yml --port \$port" >> singularity_run_nni.sh
echo "sleep 11h 50m" >> singularity_run_nni.sh
echo "nnictl stop" >> singularity_run_nni.sh

# execute the singularity container
singularity exec --nv --bind /scratch/k/k202143/singularity/cache:/miniconda3/envs/ai4foodsecurity/nni --bind /scratch/k/k202143/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/ka1176:/swork /mnt/lustre02/work/ka1176/frauke/ai4foodsecurity/images/ai-4-food-security_nni.sif bash $scriptdir_c/singularity_run_nni.sh
"start_nni_job.sh"    
`
