## Preprocessing

Commands for running preprocessing in interactive jobs - adapt arguments as needed. Corresponding job submission scripts in `/work/ka1176/caroline/jobs/ai4food/prepare_data`.

### Planet

Creates train data and test data as hdf5 files:

`python preprocess_planet.py --n-processes 16 --t-spatial-encoder`

For the version of Planet data with five day cadence

`python preprocess_planet.py --five-day --n-processes 16 --t-spatial-encoder --target-data-dir /work/ka1176/shared_data/2021-ai4food/dev_data/planet_5day/default/`

### Sentinel 1

Creates train data and test data as hdf5 files:

`python preprocess_sentinel_1.py --n-processes 16 --t-spatial-encoder`

### Sentinel 2

Creates train data and test data:

`python preprocess_sentinel_2.py --n-processes 16 --t-spatial-encoder`

## Training

The main training script can be found here: ai4foodsecurity/ai4food/training.py. The script reads the preprocessed data (.h5) and returns predictions in a .json file.

There are several parameters that can be changed. When running from the container the data path has to be changed, e.g. for the planet data:

`python3 training.py --dev-data-dir '/swork/shared_data/2021-ai4food/dev_data/south-africa/planet/default'`

If `--save-preds` is set to `True` the predictions are saved to a .json file. Depending on whether `--split` is set to `train` or `test` the predictions are made on the validation or test set, repectively.
