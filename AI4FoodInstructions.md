## Preprocessing

Commands for running preprocessing in interactive jobs - adapt arguments as needed. Corresponding job submission scripts in `/work/ka1176/caroline/jobs/ai4food/prepare_data`.

### Planet

Creates train data and test data as hdf5 files:

`python preprocess_planet.py --n-processes 16 --t-spatial-encoder`

For the version of Planet data with five day cadence

`python preprocess_planet.py --five-day --n-processes 16 --t-spatial-encoder --target-data-dir /work/ka1176/sha    red_data/2021-ai4food/dev_data/planet_5day/default/`

### Sentinel 1

Creates train data and test data as hdf5 files:

`python preprocess_sentinel_1.py --n-processes 16 --t-spatial-encoder`

### Sentinel 2

Creates train data and test data:

`python preprocess_sentinel_2.py --n-processes 16 --t-spatial-encoder`
