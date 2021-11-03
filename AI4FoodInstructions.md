## Preprocessing

Commands for running preprocessing in interactive jobs - adapt arguments as needed

### Sentinel 1

Creates train data and test data as hdf5 files:

`python preprocess_sentinel_1.py --n-processes 1 --t-spatial-encoder --t-normalize`

### Sentinel 2

Creates train data and test data:

`python preprocess_sentinel_2.py --n-processes 16 --t-spatial-encoder --t-normalize`
