#!/usr/bin/env python

import os
import sys

import argparse

import numpy as np
import geopandas as gpd

import h5py

import custom_planet_reader as CPReader
from custom_data_transform import PlanetTransform

import time
import datetime

import warnings

def extend_dataset(reader, keys, raw_ds):
    '''
    Extend a given raw_ds by iterating reader with keys
    '''

    start_time = time.time()

    for i,sample in enumerate(reader):
        if i%100 == 0:
            print(f'Processed 100 samples in {time.time() - start_time:.0f} seconds')
            start_time = time.time()
        for key in keys:
            if key == 'image_stack':
               raw_ds[key].extend(sample[0].numpy())
            elif key == 'label':
               raw_ds[key].append(sample[1])
            elif key == 'mask':
               raw_ds[key].extend(sample[2].numpy())
            elif key == 'fid':
               raw_ds[key].append(sample[3])
            else:
               raise ValueError(f'Invalid key: {key}')

    return raw_ds

def save_dataset(raw_ds, keys, image_size, filename):
    '''Save raw_ds to filename (hdf5)'''

    n_samples = len(raw_ds['label'])
    print(f'Total samples available in {filename}: {n_samples}')

    image_dims = (244, 4, image_size, image_size,)
    chunk_size = 100
    mask_dims  = (image_size, image_size,)

    # save
    h5_file = h5py.File(os.path.join(args.target_data_dir, filename), 'w')
    for key in keys:
        print(key)
        if key == 'image_stack':
            shape = (n_samples,) + image_dims
            h5_file.create_dataset(key,
                                   shape=shape,
                                   chunks=(chunk_size,) + image_dims,
                                   fletcher32=True,
                                   dtype='float32')
            vals = np.array(raw_ds[key]).reshape(shape)
        elif key == 'label':
            h5_file.create_dataset(key,
                                   shape=(n_samples,), 
                                   chunks=(chunk_size,), 
                                   fletcher32=True,
                                   dtype='int')
            vals = np.array(raw_ds[key])
        elif key == 'mask':
            shape = (n_samples,) + mask_dims
            h5_file.create_dataset(key,
                                   shape=shape,
                                   chunks=(chunk_size,) + mask_dims,
                                   fletcher32=True,
                                   dtype='float32')
            vals = np.array(raw_ds[key]).reshape(shape)
        elif key == 'fid':
            h5_file.create_dataset(key,
                                   shape=(n_samples,), 
                                   chunks=(chunk_size,), 
                                   fletcher32=True,
                                   dtype='int')
            vals = np.array(raw_ds[key])
        h5_file[key][:] = vals
        h5_file.flush()
    h5_file.attrs['time_created'] = str(datetime.datetime.now())
    print(f'closing {filename}')
    h5_file.close()

def main(args):

    warnings.filterwarnings("ignore")
    
    # set up paths
    if args.region == 'south-africa':
        train_dir = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_train_source_planet')
        test_dir = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_test_source_planet')
        if args.five_day:
            train_dir += '_5day'
            test_dir  += '_5day'
            train_dir_1 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_5day_34S_19E_258N')
            train_dir_2 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_5day_34S_19E_259N')
        else:
            train_dir_1 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_34S_19E_258N')
            train_dir_2 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_34S_19E_259N')

        label_dir = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_train_labels')
        label_dir_1 = os.path.join(label_dir, 'ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson')
        label_dir_2 = os.path.join(label_dir, 'ref_fusion_competition_south_africa_train_labels_34S_19E_259N/labels.geojson')
        label_dir = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_test_labels')
        label_dir_T = os.path.join(label_dir, 'ref_fusion_competition_south_africa_test_labels_34S_20E_259N/labels.geojson')

        list_source = [train_dir_1, train_dir_2, test_dir]
        list_labels = [label_dir_1, label_dir_2, label_dir_T]
        list_targets = [os.path.join(args.target_data_dir, tt) for tt in ['train_1', 'train_2', 'test']]
        list_is_train = [True, True, False]
        list_comment = ['First train set', 'Second train set', 'Test set']

    elif args.region == 'germany':
        raise ValueError(f'Not implemented: region = {args.region}')

        list_source = ...
        list_labels = ...
        list_targets = ...
        list_is_train = ...
        list_comment = ...

    # Get the transformer
    transform = PlanetTransform(spatial_encoder=args.t_spatial_encoder,
                                   normalize=args.t_normalize,
                                   image_size=args.t_image_size)

    # process the raw data to patches

    train_readers = []
    test_readers = []

    for source, label, target, comment, is_train in zip(list_source, list_labels, list_targets, list_comment, list_is_train):

        print(comment)
        print(f'Reading from {source}')
        print(f'Labels from {label}')

        labels = gpd.read_file(label)
        label_ids = labels['crop_id'].unique()
        label_names = labels['crop_name'].unique()

        reader = CPReader.PlanetReader(input_dir=source,
                                    label_dir=label,
                                    output_dir=target,
                                    label_ids=label_ids,
                                    transform=transform.transform,
                                    min_area_to_ignore=args.min_area_to_ignore,
                                    overwrite=True,
                                    n_processes=args.n_processes)

        if is_train:
            train_readers.append(reader)
        else:
            test_readers.append(reader)


    # process the patches to hdf5
    keys = ['image_stack', 'label', 'mask', 'fid']

    # process all training data sets (validation will be split later)
    raw_ds = {key:[] for key in keys}

    for reader in train_readers:
        raw_ds = extend_dataset(reader, keys, raw_ds)

    save_dataset(raw_ds, keys, args.t_image_size, 'train_data.h5')

    # process all test data sets
    raw_ds = {key:[] for key in keys}

    for reader in test_readers:
        raw_ds = extend_dataset(reader, keys, raw_ds)

    save_dataset(raw_ds, keys, args.t_image_size, 'test_data.h5')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', type=str,
                        default='/work/ka1176/shared_data/2021-ai4food/raw_data/')
    parser.add_argument('--target-data-dir', type=str,
                        default='/work/ka1176/shared_data/2021-ai4food/dev_data/planet/default/')
    parser.add_argument('--region', type=str, choices=['south-africa', 'germany'],
                        default='south-africa', help='Select region')
    parser.add_argument('--n-processes', type=int, default=1)
    parser.add_argument('--min-area-to-ignore', type=float, default=1000, help='Fields below minimum area are ignored')
    parser.add_argument('--five-day', action='store_true', help='use the planet fusion data averaged over 5 days') 
    # transformer arguments
    parser.add_argument('--t-spatial-encoder', action='store_true', help='Transformer variable')
    parser.add_argument('--t-normalize', action='store_true', help='Transformer variable')
    parser.add_argument('--t-image-size', type=int, default=32, help='Transformer variable')

    args = parser.parse_args()

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***')

    main(args)
