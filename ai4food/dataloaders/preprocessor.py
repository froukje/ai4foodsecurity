#!/usr/bin/env python

import os
import sys

import argparse

import numpy as np
import geopandas as gpd

import h5py

import custom_planet_reader as CPReader
import custom_sentinel_1_reader as CS1Reader
import custom_sentinel_2_reader as CS2Reader

from custom_data_transform import PlanetTransform, Sentinel1Transform, Sentinel2Transform

import time
import datetime

import warnings


class Preprocessor(object):

    def __init__(self, args):
        '''
        Initialize the preprocessor: set up paths for given data source and region

        args is an argparse namespace with
        * data-source
        * raw-data-dir
        * target-sub-dir
        * region
        * n-processes
        * min-area-to-ignore
        * overwrite
        * # transformer arguments
        * t-spatial-encoder
        * t-normalize
        * t-image-size

        See preprocessor.py (main) for description
        '''

        self.args = args
        self._setup_data_paths()
        self._setup_transform()

        # keys for the hdf5 file
        self.keys = ['image_stack', 'label', 'mask', 'fid', 'crop_name']

    def run(self):
        '''
        Run the data preprocessing

        - Create readers for train and test data
        - From readers, generate samples, apply transforms
        - Save hdf5 files
        '''

        self.train_readers = []
        self.test_readers = []

        # iterate all train sources
        for input_dir, label_dir, output_dir in zip(self.train_sources, self.train_labels, self.train_targets):
            print(f'Reading from {input_dir}')
            print(f'Labels from {label_dir}')
            print(f'Save to {output_dir}')

            labels = gpd.read_file(label_dir)
            label_ids = labels['crop_id'].unique()
            label_names = labels['crop_name'].unique()

            reader = Preprocessor.create_reader(self.args.data_source, input_dir, label_dir, output_dir, label_ids, self.transform, self.args.min_area_to_ignore, self.args.overwrite, self.args.n_processes)

            self.train_readers.append(reader)

        # iterate all test sources
        for input_dir, label_dir, output_dir in zip(self.test_sources, self.test_labels, self.test_targets):
            print(f'Reading from {input_dir}')
            print(f'Labels from {label_dir}')
            print(f'Save to {output_dir}')

            labels = gpd.read_file(label_dir)
            label_ids = labels['crop_id'].unique()
            label_names = labels['crop_name'].unique()

            # min area to ignore is 0 for the test set
            reader = Preprocessor.create_reader(self.args.data_source, input_dir, label_dir, output_dir, label_ids, self.transform, 0, self.args.overwrite, self.args.n_processes)

            self.test_readers.append(reader)

        # save the processed dataset for train
        self._save_datasets()

    def _save_datasets(self):
        '''
        Save the processed patches as hdf5
        
        train sets --> train_data.h5
        test sets  --> test_data.h5
        '''

        # train
        print('Create datasets with keys', self.keys)
        raw_ds = {key:[] for key in self.keys}

        for reader in self.train_readers:
            raw_ds = Preprocessor.extend_dataset(reader, self.keys, raw_ds)

        Preprocessor.create_hdf5(raw_ds, 
                                 self.keys, 
                                 self.time_size_train, 
                                 self.band_size, 
                                 self.custom_transform, 
                                 self.custom_size, 
                                 os.path.join(self.target_dir, 'train_data.h5'))

        # test 
        raw_ds = {key:[] for key in self.keys}

        for reader in self.test_readers:
            raw_ds = Preprocessor.extend_dataset(reader, self.keys, raw_ds)

        Preprocessor.create_hdf5(raw_ds, 
                                 self.keys, 
                                 self.time_size_train, 
                                 self.band_size, 
                                 self.custom_transform, 
                                 self.custom_size, 
                                 os.path.join(self.target_dir, 'test_data.h5'))


    def _setup_transform(self):
        '''
        Initialize the class attribute transform
        '''

        if self.args.data_source == 'planet' or self.args.data_source == 'planet-5':
            transform = PlanetTransform(spatial_encoder=self.args.t_spatial_encoder,
                                        random_extraction=self.args.t_random_extraction,
                                        normalize=self.args.t_normalize,
                                        image_size=self.args.t_image_size)

        elif self.args.data_source == 'sentinel-1':
            transform = Sentinel1Transform(spatial_encoder=self.args.t_spatial_encoder,
                                           random_extraction=self.args.t_random_extraction,
                                           normalize=self.args.t_normalize,
                                           image_size=self.args.t_image_size)
        elif self.args.data_source == 'sentinel-2':
            transform = Sentinel2Transform(spatial_encoder=self.args.t_spatial_encoder,
                                           random_extraction=self.args.t_random_extraction,
                                           normalize=self.args.t_normalize,
                                           image_size=self.args.t_image_size)

        self.transform = transform

        if self.args.t_spatial_encoder: # image transform
            self.custom_transform = 'image_transform'
            self.custom_size = self.args.t_image_size 
        else:
            if self.args.t_random_extraction > 0:
                self.custom_transform = 'extract_transform'
                self.custom_size = self.args.t_random_extraction
            else:
                print('Select spatial average transform')
                self.custom_transform = 'average_transform'
                self.custom_size = 1

        print(f'Transform {self.custom_transform} with size {self.custom_size}')

    def _setup_data_paths(self):
        '''
        Initialize input data paths for the given data_source and region

        source : [planet, planet-5, sentinel-1, sentinel-2]
        region:  [south-africa, germany]

        Initializes class attributes

        train_labels  : list of labels for training
        test_labels   : list of labels for testing
        train_sources : list of dirs for training
        test_sources  : list of dirs for testing
        train_targets : list of dirs for processed training data
        test_targets  : list of dirs for processed test data
        time_size_train     : time dimension of output (train set)
        time_size_test      : time dimension of output (test set)
        band_size     : band dimension of output

        '''

        self.target_dir = os.path.join('/work/ka1176/shared_data/2021-ai4food/dev_data/', self.args.region, self.args.data_source, self.args.target_sub_dir)

        # labels are different by region
        if self.args.region == 'south-africa':
            label_dir = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_train_labels')
            label_dir_1 = os.path.join(label_dir, 'ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson')
            label_dir_2 = os.path.join(label_dir, 'ref_fusion_competition_south_africa_train_labels_34S_19E_259N/labels.geojson')
            label_dir = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_test_labels')
            label_dir_T = os.path.join(label_dir, 'ref_fusion_competition_south_africa_test_labels_34S_20E_259N/labels.geojson')

            self.train_labels = [label_dir_1, label_dir_2]
            self.test_labels  = [label_dir_T]

        elif self.args.region == 'germany':
            label_dir_1 = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_train_labels',
                                                               'dlr_fusion_competition_germany_train_labels_33N_18E_242N',
                                                               'labels.geojson')
            label_dir_T = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_test_labels',
                                                               'dlr_fusion_competition_germany_test_labels_33N_17E_243N',
                                                               'labels.geojson')

            self.train_labels = [label_dir_1]
            self.test_labels  = [label_dir_T]

        # train source is different by source and region
        if self.args.region == 'south-africa':
            if self.args.data_source == 'planet':
                train_dir = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_train_source_planet')
                test_dir = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_test_source_planet')
                train_dir_1 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_34S_19E_258N')
                train_dir_2 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_34S_19E_259N')

                self.time_size_train = 244 # time steps available
                self.band_size = 4 # bands available

            elif self.args.data_source == 'planet-5':
                train_dir = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_train_source_planet_5day')
                test_dir = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_test_source_planet_5day')
                train_dir_1 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_5day_34S_19E_258N')
                train_dir_2 = os.path.join(train_dir, 'ref_fusion_competition_south_africa_train_source_planet_5day_34S_19E_259N')

                self.time_size_train = 48 # time steps available
                self.band_size = 4 # bands available

            elif self.args.data_source == 'sentinel-1':
                tmp = os.path.join(self.args.raw_data_dir, 'ref_fusion_competition_south_africa_train_source_sentinel_1')
                train_dir_1 = os.path.join(tmp, 'ref_fusion_competition_south_africa_train_source_sentinel_1_34S_19E_258N_asc_34S_19E_258N_2017')
                train_dir_2 = os.path.join(tmp, 'ref_fusion_competition_south_africa_train_source_sentinel_1_34S_19E_259N_asc_34S_19E_259N_2017')
                test_dir = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_test_source_sentinel_1', \
                                        'ref_fusion_competition_south_africa_test_source_sentinel_1_asc_34S_20E_259N_2017')

                self.time_size_train = 41 # time steps available
                self.band_size = 2 # bands available

            elif self.args.data_source == 'sentinel-2':
                tmp = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_train_source_sentinel_2')
                train_dir_1 = os.path.join(tmp, 'ref_fusion_competition_south_africa_train_source_sentinel_2_34S_19E_258N_34S_19E_258N_2017')
                train_dir_2 = os.path.join(tmp, 'ref_fusion_competition_south_africa_train_source_sentinel_2_34S_19E_259N_34S_19E_259N_2017')
                test_dir = os.path.join(args.raw_data_dir, 'ref_fusion_competition_south_africa_test_source_sentinel_2', \
                                        'ref_fusion_competition_south_africa_test_source_sentinel_2_34S_20E_259N_2017')

                self.time_size_train = 76 # time steps available
                self.band_size = 12 # bands available

            # this is the same for all south-africa sources
            self.train_sources = [train_dir_1, train_dir_2]
            self.test_sources  = [test_dir]
            self.train_targets = [os.path.join(self.target_dir, 'train_1'), os.path.join(self.target_dir, 'train_2')]
            self.test_targets  = [os.path.join(self.target_dir, 'test')]
            self.time_size_test = self.time_size_train # in season prediction: same time steps

        elif self.args.region == 'germany':
            if self.args.data_source == 'planet':
                train_dir = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_train_source_planet')
                test_dir  = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_test_source_planet')

                self.time_size_train = 365 # time steps available
                self.time_size_test  = 365
                self.band_size = 4 # bands available

            elif self.args.data_source == 'planet-5':
                train_dir = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_train_source_planet_5day')
                test_dir  = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_test_source_planet_5day')

                self.time_size_train = 73 # time steps available
                self.time_size_test  = 73
                self.band_size = 4 # bands available

            elif self.args.data_source == 'sentinel-1':
                train_dir = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_train_source_sentinel_1', 'dlr_fusion_competition_germany_train_source_sentinel_1_asc_33N_18E_242N_2018')
                test_dir  = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_test_source_sentinel_1', 'dlr_fusion_competition_germany_test_source_sentinel_1_asc_33N_17E_243N_2019')

                self.time_size_train = 122 # time steps available
                self.time_size_test  = 120
                self.band_size = 2 # bands available

            elif self.args.data_source == 'sentinel-2':
                train_dir = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_train_source_sentinel_2', 'dlr_fusion_competition_germany_train_source_sentinel_2_33N_18E_242N_2018')
                test_dir  = os.path.join(self.args.raw_data_dir, 'dlr_fusion_competition_germany_test_source_sentinel_2', 'dlr_fusion_competition_germany_test_source_sentinel_2_33N_17E_243N_2019')

                self.time_size_train = 144 # time steps available
                self.time_size_test  = 144
                self.band_size = 12 # bands available

            # this is the same for all germany sources
            # for compatibility with the south-africa processing, keep these as lists
            self.train_sources = [train_dir]
            self.test_sources  = [test_dir]
            self.train_targets = [os.path.join(self.target_dir, 'train')]
            self.test_targets  = [os.path.join(self.target_dir, 'test')]

    @staticmethod
    def create_reader(data_source, input_dir, label_dir, output_dir, label_ids, transform, min_area_to_ignore, overwrite, n_processes):
        '''
        Create a custom reader for a dataset

        Parameters:
        data_source : Specify data source to select the correct reader
        input_dir   : folder containing raw data
        label_dir   : folder containing labels
        output_dir  : folder for saving processed patches
        label_ids   : label IDs
        transform   : custom transform
        min_area_to_ignore : ignore fields below this area (m2)
        overwrite   : If True, overwrite existing processed patches
        n_processes : Use for multiprocessing

        Returns:
        reader
        '''

        if data_source == 'planet' or data_source == 'planet-5':
            reader = CPReader.PlanetReader(input_dir=input_dir,
                                           label_dir=label_dir,
                                           output_dir=output_dir,
                                           label_ids=label_ids,
                                           transform=transform.transform,
                                           min_area_to_ignore=min_area_to_ignore,
                                           overwrite=overwrite,
                                           n_processes=n_processes)
        elif data_source == 'sentinel-1':
            reader = CS1Reader.S1Reader(input_dir=input_dir,
                                        label_dir=label_dir,
                                        output_dir=output_dir,
                                        label_ids=label_ids,
                                        transform=transform.transform,
                                        min_area_to_ignore=min_area_to_ignore,
                                        include_cloud=False, # do not change
                                        overwrite=overwrite,
                                        n_processes=n_processes)
        elif data_source == 'sentinel-2':
            reader = CS2Reader.S2Reader(input_dir=input_dir,
                                        label_dir=label_dir,
                                        output_dir=output_dir,
                                        label_ids=label_ids,
                                        transform=transform.transform,
                                        min_area_to_ignore=min_area_to_ignore,
                                        include_cloud=False, # do not change
                                        overwrite=overwrite,
                                        n_processes=n_processes)

        return reader

    @staticmethod
    def extend_dataset(reader, keys, raw_ds):
        '''
        Extend a given raw_ds by iterating reader with keys

        Parameters:
        reader : a custom data source reader
        keys   : keys for the hdf5 file
        raw_ds : raw dataset (dictionary)

        Returns:
        raw_ds, now containing all samples that were provided by reader
        '''

        start_time = time.time()

        for i,sample in enumerate(reader):
            if sample is None:
                print('skipping sample in extend_dataset', i)
                continue
            if i%100 == 0:
                print(f'Processed 100 samples in {time.time() - start_time:.0f} seconds')
                start_time = time.time()
            for key in keys:
                if key == 'image_stack':
                   raw_ds[key].extend(sample[0].astype(np.float32))
                elif key == 'label':
                   raw_ds[key].append(sample[1])
                elif key == 'mask':
                   if sample[2].shape == ():
                       raw_ds[key].append(sample[2].astype(np.float32))
                   else:
                       raw_ds[key].extend(sample[2].astype(np.float32))
                elif key == 'fid':
                   raw_ds[key].append(sample[3])
                elif key == 'crop_name':
                   raw_ds[key].append(sample[4])
                else:
                   raise ValueError(f'Invalid key: {key}')

        return raw_ds

    @staticmethod
    def create_hdf5(raw_ds, keys, time_size, band_size, custom_transform, custom_size, filename):
        '''
        Save raw_ds to filename (hdf5)
        
        Parameters:
        raw_ds : raw dataset (dictionary)
        keys   : keys for the hdf5 file
        time_size : size along time dimension
        band_size : size along the band dimension
        custom_transform : which custom transform was applied (determines shape)
        custom_size : size along the width & height dimension (image transform)
                      size of random extracted pixels (extraction transform)
                      size of 1 (average transform)
        filename : target file name
        '''

        n_samples = len(raw_ds['label'])
        print(f'Total samples available in {filename}: {n_samples}')

        chunk_size = 100

        if custom_transform == 'image_transform':
            image_dims = (time_size, band_size, custom_size, custom_size,)
            mask_dims  = (custom_size, custom_size,)
        elif custom_transform == 'extract_transform':
            image_dims = (time_size, band_size, custom_size)
            mask_dims = (custom_size,)
        elif custom_transform == 'average_transform':
            image_dims = (time_size, band_size, custom_size)
            mask_dims = ()
        else:
            raise ValueError("Not supported: ", custom_transform)

        # save
        h5_file = h5py.File(filename, 'w')

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
            elif key == 'crop_name':
                h5_file.create_dataset(key,
                                       shape=(n_samples,), 
                                       chunks=(chunk_size,), 
                                       dtype=h5py.string_dtype())
                vals = np.array(raw_ds[key])
            h5_file[key][:] = vals
            h5_file.flush()
        h5_file.attrs['time_created'] = str(datetime.datetime.now())
        print(f'closing {filename}')
        h5_file.close()

def main(args):

    warnings.filterwarnings("ignore")

    preprocessor = Preprocessor(args)
    preprocessor.run() 
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-source', type=str, choices=['planet', 'planet-5', 'sentinel-1', 'sentinel-2'], help='Data source')
    parser.add_argument('--raw-data-dir', type=str,
                        default='/work/ka1176/shared_data/2021-ai4food/raw_data/')
    parser.add_argument('--target-sub-dir', type=str, default='default', help='subdirectory for storing the processed data in /dev_data/{region}/{source}')
    parser.add_argument('--region', type=str, choices=['south-africa', 'germany'],
                        default='south-africa', help='Select region')
    parser.add_argument('--n-processes', type=int, default=64)
    parser.add_argument('--min-area-to-ignore', type=float, default=1000, help='Fields below minimum area are ignored')
    parser.add_argument('--overwrite', action='store_true', help='overwrite npz data')
    # transformer arguments
    parser.add_argument('--t-spatial-encoder', action='store_true', help='Transformer variable')
    parser.add_argument('--t-normalize', action='store_true', help='Transformer variable')
    parser.add_argument('--t-image-size', type=int, default=32, help='Transformer variable')
    parser.add_argument('--t-random-extraction', type=int, default=0, help='Transformer variable')

    args = parser.parse_args()

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***')
    pass
    main(args)
