#!/usr/bin/env python

from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import geopandas as gpd

class EarthObservationDataset(Dataset):
    '''
    Parent class for Earth Observation Datasets

    args namespace should provide

    split : {train, test}
    fill_value : fill value for masked pixels (e.g. 0 / None)

    '''

    def __init__(self, args):
        super().__init__()
        self.args = args

        ### for now only
        if args.input_data[0]=='sentinel-1':
            self.h5_file = h5py.File(f'{args.split}_data.h5', 'r')
        else:
            self.h5_file = h5py.File(os.path.join(args.dev_data_dir, args.input_data[0], args.input_data_type, f'{args.split}_data.h5'), 'r')

        self.X = self.h5_file['image_stack'][:].astype(np.float32)
        self.mask = self.h5_file['mask'][:].astype(bool)
        self.fid = self.h5_file['fid'][:]
        self.labels = self.h5_file['label'][:]
        
        if args.include_extras:
            labels_path='/work/ka1176/shared_data/2021-ai4food/labels_combined.geojson' # when moved to data dir change to os.path.join(data_dir,'labels_combined.geojson')
            extras=gpd.read_file(labels_path)
            crop_area = np.array(extras['SHAPE_AREA'])
            crop_length = np.array(extras['SHAPE_LEN'])
            self.extra_features = np.array([crop_area, crop_length]).T
        else:
            self.extra_features = None
    
    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        X = self.X[idx]
        label = self.labels[idx]
        mask = self.mask[idx]
        fid = self.fid[idx]
        #X[:,:,~mask] = self.args.fill_value
        if self.extra_features is not None:
            extra_f = self.extra_features[idx]
            return X, label, mask, fid, extra_f
        else:
            return X, label, mask, fid

class Sentinel2Dataset(EarthObservationDataset):
    '''
    Sentinel 2 Dataset

    args namespace should provide

    bands    : List of Sentinel 2 bands
    ndvi     : If TRUE, include the NDVI in a band like fashion
    '''

    def __init__(self, args): 
        super().__init__(args)
        # TODO select bands

        # add NDVI
        if args.ndvi:
            ndvi = Sentinel2Dataset._calc_ndvi(self.X)
            ndvi = np.expand_dims(ndvi, axis=1)
            self.X = np.concatenate([self.X, ndvi], axis=1)


    @staticmethod
    def _calc_ndvi(X):
        '''
        Calculate the normalized vegetation index

        NDVI = (NIR - RED) / (NIR + RED)

        '''

        nir = X[:, 7]
        red = X[:, 3]

        return (nir - red) / (nir + red)

class Sentinel1Dataset(EarthObservationDataset):
    '''
    Sentinel 1 Dataset
    '''
    def __init__(self, args):
        super().__init__(args)
        if args.nri:
            nri = Sentinel1Dataset._calc_rvi(self.X)
            nri = np.expand_dims(nri, axis=2) # changed axis from 1 to 2
            self.X = np.concatenate([self.X, nri], axis=2) # changed axis from 1 to 2  
            
    @staticmethod
    def _calc_rvi(X):
        VV = X[:,:,0,:]
        VH = X[:,:,1,:]
        dop = (VV/(VV+VH))
        m = 1 - dop
        radar_vegetation_index = (np.sqrt(dop))*((4*(VH))/(VV+VH))
        print('lalalla', radar_vegetation_index.shape)
        return radar_vegetation_index


class PlanetDataset(EarthObservationDataset):
    '''
    Planet Dataset

    args namespace:
    ndvi : if TRUE, include the NDVI in a band like fashion
    '''

    def __init__(self, args): 
        super().__init__(args)
        if args.ndvi:
            ndvi = PlanetDataset._calc_ndvi(self.X)
            ndvi = np.expand_dims(ndvi, axis=2) # changed axis from 1 to 2
            self.X = np.concatenate([self.X, ndvi], axis=2) # changed axis from 1 to 2

    @staticmethod
    def _calc_ndvi(X):
        '''
        Calculate the normalized vegetation index

        NDVI = (NIR - RED) / (NIR + RED)

        '''
        #print(X.shape) #(4143, 244, 4, 64)
        nir = X[:, :, 3, :] # X[:, 3]
        red = X[:, :, 2, :] # X[:, 2]
        return (nir - red) / (nir + red)


    
class CombinedDataset(Dataset):
    def __init__(self, args): 
        super().__init__()
        self.datasets =[]
        self.input_data = args.input_data.copy()
        for input_data in self.input_data:
            if input_data=='planet':
                args.input_data = ['planet']
                planet_dataset = PlanetDataset(args)
                self.datasets.append(planet_dataset)
            elif input_data=='sentinel-1':
                args.input_data = ['sentinel-1']
                sentinel1_dataset = Sentinel1Dataset(args)
                self.datasets.append(sentinel1_dataset)
            elif input_data=='sentinel-2':
                args.input_data = ['sentinel-2']
                sentinel2_dataset = Sentinel2Dataset(args)
                self.datasets.append(sentinel2_dataset)
    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)