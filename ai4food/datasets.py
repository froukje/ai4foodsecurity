#!/usr/bin/env python

from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import geopandas as gpd
import time

from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter

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
        
        ### for now only - remove when we have proper path to sentinel-1 data
        self.h5_file = h5py.File(os.path.join(args.dev_data_dir, args.input_data[0], args.input_data_type, f'{args.split}_data.h5'), 'r')

        self.X = self.h5_file['image_stack'][:].astype(np.float32)
        self.mask = self.h5_file['mask'][:].astype(bool)
        self.fid = self.h5_file['fid'][:]
        self.labels = self.h5_file['label'][:]
        self.labels = self.labels - 1 # generated datafiles with classes from 1 ... k --> 0 ... k-1

        if np.sum(np.isnan(self.X)) > 0:
            print('WARNING: Filled NaNs and INFs with 0 in ', os.path.join(args.dev_data_dir, args.input_data[0], args.input_data_type, f'{args.split}_data.h5'))
            self.X = np.nan_to_num(self.X, nan=0, posinf=0, neginf=0)
        
        if args.include_extras:
            labels_path = os.path.join(args.dev_data_dir,'labels_combined.geojson')
            print('Adding extra features from ', labels_path)
            extras = gpd.read_file(labels_path)

            crop_area = []
            crop_len  = []

            extras_fid = extras["fid"].values
            extras_crop_area = extras["NORMALIZED_SHAPE_AREA"].values
            extras_crop_len = extras["NORMALIZED_SHAPE_LEN"].values

            for ii, ffid in enumerate(self.fid):
                ix = np.where(extras_fid==ffid)[0][0]
                crop_area.append(extras_crop_area[ix])
                crop_len.append(extras_crop_len[ix])

                if ii%(len(self.fid)//20) == 0:
                    print(f'... finished {ii:8d}/{len(self.fid):8d} entries ({ii/len(self.fid)*100:.1f} %)')

            self.extra_features = np.array([crop_area, crop_len]).T
        else:
            self.extra_features = None
            
        self.augmentation = args.augmentation
        if self.augmentation:
            '''
            paper: For aug- mentation purpose, we add a random Gaussian noise to x(t) with standard deviation 10−2 and clipped to 5.10−2 on the values of the pixels, normalized channel-wise and for each date individually.
            '''
            self.gaussian_noise_aug = AddGaussianNoise(mean=0, std=0.01)
        
        '''
        # normalization of datasets min-max
        xmin=np.min(self.X, axis=(0,2,3))
        xmax=np.min(self.X, axis=(0,2,3))
        '''
    
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
        else: extra_f = np.zeros_like(1)
        
        if self.augmentation:
            X = self.gaussian_noise_aug(X)
            
        return (X, mask, fid, extra_f), label

class Sentinel2Dataset(EarthObservationDataset):
    '''
    Sentinel 2 Dataset

    args namespace should provide

    cloud_threshold - interpolate values exceeding threshold

    '''

    def __init__(self, args): 
        super().__init__(args)

        # Sentinel-2 Band Information
        band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'] # from AI4EO challenge

        # calculate indices
        ix_B8 = band_names.index('B08')
        ix_B4 = band_names.index('B04')

        ndvi = Sentinel2Dataset._calc_two_band_index(self.X, ix_B8, ix_B4)

        # pixel-wise interpolation
        clp = self.X[:, :, -1, :] # cloud probability is attached as the last band
        clp = clp * 1e4 / 255 # transform to clp in [0 ... 1] where 1 = fully covered by clouds

        ndvi = Sentinel2Dataset._interpolate(ndvi, clp, args.cloud_probability_threshold, args.sentinel_2_spline)

        self.X = ndvi

    @staticmethod
    def _calc_two_band_index(X, i, j):
        '''
        Calculate a normalized index using two bands
        at indices i,j

        (B1 - B2) / (B1 + B2)
        '''

        B1 = X[:, :, i, :]
        B2 = X[:, :, j, :]

        N = (B1 - B2) / (B1 + B2)
        N = np.nan_to_num(N, posinf=0, neginf=0)
        N = np.expand_dims(N, axis=2)

        return N

    @staticmethod
    def _calc_three_band_index(X, i, j, k):
        '''
        Calculate a normalized index using three bands
        at indices i,j

        (B1 - (B2 - B3)) / (B1 + (B2 - B3))
        '''

        B1 = X[:, :, i, :]
        B2 = X[:, :, j, :]
        B3 = X[:, :, k, :]

        N = (B1 - (B2 - B3)) / (B1 + (B2 - B3))
        N = np.nan_to_num(N, posinf=0, neginf=0)
        N = np.expand_dims(N, axis=2)

        return N

    @staticmethod
    def _interpolate(normalized_index, cloud_probability, cloud_probability_threshold, sentinel_2_spline):
        '''
        Interpolate an index
        Remove values below the cloud probability threhold

        Use l-spline to reconstruct missing values

        Arguments:

        normalized_index - index to be normalized [Samples, Time, 1, Pixels]
        cloud_probability - all cloud probabilities
        cloud_probability_threshold - upper limit for cloud coverage
        sentinel_2_spline - l-spline for interpolation

        '''
        # time steps
        n_t = normalized_index.shape[1]
        x = np.arange(n_t)

        start_time=time.time()

        for sample_ix in range(normalized_index.shape[0]):
            for pixel_ix in range(normalized_index.shape[-1]):
                y = normalized_index[sample_ix, :, 0, pixel_ix] # measured normalized_index for this pixel
                c = cloud_probability[sample_ix, :, pixel_ix] # cloud probability for this pixel

                good_ix = np.where(c < cloud_probability_threshold)[0]

                if not 0 in good_ix:
                    good_ix = list([0]) + list(good_ix)
                if not n_t - 1 in good_ix:
                    good_ix = list(good_ix) + list([n_t-1])

                good_x = x[good_ix]
                good_y = y[good_ix]

                # g = splrep(good_x, good_y, k=3, w=weights)
                g = splrep(good_x, good_y, k=sentinel_2_spline)
                interp_y1 = splev(x, g)

                # replace the index with the interpolated index
                normalized_index[sample_ix, :, 0, pixel_ix] = interp_y1

        print(f'Interpolated INDEX for {normalized_index.shape[0]} samples with {normalized_index.shape[-1]} pixels in {time.time() - start_time:.1f} seconds')

        return normalized_index

class Sentinel1Dataset(EarthObservationDataset):
    '''
    Sentinel 1 Dataset
    '''
    def __init__(self, args):
        super().__init__(args)
        if args.nri:
            nri = Sentinel1Dataset._calc_rvi(self.X)
            nri = np.expand_dims(nri, axis=2) # changed axis from 1 to 2
            if args.drop_channels:
                self.X = nri
            else:
                self.X = np.concatenate([self.X, nri], axis=2) # changed axis from 1 to 2  
                
    @staticmethod
    def _calc_rvi(X):
        VV = X[:,:,0,:]
        VH = X[:,:,1,:]
        dop = (VV/(VV+VH))
        m = 1 - dop
        radar_vegetation_index = (np.sqrt(dop))*((4*(VH))/(VV+VH))
        radar_vegetation_index = np.nan_to_num(radar_vegetation_index)

        start_time = time.time()

        print('Start to apply Savitzky Golay Filter to Sentinel-1 RVI')
        for i in range(radar_vegetation_index.shape[0]):
            for j in range(radar_vegetation_index.shape[-1]):
                tmp = radar_vegetation_index[i, :, j]

                smooth_rvi = savgol_filter(tmp, 15, 3)

                radar_vegetation_index[i, :, j] = smooth_rvi

        print('Applied Savitzky Golay Filter to Sentinel-1 RVI in {time.time() - start_time:.1f} seconds')

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
            if args.drop_channels:
                self.X = ndvi
            else:
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
        ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi)
        return ndvi


    
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
            elif input_data=='planet-5':
                args.input_data = ['planet-5']
                planet5_dataset = PlanetDataset(args)
                self.datasets.append(planet5_dataset)

            elif input_data=='sentinel-1':
                args.input_data = ['sentinel-1']
                sentinel1_dataset = Sentinel1Dataset(args)
                self.datasets.append(sentinel1_dataset)
            elif input_data=='sentinel-2':
                args.input_data = ['sentinel-2']
                sentinel2_dataset = Sentinel2Dataset(args)
                self.datasets.append(sentinel2_dataset)
        args.input_data = self.input_data 
        for i in range(1, len(self.datasets)):
            assert (self.datasets[i-1].fid==self.datasets[i].fid).all(),'s1, s2 and/or planet not sorted correctly'
    
    def __len__(self):
        return len(self.datasets[0].labels) 
    
    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, X):
        res = (X + np.random.normal(size=X.shape) * self.std + self.mean).astype(np.float32)
        return res
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
