#!/usr/bin/env python

from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import geopandas as gpd
from torch import randn
import time
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter

class EarthObservationDataset(Dataset):
    '''
    Parent class for Earth Observation Datasets

    Preprocessed data is loaded from path:

    -- in args namespace --
      dev_data_dir    : file path on mistral, {germany, south africa}
      input_data[0]   : data source {planet, sentinel-1, sentinel-2}
      input_data_type : {extracted, extracted-640}
      split           : {train, test}

      If args.include_extras, the crop_area and crop_len are included for each sample
    '''

    def __init__(self, args):
        super().__init__()
        self.args = args
        
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

        # remove a fully masked sample from the Germany training data
        if self.args.nr_classes == 9 and self.args.split == 'train':
            if self.args.input_data_type == 'extracted':
                bad_idx = [1225]
            elif self.args.input_data_type == 'extracted-640':
                bad_idx = [12250, 12251, 12252, 12253, 12254, 12255, 12256, 12257, 12258, 12259]
            else:
                bad_idx = []
            self.X = np.delete(self.X, bad_idx, axis=0)
            self.mask = np.delete(self.mask, bad_idx, axis=0)
            self.fid = np.delete(self.fid, bad_idx, axis=0)
            self.labels = np.delete(self.labels, bad_idx, axis=0)
        
    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        X = self.X[idx]
        label = self.labels[idx]
        mask = self.mask[idx]
        fid = self.fid[idx]
        if self.extra_features is not None:
            extra_f = self.extra_features[idx]
        else: extra_f = np.zeros_like(1)
            
        return (X, mask, fid, extra_f), label

class Sentinel2Dataset(EarthObservationDataset):
    '''
    Sentinel 2 Dataset

    Calculates indices:

    NDVI
    leaf index
    moisture index
    drought index

    If args.drop_channels_sentinel2, drop all spectral bands, else keep
    spectral bands at 10m and 20m spatial resolution
    '''

    def __init__(self, args): 
        super().__init__(args)
        
        # Sentinel-2 Band Information
        band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'] # from AI4EO challenge

        # calculate indices
        ix_B4 = band_names.index('B04')
        ix_B5 = band_names.index('B05')
        ix_B8 = band_names.index('B08')
        ix_B8A = band_names.index('B8A')
        ix_B11 = band_names.index('B11')
        ix_B12 = band_names.index('B12')

        ndvi = Sentinel2Dataset._calc_two_band_index(self.X, ix_B8, ix_B4) # vegetation
        nlfi = Sentinel2Dataset._calc_two_band_index(self.X, ix_B8A, ix_B5) # leaf
        nmoi = Sentinel2Dataset._calc_two_band_index(self.X, ix_B8A, ix_B11) # moisture
        nbdi = Sentinel2Dataset._calc_three_band_index(self.X, ix_B8A, ix_B11, ix_B12) # multi-band drought


        # pixel-wise interpolation
        # Dropped as it did not increase the accuracy
        #clp = self.X[:, :, -1, :] # cloud probability is attached as the last band
        #clp = clp * 1e4 / 255 # transform to clp in [0 ... 1] where 1 = fully covered by clouds

        #ndvi = Sentinel2Dataset._interpolate(ndvi, clp, args.cloud_probability_threshold, args.sentinel_2_spline)
        #nlfi = Sentinel2Dataset._interpolate(nlfi, clp, args.cloud_probability_threshold, args.sentinel_2_spline)
        #nmoi = Sentinel2Dataset._interpolate(nmoi, clp, args.cloud_probability_threshold, args.sentinel_2_spline)
        #nbdi = Sentinel2Dataset._interpolate(nbdi, clp, args.cloud_probability_threshold, args.sentinel_2_spline)

        # stack all bands
        if self.args.drop_channels_sentinel2:
            self.X = np.stack([ndvi, nlfi, nmoi, nbdi], axis=2).squeeze()
        else:
            self.X = np.stack([np.expand_dims(self.X[:, :, 1], axis=2),
                               np.expand_dims(self.X[:, :, 2], axis=2),
                               np.expand_dims(self.X[:, :, 3], axis=2),
                               np.expand_dims(self.X[:, :, 4], axis=2),
                               np.expand_dims(self.X[:, :, 5], axis=2),
                               np.expand_dims(self.X[:, :, 6], axis=2),
                               np.expand_dims(self.X[:, :, 7], axis=2),
                               np.expand_dims(self.X[:, :, 8], axis=2),
                               np.expand_dims(self.X[:, :, 9], axis=2),
                               np.expand_dims(self.X[:, :, 11], axis=2),
                               np.expand_dims(self.X[:, :, 12], axis=2), 
                               ndvi, nlfi, nmoi, nbdi], axis=2).squeeze()
        print('Final shape for Sentinel-2 image stack', self.X.shape)

        
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

    If args.nri, calculates indices:

    Normalized radar vegetation index (RVI)

    If args.drop_channels_sentinel1, drop all bands

    If args.savgol_filter, smooth the RVI with a Savitzky Golay filter

    If args.split_nri, create two separate RVI channels for each observation angle
    '''
    def __init__(self, args):
        super().__init__(args)
        
        self.X = self.X[:, :, :2, :] # only VV and VH (2) is the angle

        # ! -- Germany test data has different length, cut to to train data length
        if args.nr_classes == 9 and args.split == 'test':
            print('Sentinel-1 shape before cut: ', self.X.shape)
            self.X = self.X[:, 1:-1]
            print('Sentinel-1 shape after cut (test set germany): ', self.X.shape)
              
        if args.nri:
            nri = Sentinel1Dataset._calc_rvi(self.X, self.args.savgol_filter)
            nri = np.expand_dims(nri, axis=2) # changed axis from 1 to 2
            if args.drop_channels or args.drop_channels_sentinel1:
                if args.split_nri: # split into two channels for different angles
                    if self.args.savgol_filter:
                        raise ValueError("Do not use Savitzky Golay filter and NRI split together")
                    nri_odd = nri[:, ::2]
                    nri_even = nri[:, 1::2]
                    nri_min_len = min(nri_odd.shape[1], nri_even.shape[1])
                    self.X = np.concatenate([nri_odd[:, :nri_min_len], nri_even[:, :nri_min_len]], axis=2) 
                else:
                    self.X = nri
            else:
                self.X = np.concatenate([self.X, nri], axis=2) 

        print('Final shape for Sentinel-1 image stack: ', self.X.shape)

        '''
        # normalization of datasets min-max
        xmin=np.min(self.X, axis=(0,1,3))
        xmax=np.max(self.X, axis=(0,1,3))
        for i in range(self.X.shape[2]):
            self.X[:,:,i,:] = (self.X[:,:,i,:] - xmin[i])/(xmax[i] - xmin[i])
        '''
                
    @staticmethod
    def _calc_rvi(X, rvi_filter=False):
        VV = X[:,:,0,:]
        VH = X[:,:,1,:]
        dop = (VV/(VV+VH))
        m = 1 - dop
        radar_vegetation_index = (np.sqrt(dop))*((4*(VH))/(VV+VH))

        eps = 1e-9 # avoid zero values
        radar_vegetation_index = np.nan_to_num(radar_vegetation_index, nan=eps, posinf=eps, neginf=eps)

        if not rvi_filter:
            return radar_vegetation_index

        start_time = time.time()

        print('Start to apply Savitzky Golay Filter to Sentinel-1 RVI')
        for i in range(radar_vegetation_index.shape[0]):
            for j in range(radar_vegetation_index.shape[-1]):
                tmp = radar_vegetation_index[i, :, j]

                smooth_rvi = savgol_filter(tmp, 15, 3)

                radar_vegetation_index[i, :, j] = smooth_rvi

        print(f'Applied Savitzky Golay Filter to Sentinel-1 RVI in {time.time() - start_time:.1f} seconds')

        return radar_vegetation_index


class PlanetDataset(EarthObservationDataset):
    '''
    Planet Dataset

    If args.ndvi, calculates indices:

    Normalized difference vegetation index (NDVI)

    If args.drop_channels, drop all bands

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

        if args.nr_classes == 9 and args.vegetation_period: # Germany
            ix_train_start = 83 # defined as the minimum of NDVI
            ix_test_start  = 90 # such that NDVI maxima match
            vg_length      = 180 # length of the vegetation period

            if args.split == 'train':
                self.X = self.X[:, ix_train_start:ix_train_start + vg_length]
            elif args.split == 'test':
                self.X = self.X[:, ix_test_start:ix_test_start + vg_length]

        print('Final shape for Planet image stack', self.X.shape)

        '''
        # normalization of datasets min-max
        xmin=np.min(self.X, axis=(0,1,3))
        xmax=np.max(self.X, axis=(0,1,3))
        for i in range(self.X.shape[2]):
            self.X[:,:,i,:] = (self.X[:,:,i,:] - xmin[i])/(xmax[i] - xmin[i])
        '''

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
    '''
    Class for a combined dataset, holds PlanetDataset, Sentinel1Dataset, Sentinel2Dataset
    as specified by args.input_data
    '''
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
            print('Assert dataset shape match', i-1, i)
            assert (self.datasets[i-1].fid==self.datasets[i].fid).all(),'s1, s2 and/or planet not sorted correctly'
    
    def __len__(self):
        return len(self.datasets[0].labels) 
    
    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    
class AddGaussianNoise(object):
    '''
    Add Gaussian noise to a sample
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
