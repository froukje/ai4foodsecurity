"""
Adapted from the Planet data reader:

This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines a data reader for Planet Fusion eath observation data
"""

import os
import torch
from torch.utils.data import Dataset
import zipfile
import tarfile
from sh import gunzip
import glob
import pickle
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from tqdm import tqdm

from itertools import repeat
from multiprocessing import Pool, RawArray
import time

var_dict= {} # global variable for the multiprocessing

class PlanetReader(Dataset):
    """
    THIS CLASS INITIALIZES THE DATA READER FOR PLANET DATA
    """
    def __init__(self, input_dir, label_dir, output_dir=None, label_ids=None, transform=None, min_area_to_ignore = 1000,  selected_time_points=None, overwrite=True, n_processes=1):
        '''
        THIS FUNCTION INITIALIZES DATA READER.
        :param input_dir: directory of input images in TIF format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param output_dir: directory where to store the zipped processes patches
        :param label_ids: an array of crop IDs in order. if the crop labels in GeoJSON data is not started from index 0 it can be used. Otherwise it is not required.
        :param transform: data transformer function for the augmentation or data processing
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :param selected_time_points: If a sub set of the time series will be exploited, it can determine the index of those times in a given time series dataset
        :param overwrite: Overwrite the preprocessed data
        :param n_processes: Parallel processing during setup (default: single core)

        :return: None
        '''

        self.data_transform = transform
        self.selected_time_points=selected_time_points
        self.crop_ids = label_ids
        if label_ids is not None and not isinstance(label_ids, list):
            self.crop_ids = label_ids.tolist()

        if output_dir is None:
            self.npyfolder = input_dir.replace(".zip", "/time_series")
        else:
            self.npyfolder = output_dir
        self.labels = PlanetReader._setup(input_dir, label_dir,self.npyfolder,min_area_to_ignore, overwrite, n_processes)

    def __len__(self):
        """
         THIS FUNCTION RETURNS THE LENGTH OF DATASET
         """
        return len(self.labels)

    def __getitem__(self, item):
        """
        THIS FUNCTION ITERATE OVER THE DATASET BY GIVEN ITEM NO AND RETURNS FOLLOWINGS:
        :return: image_stack in size of [Time Stamp, Image Dimension (Channel), Height, Width] , crop_label, field_mask in size of [Height, Width], field_id, crop_name
        """

        feature = self.labels.iloc[item]

        npyfile = os.path.join(self.npyfolder, "fid_{}.npz".format(feature.fid))
        if os.path.exists(npyfile): # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                mask = object["mask"]
            except zipfile.BadZipFile:
                print("ERROR: {} is a bad zipfile...".format(npyfile))
                return None
        else:
            print("ERROR: {} is a missing...".format(npyfile))
            return None

        if self.data_transform is not None:
            image_stack, mask = self.data_transform(image_stack, mask)

        if self.selected_time_points is not None:
            image_stack = image_stack[self.selected_time_points]

        if self.crop_ids is not None:
            label = self.crop_ids.index(feature.crop_id)
        else:
            label = feature.crop_id

        return image_stack, label, mask, feature.fid, feature.crop_name


    @staticmethod
    def _setup(input_dir, label_dir, npyfolder, min_area_to_ignore=1000, overwrite=False, n_processes=1):
        """
        THIS FUNCTION PREPARES THE PLANET READER BY SPLITTING AND RASTERIZING EACH CROP FIELD AND SAVING INTO SEPERATE FILES FOR SPEED UP THE FURTHER USE OF DATA.
        :param input_dir: directory of input images in TIF format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param npyfolder: folder to save the field data for each field polygon
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :param overwrite: if TRUE, overwrite the previously setup data
        :param n_processes: Parallel processes for polygon extraction, default: 1
        :return: labels of the saved fields
        """

        '''
        I had to fix some things in the planet data reader.

        * Inputs: Use glob to only look for sr.tif, the superresolution images. 
          qa.tif is also contained, but it has a different number of channels
          From the data set description, these do not seem so relevant
        * Transform: All planet data was stored in one folder. 
          In the folder 
          /work/ka1176/shared_data/2021-ai4food/raw_data/ref_fusion_competition_south_africa_train_source_planet
          I created the two new subfolders

          ** ref_fusion_competition_south_africa_train_source_planet_34S_19E_258N
          ** ref_fusion_competition_south_africa_train_source_planet_34S_19E_259N

          and moved the individual tif folders accordingly. This way we use the right transform. 
          If this is not done, wrong images will be read from the patches from the wrong region (x2 in tifs)

          -- CA 03.11.21


        '''
        inputs = glob.glob(input_dir + '/*/sr.tif', recursive=True)
        tifs = sorted(inputs)
        labels = gpd.read_file(label_dir)

        # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
        tif_idx = 0
        #if '34S_19E_259N' in label_dir:
        #    tif_idx = -1
        
        with rio.open(tifs[tif_idx]) as image:
            crs = image.crs
            print('INFO: Coordinate system of the data is: {}'.format(crs))
            transform = image.transform

        mask = labels.geometry.area > min_area_to_ignore
        print(f"INFO: Ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area_to_ignore}m2")

        labels = labels.loc[mask]
        labels = labels.to_crs(crs) #TODO: CHECK IF REQUIRED

        # prepare for multiprocessing
        print('starting the multiprocessing preparation')
        # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-html
        #p_tifs = RawArray('f', tifs.flatten().shape[0])
        ## wrap buffers as np arrays for easier manipulation
        #p_tifs_np = np.frombuffer(p_tifs, dtype=np.float32).reshape(tifs.shape)
        # copy data to shared arrays
        start_time = time.time()
        #np.copyto(p_tifs_np, tifs.astype(np.float32))

        print(f'Copied data to shared arrays in {time.time() - start_time:.0f} seconds')

        def init_worker(input_dir, npyfolder, transform, overwrite):
            var_dict['input_dir'] = input_dir
            var_dict['npyfolder'] = npyfolder
            var_dict['transform'] = transform
            var_dict['overwrite'] = overwrite

        initargs = (input_dir, npyfolder, transform, overwrite)

        with Pool(n_processes, initializer=init_worker, initargs=initargs) as p:
            print(f'Starting {n_processes} parallel processes')
            start_time = time.time()
            p.starmap(PlanetReader._extract_field, labels.iterrows())
            print(f'Finished setup in {(time.time() - start_time) / 60:.2f} minutes')

        return labels


    @staticmethod
    def _extract_field(index, feature):
        '''
        Separate function for extracting the individual polygons
        For use with Pool
        '''

        # from buffer
        input_dir = var_dict['input_dir']
        npyfolder = var_dict['npyfolder']
        transform = var_dict['transform']
        overwrite = var_dict['overwrite']

        inputs = glob.glob(input_dir + '/*/sr.tif', recursive=True)
        tifs = sorted(inputs)

        npyfile = os.path.join(npyfolder, "fid_{}.npz".format(feature.fid))

        if overwrite or not os.path.exists(npyfile):
            #print('Starting with', feature.fid)

            left, bottom, right, top = feature.geometry.bounds
            window = rio.windows.from_bounds(left, bottom, right, top, transform)

            # reads each tif in tifs on the bounds of the feature. shape T x D x H x W
            image_stack = np.stack([rio.open(tif).read(window=window) for tif in tifs])

            with rio.open(tifs[0]) as src:
                win_transform = src.window_transform(window)

            out_shape = image_stack[0, 0].shape
            try:
                assert out_shape[0] > 0 and out_shape[1] > 0, \
                    "WARNING: fid:{} image stack shape {} is zero in one dimension".format(feature.fid,image_stack.shape)
            except:
                print('assertion failed', feature.fid)
                return None

            # rasterize polygon to get positions of field within crop
            mask = features.rasterize(feature.geometry, all_touched=True,transform=win_transform, out_shape=image_stack[0, 0].shape)

            #mask[mask != feature.fid] = 0
            #mask[mask == feature.fid] = 1
            os.makedirs(npyfolder, exist_ok=True)
            np.savez(npyfile, image_stack=image_stack, mask=mask, feature=feature.drop("geometry").to_dict())