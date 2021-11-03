"""
Adapted from:

This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines a data reader for Sentinel-1 eath observation data
"""

import os
from torch.utils.data import Dataset
import zipfile
import tarfile
from sh import gunzip
from glob import glob
import pickle
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from tqdm import tqdm


class S1Reader(Dataset):
    """
    THIS CLASS INITIALIZES THE DATA READER FOR SENTINEL-1 DATA
    """
    def __init__(self, input_dir, label_dir, label_ids=None, transform=None, min_area_to_ignore = 1000, selected_time_points=None):
        '''
        THIS FUNCTION INITIALIZES DATA READER.
        :param input_dir: directory of input images in zip format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param label_ids: an array of crop IDs in order. if the crop labels in GeoJSON data is not started from index 0 it can be used. Otherwise it is not required.
        :param transform: data transformer function for the augmentation or data processing
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :param selected_time_points: If a sub set of the time series will be exploited, it can determine the index of those times in a given time series dataset

        :return: None
        '''

        self.data_transform = transform
        self.selected_time_points=selected_time_points
        self.crop_ids = label_ids
        if label_ids is not None and not isinstance(label_ids, list):
            self.crop_ids = label_ids.tolist()

        self.npyfolder = input_dir.replace(".zip", "/time_series")
        self.labels = S1Reader._setup(input_dir, label_dir,self.npyfolder,min_area_to_ignore)

    def __len__(self):
        """
        THIS FUNCTION RETURNS THE LENGTH OF DATASET
        """
        return len(self.labels)

    def __getitem__(self, item):
        """
        THIS FUNCTION ITERATE OVER THE DATASET BY GIVEN ITEM NO AND RETURNS FOLLOWINGS:
        :return: image_stack in size of [Time Stamp, Image Dimension (Channel), Height, Width] , crop_label, field_mask in size of [Height, Width], field_id
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
                raise
        else:
            print("ERROR: {} is a missing...".format(npyfile))
            raise

        if self.data_transform is not None:
            image_stack, mask = self.data_transform(image_stack, mask)

        if self.selected_time_points is not None:
            image_stack = image_stack[self.selected_time_points]

        if self.crop_ids is not None:
            label = self.crop_ids.index(feature.crop_id)
        else:
            label = feature.crop_id

        return image_stack, label, mask, feature.fid


    @staticmethod
    def _setup(rootpath, labelgeojson, npyfolder, min_area_to_ignore=1000):
        """
        THIS FUNCTION PREPARES THE PLANET READER BY SPLITTING AND RASTERIZING EACH CROP FIELD AND SAVING INTO SEPERATE FILES FOR SPEED UP THE FURTHER USE OF DATA.

        This utility function unzipps a dataset and performs a field-wise aggregation.
        results are written to a .npz cache with same name as zippath

        :param rootpath: directory of input images
        :param labelgeojson: directory of ground-truth polygons in GeoJSON format
        :param npyfolder: folder to save the field data for each field polygon
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :return: labels of the saved fields
        """

        with open(os.path.join(rootpath, "bbox.pkl"), 'rb') as f:
            bbox = pickle.load(f)
            crs = str(bbox.crs)
            minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y

        labels = gpd.read_file(labelgeojson)
        # project to same coordinate reference system (crs) as the imagery
        ignore = labels.geometry.area > min_area_to_ignore
        print(f"INFO: Ignoring {(~ignore).sum()}/{len(ignore)} fields with area < {min_area_to_ignore}m2")
        labels = labels.loc[ignore]
        labels = labels.to_crs(crs) #TODO: CHECK IF NECESSARY

        vv = np.load(os.path.join(rootpath, "vv.npy"))
        vh = np.load(os.path.join(rootpath, "vh.npy"))
        bands = np.stack([vv[:,:,:,0],vh[:,:,:,0]], axis=3)
        _, width, height, _ = bands.shape

        bands=bands.transpose(0, 3, 1, 2)

        transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

        fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                                  transform=transform, out_shape=(width, height))
        assert len(np.unique(fid_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                             f"Does the label geojson {labelgeojson} cover the region defined by {rootpath}?"

        crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                                  transform=transform, out_shape=(width, height))
        assert len(np.unique(crop_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                              f"Does the label geojson {labelgeojson} cover the region defined by {rootpath}?"


        for index, feature in tqdm(labels.iterrows(), total=len(labels), position=0, leave=True, desc="INFO: Extracting time series into the folder: {}".format(npyfolder)):

            npyfile = os.path.join(npyfolder, "fid_{}.npz".format(feature.fid))
            if not os.path.exists(npyfile):

                left, bottom, right, top = feature.geometry.bounds
                window = rio.windows.from_bounds(left, bottom, right, top, transform)

                row_start = round(window.row_off)
                row_end = round(window.row_off) + round(window.height)
                col_start = round(window.col_off)
                col_end = round(window.col_off) + round(window.width)

                image_stack = bands[:, :,row_start:row_end, col_start:col_end]
                mask = fid_mask[row_start:row_end, col_start:col_end]
                mask[mask != feature.fid] = 0
                mask[mask == feature.fid] = 1
                os.makedirs(npyfolder, exist_ok=True)
                np.savez(npyfile, image_stack=image_stack.astype(np.float32), mask=mask.astype(np.float32), feature=feature.drop("geometry").to_dict())

        return labels




if __name__ == '__main__':
    """
    EXAMPLE USAGE OF DATA READER
    """

    rootpath = "../data/dlr_fusion_competition_germany_train_source_sentinel_1/dlr_fusion_competition_germany_train_source_sentinel_1_33N_18E_242N_2018/"
    labelgeojson = "../data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson"
    ds = S1Reader(rootpath, labelgeojson,selected_time_points=[1,2,3])
    X,y,m,fid = ds[0]
