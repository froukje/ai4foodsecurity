#!/usr/bin/env python

from torch.utils.data import Dataset


class EarthObservationDataset(Dataset):
    '''
    Parent class for Earth Observation Datasets

    args namespace should provide

    split : {train, test}
    fill_value : fill value for masked pixels (e.g. 0 / None)

    '''
    super().__init__()

    def __init__(self, args):
        self.args = args
        self.h5_file = h5py.File(os.path.join(self.args.dev_data_dir, f'{args.split}_data.h5'), 'r')

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        X = self.X[idx]
        label = self.labels[idx]
        mask = self.mask[idx]

        X[~mask] = self.args.fill_value

        return X, label

class Sentinel2Dataset(EarthObservationDataset):
    '''
    Sentinel 2 Dataset

    args namespace should provide

    data_dir : 
    split    :
    bands    : List of Sentinel 2 bands
    ndvi     : If TRUE, include the NDVI in a band like fashion
    '''
    super().__init__()

    def __init__(self, args): 
        self.X = self.h5_file['image_stack'][:]
        self.mask = self.h5_file['mask'][:].astype(bool)
        self.feature = self.h5_file['feature'][:]
        self.labels = self.h5_file['labels'][:]

        # add NDVI
        if args.ndvi:
            ndvi = _calc_ndvi(self.X)
            self.X = np.concatenate([self.X, ndvi], axis=1)


    def __len__(self):
        return super().__len__(self)

    def __getitem__(self, idx):
        return super().__getitem__(self, idx)
        

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
    super().__init__()

    def __init__(self, ...):
        pass

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        pass



class PlanetDataset(EarthObservationDataset):
    '''
    Planet Dataset
    '''
    super().__init__()

    def __init__(self, ...):
        pass

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        pass

