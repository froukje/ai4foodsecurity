#!/usr/bin/env python

import sys
import os

import argparse

# make sure this is on pythonpath
sys.path.append('../notebooks/starter_files/')
from utils.planet_reader import PlanetReader

from multiprocessing import Pool

def main(args):
    
    # needs to be list of lists for starmap
    selected_time_points = [[i] for i in range(365)] # select all days

    if args.region == 'south-africa':
        input_dir = 'ref_fusion_competition_south_africa_train_source_planet'
        label_dir = 'ref_fusion_competition_south_africa_train_labels'
    elif args.region == 'germany':
        raise ValueError('Not implemented', args.region)

    if args.five_day:
        input_dir += '_5day'

    if args.train_set == 1: # only for S-Africa actually
        label_dir = os.path.join(label_dir, 'ref_fusion_competition_south_africa_train_labels_34S_19E_258N')
    elif args.train_set == 2:
        label_dir = os.path.join(label_dir, 'ref_fusion_competition_south_africa_train_labels_34S_19E_259N')

    input_dir = os.path.join(args.raw_data_dir, input_dir) 
    label_dir = os.path.join(args.raw_data_dir, label_dir, 'labels.geojson')

    with Pool(args.n_processes) as p:
        p.starmap(PlanetReader, zip([input_dir]*len(selected_time_points), 
	                            [label_dir]*len(selected_time_points), 
				    selected_time_points))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', type=str,
                        default='/work/ka1176/shared_data/2021-ai4food/raw_data/')
    parser.add_argument('--region', type=str, choices=['south-africa', 'germany'],
                        default='south-africa', help='Select region')
    parser.add_argument('--five-day', type=bool, default=False)
    parser.add_argument('--train-set', type=int, choices=[1, 2], default=1, help='Two train sets to choose')
    parser.add_argument('--n-processes', type=int, default=1)

    args = parser.parse_args()

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***')

    main(args)







