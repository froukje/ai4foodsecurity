#!/usr/bin/env python

try:
    import nni
except ImportError:
    pass

import argparse
import os
import sys
import h5py
from evaluation_utils import metrics, save_predictions

sys.path.append('../notebooks/starter_files/')
from utils.data_transform import PlanetTransform
from utils.baseline_models import SpatiotemporalModel
from utils import train_valid_eval_utils as tveu

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from datasets import EarthObservationDataset, PlanetDataset

import numpy as np
import geopandas as gpd
import pandas as pd
import copy
import time


def main(args):
   
    # construct the dataset
    test_dataset = PlanetDataset(args)
    # if training, split dataset in train and valid
    if args.split=='train':
        # lengths of train and valid datasets
        train_length = int(len(test_dataset) * 0.8)
        valid_length = len(test_dataset) - train_length
        lengths = [train_length, valid_length]
        train_dataset, valid_dataset = torch.utils.data.random_split(test_dataset, 
                                                lengths=lengths, 
                                                generator=torch.Generator().manual_seed(42))

    # TODO read names from h5 file!    
    # Read label ids and names
    label_ids = np.unique(test_dataset.labels)+1
    train_labels_dir = os.path.join(args.raw_data_dir, args.label_dir)
    train_labels = gpd.read_file(train_labels_dir)
    #label_ids = train_labels['crop_id'].unique()
    label_names = train_labels['crop_name'].unique()

    # sort label ids and names
    zipped_lists = zip(label_ids, label_names)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    label_ids, label_names = [list(tuple) for tuple in tuples]

    print(f'label_ids: {label_ids}')
    print(f'label_names: {label_names}\n')

    # Initialize data loaders
    if args.split == 'train':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8)
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    # set device to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instatiate the model
    model = SpatiotemporalModel(input_dim=args.input_dim, num_classes=len(label_ids), sequencelength=args.sequence_length, device=device)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f'\nDevice {device}')

    # Initialize model optimizer and loss criterion:
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = CrossEntropyLoss(reduction="mean")

    # training
    best_loss = np.inf
    best_epoch = 0
    patience_count = 0
    all_train_losses = []
    all_valid_losses = []
    log_scores= []

    if args.split=='train':
        for epoch in range(args.max_epochs):
            # train
            model.train()
            start_time = time.time()
            print(f'\nEpoch: {epoch}')
    
            train_loss = tveu.train_epoch(model, optimizer, loss_criterion, train_loader, device=device)
            train_loss = train_loss.cpu().detach().numpy()[0]
            all_train_losses.append(train_loss)

            print(f'Training took {(time.time() - start_time) / 60:.2f} minutes, train_loss: {train_loss:.4}')
            start_time = time.time()

            # validation
            valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(model, loss_criterion, valid_loader, device=device)
            valid_loss = valid_loss.cpu().detach().numpy()[0]
            all_valid_losses.append(valid_loss)

            # calculate metrics
            scores = metrics(y_true.cpu(), y_pred.cpu())
            scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
            scores["epoch"] = epoch
            scores["train_loss"] = train_loss
            scores["valid_loss"] = valid_loss
            log_scores.append(scores)
            print(f'Validation took {(time.time() - start_time) / 60:.2f} minutes, valid_loss: {valid_loss:.4f}')

            # early stopping
            if valid_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_preds = y_pred
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == args.patience:
                print(f'no improvement for {args.patience} epochs -> early stopping')
                break
        
            # save checkpoints
            save_model_path = os.path.join(args.target_dir, 'best_model.pt')
            torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(), epoch=epoch, log=log_scores)
                    , save_model_path)
            print(f'saved best model to {save_model_path}')
        
            # save training and validation history
            with open(os.path.join(args.target_dir, 'train_losses.txt'), 'w') as f:
                for tl in all_train_losses:
                    f.write(f'{tl:.4f}\n')
            with open(os.path.join(args.target_dir, 'valid_losses.txt'), 'w') as f:
                for vl in all_valid_losses:
                    f.write(f'{vl:.4f}\n')
       
            print(f"\nINFO: Epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} " + scores_msg) 

    # make predictions   
    if args.save_preds:
        if args.split == 'train':
            test_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8)
        else:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)
            save_model_path = os.path.join(args.target_dir, 'best_model.pt')
        
        print(f'\nINFO: saving predictions from the {args.save_preds} set')
        save_predictions(save_model_path, model, test_loader, device, label_ids, label_names, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', type=str, default='/mnt/lustre02/work/ka1176/shared_data/2021-ai4food/raw_data/')
    parser.add_argument('--label-dir', type=str, default='ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson')
    parser.add_argument('--dev-data-dir', type=str, default='/mnt/lustre02/work/ka1176/shared_data/2021-ai4food/dev_data/planet_5day/default')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test']) 
    parser.add_argument('--save_preds', action='store_true', default=False) 
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint-epoch', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=4)
    parser.add_argument('--sequence-length', type=int, default=74)
    parser.add_argument('--min-area-to-ignore', type=int, default=1000)
    parser.add_argument('--ndvi', action='store_true', default=False)
    parser.add_argument('--fill-value', type=bool, default=0)
    args = parser.parse_args()

    print('\nbegin args key / value')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('end args keys / value\n')

    main(args)


