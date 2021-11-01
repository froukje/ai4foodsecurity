#!/usr/bin/env python

try:
    import nni
except ImportError:
    pass

import argparse
import os
import sys
sys.path.append('../notebooks/starter_files/')
from utils.data_transform import PlanetTransform
from utils.planet_reader import PlanetReader
from utils.data_loader import DataLoader
from utils.baseline_models import SpatiotemporalModel
from utils import train_valid_eval_utils as tveu

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import numpy as np
import geopandas as gpd
import pandas as pd
import copy
import time


def predict(x, y, model, criterion, device, eval_=True):
    """
    applies a model to some input data;
    if eval_ is set to False the model is trained, else it is in inference mode
    x: input data
    y: target data
    model: model
    criterion: loss criterion
    eval_: True / False model set to inference mode yes / no
    """

    device = device
    y = y.to(device)
    x = x.to(device)

    if eval_:
        with torch.no_grad():
            y_hat = model(x)

    else: 
        y_hat = model(x)

    loss = criterion(y_hat, y)

    # get the predicted values off the GPU / off torch
    if torch.cuda.is_available():
        y_hat_values = y_hat.cpu().detach().numpy()
    else:
        y_hat_values = y_hat.detach().numpy()


    return loss, y_hat_values

def main(args):
   
    # read data
    train_data_dir = os.path.join(args.data_dir, args.input_dir)
    train_labels_dir = os.path.join(args.data_dir, args.label_dir)
    
    train_labels = gpd.read_file(train_labels_dir)

    # Get data transformer for planet images
    planet_transformer = PlanetTransform()

    # Initialize data reader for planet images
    label_ids = train_labels['crop_id'].unique()
    label_names = train_labels['crop_name'].unique()

    # sort label ids and names
    zipped_lists = zip(label_ids, label_names)
    sorted_pairs = sorted(zipped_lists)


    tuples = zip(*sorted_pairs)
    label_ids, label_names = [list(tuple) for tuple in tuples]

    print(f'label_ids: {label_ids}')
    print(f'label_names: {label_names}\n')

    planet_reader = PlanetReader(input_dir=train_data_dir,
                                 label_dir=train_labels_dir,
                                 label_ids=label_ids,
                                 transform=planet_transformer.transform,
                                 min_area_to_ignore=args.min_area_to_ignore)

    #Initialize data loaders
    data_loader=DataLoader(train_val_reader=planet_reader, validation_split=0.25)
    train_loader=data_loader.get_train_loader(batch_size=args.batch_size, num_workers=1)
    valid_loader=data_loader.get_validation_loader(batch_size=args.batch_size, num_workers=1)
   
    # set device to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instatiate the model
    model = SpatiotemporalModel(input_dim=args.input_dim, num_classes=len(label_ids), sequencelength=args.sequence_length, device=device)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f'\nDevice {device}')

    #Initialize model optimizer and loss criterion:
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = CrossEntropyLoss(reduction="mean")

    # training
    best_loss = np.inf
    best_epoch = 0
    patience_count = 0
    all_train_losses = []
    all_valid_losses = []

    for epoch in range(args.max_epochs):
        # train
        model.train()
        start_time = time.time()
        print(f'\nEpoch: {epoch}')
        train_losses = []
        for idx, batch in enumerate(train_loader):
            inputs, target, _, _ = batch
            loss, _ = predict(inputs, target, model, loss_criterion, device, eval_=False)
            train_losses.append(loss.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(np.array(train_losses))
        all_train_losses.append(train_loss)
        print(f'Training took {(time.time() - start_time) / 60:.2f} minutes, train_loss: {train_loss:.4}')
        start_time = time.time()

        # validation
        model.eval()
        valid_losses, preds, targets = [], [], []
        for idx, batch in enumerate(valid_loader):
            inputs, target, _, _ = batch
            # last batch may be too small
            if len(target) < args.batch_size:
                continue
            loss, pred = predict(inputs, target, model, loss_criterion, device, eval_=True)
            valid_losses.append(loss.detach().cpu().numpy())
            preds.append(pred)
            targets.append(target)
       
        valid_loss = np.mean(np.array(valid_losses))
        all_valid_losses.append(valid_loss)
        targets = np.stack(targets)
        print(f'Validation took {(time.time() - start_time) / 60:.2f} minutes, valid_loss: {valid_loss:.4f}')
       
        # early stopping
        if valid_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_preds = preds
            patience_count = 0
        else:
            patience_count += 1

        if patience_count == args.patience:
            print(f'no improvement for {args.patience} epochs -> early stopping')
            break
        
        # save checkpoints
        save_model_path = os.path.join(args.target_dir, 'best_model.pt')
        torch.save(best_model.state_dict(), save_model_path)
        print(f'saved best model to {save_model_path}')
        
        # save training and validation history
        with open(os.path.join(args.target_dir, 'train_losses.txt'), 'w') as f:
            for tl in all_train_losses:
                f.write(f'{tl:.4f}\n')
        with open(os.path.join(args.target_dir, 'valid_losses.txt'), 'w') as f:
            for vl in all_valid_losses:
                f.write(f'{vl:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/work/shared_data/2021-ai4food/raw_data')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--input-dir', type=str, default='ref_fusion_competition_south_africa_train_source_planet_5day')
    parser.add_argument('--label-dir', type=str, default='ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint-epoch', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=4)
    parser.add_argument('--sequence-length', type=int, default=74)
    parser.add_argument('--min-area-to-ignore', type=int, default=1000)
    args = parser.parse_args()

    print('\nbegin args key / value')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('end args keys / value\n')

    main(args)

