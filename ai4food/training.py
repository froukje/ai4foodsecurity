#!/usr/bin/env python

try:
    import nni
except ImportError:
    pass

import argparse
import os
import sys
import h5py
from evaluation_utils import metrics, train_epoch, validation_epoch, save_predictions, save_reference

sys.path.append('../notebooks/starter_files/')
from utils.data_transform import PlanetTransform
from baseline_models import SpatiotemporalModel

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from datasets import EarthObservationDataset, PlanetDataset, Sentinel2Dataset

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

    label_ids = [1, 2, 3, 4, 5]
    label_names = ['Wheat', 'Barley', 'Canola', 'Lucerne/Medics', 'Small grain grazing']

    print(f'label_ids: {label_ids}')
    print(f'label_names: {label_names}\n')

    # Initialize data loaders
    if args.split == 'train':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8,drop_last=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    # set device to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instatiate the model
    model = SpatiotemporalModel(input_dim=args.input_dim, num_classes=len(label_ids), sequencelength=args.sequence_length, 
                                spatial_backbone=args.spatial_backbone, temporal_backbone=args.temporal_backbone, device=device)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f'\nDevice {device}')

    # Initialize model optimizer and loss criterion:
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    #criterion = CrossEntropyLoss(reduction="mean")
    criterion = nn.NLLLoss(reduction='sum')

    # training
    best_loss = np.inf
    best_metric = np.inf
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
            classes = len(label_ids)
            train_loss, train_metric = train_epoch(model, optimizer, train_loader, classes, criterion, args, device=device)
            train_loss = train_loss.cpu().detach().numpy()[0]
            train_metric = train_metric.cpu().detach().numpy()[0]
            all_train_losses.append(train_loss)

            print(f'Training took {(time.time() - start_time) / 60:.2f} minutes, \
                    train_loss: {train_loss:.4}, eval_metric: {train_metric:.4}')
            start_time = time.time()

            # validation
            valid_loss, y_true, y_pred, *_, valid_metric = validation_epoch(model, 
                                                                            valid_loader, 
                                                                            classes, 
                                                                            criterion, 
                                                                            args, 
                                                                            device=device)
            valid_loss = valid_loss.cpu().detach().numpy()[0]
            valid_metric = valid_metric.cpu().detach().numpy()[0]
            all_valid_losses.append(valid_loss)

            # calculate metrics
            scores = metrics(y_true.cpu(), y_pred.cpu())
            scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
            scores["epoch"] = epoch
            scores["train_loss"] = train_loss
            scores["valid_loss"] = valid_loss
            log_scores.append(scores)
            print(f'Validation took {(time.time() - start_time) / 60:.2f} minutes, \
                    valid_loss: {valid_loss:.4f}, eval_metric {valid_metric:.4}')
            # nni
            if args.nni:
                #nni.report_intermediate_result(valid_metric)
                nni.report_intermediate_result(valid_loss)
        
            # early stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_metric = valid_metric
                best_model = copy.deepcopy(model)
                best_preds = y_pred
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == args.patience:
                print(f'no improvement for {args.patience} epochs -> early stopping')
                break
        
            # save checkpoints
            if epoch % args.checkpoint_epoch == 0:
                save_model_path = os.path.join(args.target_dir, f'epoch_{epoch}_model.pt')
                torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    epoch=epoch, log=log_scores), save_model_path)

        # nni
        if args.nni:
            #nni.report_final_result(best_metric)
            nni.report_final_result(best_loss)

        # save best model
        save_model_path = os.path.join(args.target_dir, 'best_model.pt')
        torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), 
                    epoch=epoch, log=log_scores), save_model_path)
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
            test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8)
            save_model_path = os.path.join(args.target_dir, 'best_model.pt')
        
        print(f'\nINFO: saving predictions from the {args.save_preds} set')
        save_predictions(save_model_path, model, test_loader, device, label_ids, label_names, args)

    # save reference
    if args.save_ref:
        if args.split == 'train':
            test_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8)
        
        print(f'\nINFO: saving reference from the {args.save_preds} set')
        save_reference(test_loader, device, label_ids, label_names, args)


def add_nni_params(args):
    args_nni = nni.get_next_parameter()
    assert all([key in args for key in args_nni.keys()]), 'need only valid parameters'
    args_dict = vars(args)
    # cast params that should be int to int if needed (nni may offer them as float)
    args_nni_casted = {key:(int(value) if type(args_dict[key]) is int else value) 
                        for key, value in args_nni.items()}
    args_dict.update(args_nni_casted)
    # adjust paths of model and prediction outputs so they get saved together with the other outputs
    nni_output_dir = os.path.expandvars('$NNI_OUTPUT_DIR')
    nni_path = os.path.join(nni_output_dir, os.path.basename(args_dict['target_dir']))
    args_dict['target_dir'] = nni_path
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-data-dir', type=str, 
                        default='/mnt/lustre02/work/ka1176/shared_data/2021-ai4food/dev_data/south-africa/planet_5day/default')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test']) 
    parser.add_argument('--nni', action='store_true', default=False)
    parser.add_argument('--save-preds', action='store_true', default=False) 
    parser.add_argument('--save-ref', action='store_true', default=False) 
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input-dim', type=int, default=4)
    parser.add_argument('--sequence-length', type=int, default=74)
    #parser.add_argument('--ndvi', action='store_true', default=False)
    parser.add_argument('--ndvi', type=int, default=0, choices=[0, 1])
    parser.add_argument('--temporal-backbone', type=str, default='lstm', 
                        choices=["inceptiontime", "lstm", "msresnet", "starrnn", "tempcnn", "transformermodel"])
    parser.add_argument('--spatial-backbone', type=str, default='mobilenet_v3_small', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101','resnext50_32x4d','resnext50_32x4d',
                                'wide_resnet50_2', 'mobilenet_v3_large',
                                "mobilenet_v3_small", 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                                'vgg16_bn', 'vgg19_bn', 'vgg19', "alexnet", 'squeezenet1_0'])
    parser.add_argument('--fill-value', type=bool, default=0)
    args = parser.parse_args()

    if args.nni:
        args = add_nni_params(args)

    print('\nbegin args key / value')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('end args keys / value\n')

    main(args)

