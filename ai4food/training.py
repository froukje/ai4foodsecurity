#!/usr/bin/env python

try:
    import nni
except ImportError:
    pass

import argparse
import os
import sys
import evaluation_utils

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
from tqdm import tqdm


def main(args):
   
    # read data
    train_data_dir = os.path.join(args.data_dir, args.input_dir)
    train_labels_dir = os.path.join(args.data_dir, args.label_dir)
    train_labels = gpd.read_file(train_labels_dir)

    # TODO read test data
    #test_data_dir = os.path.join(args.data_dir, args.test_dir)
    #test_labels_dir = os.path.join(args.data_dir, args.test_label_dir)
    #test_labels = gpd.read_file(test_labels_dir)
    
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
    # TODO read test data
    #planet_reader_test = PlanetReader(input_dir=test_data_dir,
    #                                label_dir=test_labels_dir,
    #                                label_ids=label_ids,
    #                                transform=planet_transformer.transform,
    #                                min_area_to_ignore=args.min_area_to_ignore)

    #Initialize data loaders
    data_loader=DataLoader(train_val_reader=planet_reader, validation_split=0.25)
    train_loader=data_loader.get_train_loader(batch_size=args.batch_size, num_workers=1)
    valid_loader=data_loader.get_validation_loader(batch_size=args.batch_size, num_workers=1)

    print(f'train loader: {len(train_loader)*args.batch_size} samples in {len(train_loader)} batches')
    print(f'valid loader: {len(valid_loader)*args.batch_size} samples in {len(train_loader)} batches')
   
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
    log_scores= []

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
        scores = evaluation_utils.metrics(y_true.cpu(), y_pred.cpu())
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
        if args.save_preds=='valid' or args.save_preds=='test':

            print(f'\nINFO: saving predictions from the {args.save_preds} set')
            if os.path.exists(save_model_path):
                checkpoint = torch.load(save_model_path)
                START_EPOCH = checkpoint["epoch"]
                log = checkpoint["log"]
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                print(f"INFO: Resuming from {save_model_path}, epoch {START_EPOCH}")

                # list of dictionaries with predictions:
                output_list=[]
                softmax=torch.nn.Softmax()
        
                if args.save_preds == 'valid':
                    valid_loader=data_loader.get_validation_loader(batch_size=1, num_workers=1)
                else:
                    # TODO save predictions from test set
                    pass

                with torch.no_grad():
                    with tqdm(enumerate(valid_loader), total=len(valid_loader), position=0, leave=True) as iterator:
                        for idx, batch in iterator:

                            X, y_true, _, fid = batch
                            logits = model(X.to(device))
                            predicted_probabilities = softmax(logits).cpu().detach().numpy()[0]
                            predicted_class = np.argmax(predicted_probabilities)
    
                            output_list.append({'fid': fid.cpu().detach().numpy(),
                                'crop_id': label_ids[predicted_class],
                                'crop_name': label_names[predicted_class],
                                'crop_probs': predicted_probabilities})
    
                #  save predictions into output json:
                if args.save_preds == 'valid':
                    output_name = os.path.join(args.target_dir, '34S-20E-259N-2017-validation.json')
                else:
                    output_name = os.path.join(args.target_dir, '34S-20E-259N-2017-submission.json')
                output_frame = pd.DataFrame.from_dict(output_list)
                output_frame.to_json(output_name)
                print(f'Validation / Submission was saved to location: {(output_name)}')

            else:
                print('INFO: no best model found ...')

#def make_predictions(model, dataloader, loss_criterion, device, args):
#    _, y_true, y_pred, *_ = tveu.validation_epoch(model, loss_criterion, dataloader, device=device)
#    print(y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/work/shared_data/2021-ai4food/raw_data')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--input-dir', type=str, default='ref_fusion_competition_south_africa_train_source_planet_5day')
    parser.add_argument('--test-dir', type=str, default='ref_fusion_competition_south_africa_test_source_planet_5day')
    parser.add_argument('--label-dir', type=str, default='ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson')
    #parser.add_argument('--test-label-dir', type=str, default='ref_fusion_competition_south_africa_test_labels/ref_fusion_competition_south_africa_test_labels_34S_20E_259N/labels.geojson')
    # parameter to save predictions from validation or tet set
    parser.add_argument('--save_preds', type=str, default='', choices=['valid', 'test']) 
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


