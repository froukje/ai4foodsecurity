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
from focal_loss import FocalLoss

sys.path.append('../notebooks/starter_files/')

path_to_pseltae = "models"
sys.path.append(path_to_pseltae)
from models.stclassifier import PseLTae, PseTae
from models.stclassifier_combined import PseLTaeCombinedPlanetS1S2, PseLTaeCombinedPlanetS1

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold, StratifiedKFold
from datasets import EarthObservationDataset, PlanetDataset, Sentinel2Dataset, Sentinel1Dataset, CombinedDataset

import numpy as np
import geopandas as gpd
import pandas as pd
import copy
import time

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def main(args):
    
    # setting seeds for reproducability and method comparison
    #np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    # construct the dataset
    if len(args.input_data)==1:
        test_dataset = PlanetDataset(args) 
    else:
        test_dataset = CombinedDataset(args) 

    if args.nr_classes == 5:
        label_ids = [1, 2, 3, 4, 5]
        label_names = ['Wheat', 'Barley', 'Canola', 'Lucerne/Medics', 'Small grain grazing']
    if args.nr_classes == 9:
        label_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        label_names =['Wheat', 'Rye', 'Barley', 'Oats', 'Corn', 'Oil', 'Seeds', 'Root', 'Crops', 'Meadows', 'Forage Crops']
    print(f'label_ids: {label_ids}')
    print(f'label_names: {label_names}\n')

    # set device to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice {device}')

    if args.nni:
        k_best_metrics = [] # gather the k best metrics and then report the mean

    if args.split=='train':

        print('Labels in train and valid / (test)')
        if len(args.input_data)==1:
            (unique, counts) = np.unique(test_dataset[:][1], return_counts=True)
        else:
            (unique, counts) = np.unique(test_dataset[:][0][1], return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)
        
        no_of_classes = unique.shape[0]

        if args.sample_weights == 'uniform':
            weights_for_samples = np.repeat(1, len(frequencies))
        elif args.sample_weights == 'inverse':
            weights_for_samples = 1/frequencies[:,1]
        elif args.sample_weights == 'inverse-sqrt':
            weights_for_samples = 1/np.sqrt(frequencies[:,1])

        weights_for_samples = weights_for_samples/np.sum(weights_for_samples)*no_of_classes 
        print('Use sample weights', weights_for_samples)
        weights_for_samples = torch.Tensor(weights_for_samples).to(device) 
        #criterion = CrossEntropyLoss(weight=weights_for_samples, reduction="sum") #reduction="mean") 
        if args.alpha:
            alpha = weights_for_samples
        else:
            alpha = None
        criterion = FocalLoss(gamma=args.gamma, alpha=alpha) # gamma can be set as a hyperparamter

        if len(args.input_data)==1:
            unique_field_ids = np.unique(test_dataset.fid)
            all_field_ids = test_dataset.fid
        else:
            unique_field_ids = np.unique(test_dataset.datasets[0].fid)
            all_field_ids = test_dataset.datasets[0].fid
        print('Identified unique field IDs: ', len(unique_field_ids))
        
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=7) #StratifiedKFold(n_splits=args.k_folds, shuffle=True) 

        for fold, (train_field_ids, val_field_ids) in enumerate(kfold.split(unique_field_ids)):

            print('----------------------------------------------')
            print(f'STARTING FOLD {fold}')
            print('----------------------------------------------')
            
            # training
            best_loss = np.inf
            best_epoch = 0
            patience_count = 0
            all_train_losses = []
            all_valid_losses = []
            log_scores= []

            train_ids = []
            val_ids   = []

            for ufi in unique_field_ids[train_field_ids]:
                train_ids.extend(np.where(all_field_ids == ufi)[0])

            for ufi in unique_field_ids[val_field_ids]:
                val_ids.extend(np.where(all_field_ids == ufi)[0])

            # shuffle the sequences in place
            np.random.shuffle(train_ids)

            print('Train samples: ', np.array(train_ids).shape)
            print('Valid samples: ', np.array(val_ids).shape)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            train_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                      timeout=0, drop_last=True, sampler=train_subsampler)
            valid_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                      timeout=0, drop_last=True, sampler=val_subsampler)

            print('Size of train loader: ', len(train_loader), 'and val loader: ', len(valid_loader))
            if len(args.input_data)==1:
                (unique, counts) = np.unique(test_dataset[train_ids][1], return_counts=True)
            else:
                (unique, counts) = np.unique(test_dataset[train_ids][0][1], return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print('Labels in train: ',frequencies)

            if len(args.input_data)==1:
                (unique, counts) = np.unique(test_dataset[val_ids][1], return_counts=True)
            else:
                (unique, counts) = np.unique(test_dataset[val_ids][0][1], return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print('Labels in validation: ',frequencies)

            # instantiate the model
            if len(args.input_data)==1: model = PseLTae(**model_config) 
            elif len(args.input_data)==2: model = PseLTaeCombinedPlanetS1(**model_config)
            else: model = PseLTaeCombinedPlanetS1S2(**model_config)
            if torch.cuda.is_available():
                model = model.cuda()   
            # Initialize model optimizer and loss criterion:
            optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #, eps=10e-4)

            for epoch in range(args.max_epochs):
                # train
                model.train()
                start_time = time.time()
                print(f'\nEpoch: {epoch}')
                classes = len(label_ids)
                train_loss = train_epoch(model, optimizer, train_loader, classes, criterion, args, device=device)
                train_loss = train_loss.cpu().detach().numpy()[0]
                all_train_losses.append(train_loss)

                print(f'Training took {(time.time() - start_time) / 60:.2f} minutes, \
                        train_loss: {train_loss:.4}')
                start_time = time.time()

                # validation
                valid_loss, y_true, y_pred, *_ = validation_epoch(model, 
                                                                                valid_loader, 
                                                                                classes, 
                                                                                criterion,
                                                                                args, 
                                                                                device=device)
                valid_loss = valid_loss.cpu().detach().numpy()[0]
                assert not np.isnan(valid_loss)
                all_valid_losses.append(valid_loss)

                # calculate metrics
                scores = metrics(y_true.cpu(), y_pred.cpu())
                #scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
                scores["epoch"] = epoch
                scores["train_loss"] = train_loss
                scores["valid_loss"] = valid_loss
                log_scores.append(scores)
                print(f'Validation scores:')
                for key, value in scores.items():
                    print(f'{key:20s}: {value}')
                print(f'Validation took {(time.time() - start_time) / 60:.2f} minutes, \
                        valid_loss: {valid_loss:.4f}')
                # nni
                if args.nni:
                    pass # do not report intermediate result here
                    #nni.report_intermediate_result(valid_metric)

                # early stopping
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                    best_optimizer = copy.deepcopy(optimizer)
                    best_preds = y_pred
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count == args.patience:
                    print(f'no improvement for {args.patience} epochs -> early stopping')
                    print(f'best loss: {best_loss:.2f} at epoch: {best_epoch}')
                    break

                # save checkpoints
                if epoch % args.checkpoint_epoch == 0 and epoch != 0:
                    save_model_path = os.path.join(args.target_dir, f'epoch_{epoch}_model.pt')
                    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=epoch, log=log_scores), save_model_path)

            # nni
            if args.nni:
                k_best_metrics.append(best_loss)
                nni.report_intermediate_result(best_loss)

            # save best model
            save_model_path = os.path.join(args.target_dir, f'best_model_fold_{fold}.pt') 
            torch.save(dict(model_state=best_model.state_dict(), optimizer_state=best_optimizer.state_dict(), epoch=best_epoch, log=log_scores), save_model_path)
            print(f'saved best model to {save_model_path}')

            # save training and validation history
            with open(os.path.join(args.target_dir, f'train_losses_fold_{fold}.txt'), 'w') as f:
                for tl in all_train_losses:
                    f.write(f'{tl:.4f}\n')
            with open(os.path.join(args.target_dir, f'valid_losses_fold_{fold}.txt'), 'w') as f:
                for vl in all_valid_losses:
                    f.write(f'{vl:.4f}\n')

                print(f"\nINFO: Saved training and validation history ") 
                print(f"\nINFO: Epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} ") 
            model.apply(reset_weights)
            
            if args.save_preds:
                print(f'\nINFO: saving predictions from the {args.split} set')
                save_predictions(args.target_dir, model, valid_loader, device, label_ids, label_names, args, len(val_subsampler), num_folds=1, fold_id=fold, filename=f'{args.split}_{fold}.json')
            
    else:
        # instatiate the model
        if len(args.input_data)==1: model = PseLTae(**model_config)  #PseTae(**model_config) # 
        elif len(args.input_data)==2: model = PseLTaeCombinedPlanetS1(**model_config)
        else: model = PseLTaeCombinedPlanetS1S2(**model_config)

        if torch.cuda.is_available():
            model = model.cuda()   
                    
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
        # make predictions   
        if args.save_preds:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            print(f'\nINFO: saving predictions from the {args.split} set')
            if args.majority:
                save_predictions(args.target_dir, model, test_loader, device, label_ids, label_names, args, len(test_dataset), num_folds=args.k_folds)
            else:
                save_predictions(args.target_dir, model, test_loader, device, label_ids, label_names, args, len(test_dataset), num_folds=1)

    # save reference
    if args.save_ref:
        if args.split == 'train':
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        print(f'\nINFO: saving reference from the {args.split} set')
        save_reference(test_loader, device, label_ids, label_names, args)

    # report final nni result as the average of the best metrics
    # across the k folds
    if args.nni:
        print('For NNI the k best metrics are', k_best_metrics)
        max_best_metrics = np.max(k_best_metrics)
        print('Will report max of the best metrics to NNI:', max_best_metrics)
        nni.report_final_result(max_best_metrics)


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

def get_pselatae_model_config(args, verbose=False):
    
    # adding PseLTae model configs
    include_extras = args.include_extras  # If 1 the precomputed geometrical features (f) are used in the PSE
    if include_extras: extra_size = 2
    else: extra_size = 0
    mlp2_first_layer = args.mlp1_out*2 + extra_size#128 + extra_size
    
    if args.input_data[0]=='planet':
        if args.nr_classes == 5: # south africa
            lms = 244
        elif args.nr_classes == 9: # germany
            lms = 365
    elif args.input_data[0]=='planet-5':
        if args.nr_classes == 5: # south africa
            lms = 48
        elif args.nr_classes == 9: # germany
            lms = 73
    elif args.input_data[0]=='sentinel-1':
        if args.nr_classes == 5: # south africa
            lms = 41
        elif args.nr_classes == 9: # germany
            lms = 122
    else:    # sentinel-2
        if args.nr_classes == 5: # south africa
            lms = 76
        #elif args.nr_classes == 9: # germany
            #lms = 122
    
    if len(args.input_data)==1:
        model_config = dict(input_dim = args.input_dim[0], 
                            # Number of neurons in the layers of MLP1
                            mlp1 = [args.input_dim[0],args.mlp1_in,args.mlp1_out], 
                            # Pixel-embeddings pooling strategy
                            pooling = 'mean_std',
                            # Number of neurons in the layers of MLP2
                            mlp2 = [mlp2_first_layer,mlp2_first_layer],
                            # Number of attention heads
                            n_head = args.n_head,   
                            # Dimension of the key and query vectors
                            d_k = args.d_k,
                            # Number of neurons in the layers of MLP3
                            mlp3 = [args.n_head*args.factor,args.mlp3_out],
                            # Dropout probability
                            dropout = args.dropout,
                            # Maximum period for the positional encoding
                            T = 1000, 
                            # Maximum sequence length for positional encoding (only necessary if positions == order) 
                            len_max_seq = lms,
                            # Positions to use for the positional encoding (bespoke / order)
                            positions=None, 
                            # Number of neurons in the layers of MLP4
                            mlp4 = [args.mlp3_out, args.mlp4_1, args.mlp4_2, args.nr_classes],
                            # Size of the embeddings (E), if input vectors are of a different size, 
                            # a linear layer is used to project them to a d_model-dimensional space 
                            d_model = args.n_head*args.factor)
    
    elif len(args.input_data)==2:

        model_config = dict(input_dim_planet=args.input_dim[0], 
                            input_dim_s1 = args.input_dim[1], 
                            # Number of neurons in the layers of MLP1
                            mlp1_planet = [args.input_dim[0],args.mlp1_in,args.mlp1_out],   
                            # Number of neurons in the layers of MLP1
                            mlp1_s1 = [args.input_dim[1], args.mlp1_s1_in,args.mlp1_s1_out],    
                            # Pixel-embeddings pooling strategy
                            pooling = 'mean_std',
                            # Number of neurons in the layers of MLP2
                            mlp2 = [mlp2_first_layer,mlp2_first_layer], 
                            # Number of attention heads
                            n_head = args.n_head, 
                            # Dimension of the key and query vectors
                            d_k = args.d_k, 
                            # Number of neurons in the layers of MLP3
                            mlp3_planet = [args.n_head*args.factor, args.mlp3_out],
                            # Number of neurons in the layers of MLP3
                            mlp3_s1 = [args.n_head*args.factor, int(args.scale*args.mlp3_s1_out)], 
                            # Dropout probability
                            dropout = args.dropout, 
                            T = 1000, 
                            # Maximum sequence length for positional encoding (only necessary if positions == order)
                            len_max_seq_planet = lms, 
                            len_max_seq_s1 = 41,
                            # Positions to use for the positional encoding (bespoke / order)
                            positions = None, #dt.date_positions if config['positions'] == 'bespoke' else None,
                            # Number of neurons in the layers of MLP4
                            mlp4 = [args.mlp3_out+int(args.scale*args.mlp3_s1_out), args.mlp4_1, args.mlp4_2, args.nr_classes],
                            # Size of the embeddings (E), if input vectors are of a different size, 
                            # a linear layer is used to project them to a d_model-dimensional space
                            d_model = args.n_head*args.factor)
    else:

        model_config = dict(input_dim_planet = args.input_dim[0], 
                            input_dim_s1 = args.input_dim[1], 
                            input_dim_s2 = args.input_dim[2],
                            # Number of neurons in the layers of MLP1
                            mlp1_planet = [args.input_dim[0],args.mlp1_in,args.mlp1_out], 
                            mlp1_s1 = [args.input_dim[1], args.mlp1_s1_in,args.mlp1_s1_out], 
                            mlp1_s2 = [args.input_dim[2], args.mlp1_s2_in,args.mlp1_s2_out],
                            # Pixel-embeddings pooling strategy
                            pooling = 'mean_std',
                            mlp2 = [mlp2_first_layer,mlp2_first_layer],
                            # Number of attention heads
                            n_head = args.n_head, 
                            # Dimension of the key and query vectors
                            d_k = args.d_k,
                            # Number of neurons in the layers of MLP3
                            mlp3_planet = [args.n_head*args.factor, args.mlp3_out],
                            mlp3_s1 = [args.n_head*args.factor, int(args.scale*args.mlp3_s1_out)], 
                            mlp3_s2 = [args.n_head*args.factor, int(args.scale_s2*args.mlp3_s2_out)],
                            # Dropout probability
                            dropout = args.dropout,
                            # Maximum period for the positional encoding
                            T = 1000, 
                            # Maximum sequence length for positional encoding (only necessary if positions == order) 
                            len_max_seq_planet = lms, 
                            len_max_seq_s1 = 41,
                            len_max_seq_s2 = 76,
                            # Positions to use for the positional encoding (bespoke / order)
                            positions = None, #dt.date_positions if config['positions'] == 'bespoke' else None,
                            # Number of neurons in the layers of MLP4
                            mlp4 = [args.mlp3_out+int(args.scale*args.mlp3_s1_out)+int(args.scale_s2*args.mlp3_s2_out), args.mlp4_1, args.mlp4_2, args.nr_classes],
                            # size of the embeddings (E), if input vectors are of a different size, 
                             # a linear layer is used to project them to a d_model-dimensional space
                            d_model = args.n_head*args.factor)
        
    if include_extras:
        model_config.update(with_extra=True, extra_size=extra_size) # extra_size number of extra features
    else:
        model_config.update(with_extra=False, extra_size=None)   
    
    if verbose: print('Model configs: ', model_config)
    
    return model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-data-dir', type=str, 
                        default='../../../shared_data/2021-ai4food/dev_data/south-africa')
    parser.add_argument('--input-data-type', type=str, default='extracted')
    parser.add_argument('--input-data', nargs="*", type=str, default='planet')
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test']) 
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--majority', type=int, default=1, choices=[0,1])
    parser.add_argument('--nni', action='store_true', default=False)
    parser.add_argument('--save-preds', action='store_true', default=False) 
    parser.add_argument('--save-ref', action='store_true', default=False) 
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--checkpoint-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input-dim', type=int, nargs="*", default=[4])
    parser.add_argument('--sequence-length', type=int, default=74)
    parser.add_argument('--num-workers', type=int, default=0, help='Timeout if > 0 for combined dataset')
    parser.add_argument('--ndvi', type=int, default=0, choices=[0, 1])
    parser.add_argument('--nri', type=int, default=0, choices=[0, 1])
    parser.add_argument('--drop-channels', type=int, default=0, choices=[0, 1]) # if set then ndvi and/or nri also need to be set and input-dim set to 1
    parser.add_argument('--fill-value', type=bool, default=0)
    parser.add_argument('--augmentation', type=int, default=0, choices=[0,1]) # add gaussian noise to samples
    # for pseltae model
    parser.add_argument('--include-extras', type=int, default=0, choices=[0, 1])
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='In Adam optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='In Adam optimizer')
    parser.add_argument('--sample-weights', type=str, choices=['uniform', 'inverse', 'inverse-sqrt'], default='inverse',
                           help='Sample weight strategy')
    parser.add_argument('--n-head', type=int, default=16)
    parser.add_argument('--d-k', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mlp1-in', type=int, default=32)
    parser.add_argument('--mlp1-s1-in', type=int, default=32)
    parser.add_argument('--mlp1-s2-in', type=int, default=32)
    parser.add_argument('--mlp1-out', type=int, default=64)
    parser.add_argument('--mlp1-s1-out', type=int, default=64)
    parser.add_argument('--mlp1-s2-out', type=int, default=64)
    parser.add_argument('--mlp3-out', type=int, default=128)
    parser.add_argument('--mlp3-s1-out', type=int, default=128)
    parser.add_argument('--mlp3-s2-out', type=int, default=128)
    parser.add_argument('--mlp4-1', type=int, default=64)
    parser.add_argument('--mlp4-2', type=int, default=32)
    parser.add_argument('--factor', type=int, default=16)
    parser.add_argument('--scale', type=float, default=0.25)
    parser.add_argument('--scale-s2', type=float, default=0.25, help='Scale for Sentinel 2')
    parser.add_argument('--nr-classes', type=int, choices=[5,9], default=5, help='Expected number of classes (S: 5, G: 9)')
    # pool only working for default value!
    parser.add_argument('--pool', type=str, default='mean_std', choices=['mean_std', 'mean', 'std', 'max', 'min'])
    parser.add_argument('--alpha', action='store_true', default=False)
    parser.add_argument('--gamma', type=int, default=1)
    # sentinel-2 interpolation
    parser.add_argument('--sentinel-2-spline', type=int, default=1, choices=[1,2,3,4,5], help='Spline for Sentinel 2 interpolation')
    parser.add_argument('--cloud-probability-threshold', type=float, default=0.1, help='Cloud probability threshold for Sentinel 2 interpolation')
    parser.add_argument('--savgol-filter', type=int, default=0, choices=[0, 1], help='Use Savitzky Golay filter for Sentinel 1 RVI smoothing')
    args = parser.parse_args()

    if args.nni:
        args = add_nni_params(args)

    print('\nbegin args key / value')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('end args keys / value\n')
    
    model_config = get_pselatae_model_config(args, verbose=True)
    args.model_config = model_config

    main(args)
