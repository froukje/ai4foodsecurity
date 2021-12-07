import os
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def metrics(y_true, y_pred):
    """
    THIS FUNCTION DETERMINES THE EVALUATION METRICS OF THE MODEL

    :param y_true: ground-truth labels
    :param y_pred: predicted labels

    :return: dictionary of Accuracy, Kappa, F1, Recall, and Precision
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro", zero_division=0)
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalise confusion matrix to get accurac for each class

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        accuracy_per_class=cm.diagonal()
    )


def bin_cross_entr_each_crop(logprobs, y_true, classes, device, args):
    '''
    calculates binary cross entropy for each class
    and sums the result up
    :param y_pred: model predictions
    :pram y_target: target
    :param classes: nr of classes
    
    :return: sum of binary cross entropy for each class 
    '''
    bin_ce = 0
    sm = nn.Softmax(dim=1)
    y_prob = sm(logprobs)
    # convert to one-hot representation
    y_prob_ids = torch.argmax(y_prob, dim=1)
    y_pred_onehot = nn.functional.one_hot(y_prob_ids, num_classes=5).float()
    y_true_onehot = nn.functional.one_hot(y_true, num_classes=5).float()
    #y_prob_clipped = torch.clip(y_pred_onehot, 1e-7, 1-1e-7)
    y_prob_clipped = torch.clip(y_prob, 1e-7, 1-1e-7)
    bin_ce = torch.tensor(bin_ce)
    #loss_batch = -torch.log(y_prob_clipped[range(len(y_pred_onehot)), y_true])
    loss_batch = -torch.log(y_prob_clipped[range(len(y_prob)), y_true])
    bin_ce = torch.sum(loss_batch)

    return bin_ce



def train_epoch(model, optimizer, dataloader, classes, criterion, args, device='cpu'):
    """
    THIS FUNCTION ITERATES A SINGLE EPOCH FOR TRAINING

    :param model: torch model for training
    :param optimizer: torch training optimizer
    :param criterion: torch objective for loss calculation
    :param dataloader: training data loader
    :param device: where to run the epoch

    :return: loss
    """
    model.train()
    losses = list()
    eval_metrics = list()
    with tqdm(enumerate(dataloader), total=len(dataloader),position=0, leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            
            if args.use_pselatae and args.include_extras:
                (x, mask, _, extra_features), y_true = batch
                logprobs = model(((x.to(device), mask.to(device)), extra_features.to(device)))
            # for combined model - current implementation w/o extra features
            elif len(args.input_data)>1:
                sample_planet, sample_s1 = batch
                for i in range(len(sample_planet)):
                    sample_planet[i] = sample_planet[i].to(device)
                    sample_s1[i] = sample_s1[i].to(device)
                
                (x_p, mask_p, _), y_true = sample_planet
                (x_s1, mask_s1, _), _ = sample_s1
                logprobs = model(((x_p, mask_p), (x_s1, mask_s1)))
            
            # for spatiotemporal models
            else:
                (x, mask, _), y_true = batch
                if args.use_pselatae: logprobs = model((x.to(device), mask.to(device)))
                else: logprobs = model(x.to(device))
                
            y_true = y_true.to(device)

            eval_metric = bin_cross_entr_each_crop(logprobs, y_true, classes, device, args)
            eval_metrics.append(eval_metric)
            loss = criterion(logprobs, y_true)
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses), torch.stack(eval_metrics)


def validation_epoch(model, dataloader, classes, criterion, args, device='cpu'):
    """
    THIS FUNCTION ITERATES A SINGLE EPOCH FOR VALIDATION

    :param model: torch model for validation
    :param criterion: torch objective for loss calculation
    :param dataloader: validation data loader
    :param device: where to run the epoch

    :return: loss, y_true, y_pred, y_score, field_id
    """
    model.eval()
    with torch.no_grad():
        losses = list()
        eval_metrics = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True) as iterator:
            for idx, batch in iterator:
               
                if args.use_pselatae and args.include_extras:
                    (x, mask, field_id, extra_features), y_true = batch
                    logprobs = model(((x.to(device), mask.to(device)), extra_features.to(device)))
                # for combined model - current implementation wo extra features
                elif len(args.input_data)>1:
                    sample_planet, sample_s1 = batch
                    for i in range(len(sample_planet)):
                        sample_planet[i] = sample_planet[i].to(device)
                        sample_s1[i] = sample_s1[i].to(device)

                    (x_p, mask_p, field_id), y_true = sample_planet
                    (x_s1, mask_s1, _), y_true = sample_s1
                    logprobs = model(((x_p, mask_p), (x_s1, mask_s1)))
                # for spatiotemporal models
                else:
                    (x, mask, field_id), y_true = batch
                    if args.use_pselatae: logprobs = model((x.to(device), mask.to(device)))
                    else: logprobs = model(x.to(device))
                        
                y_true = y_true.to(device)
                eval_metric = bin_cross_entr_each_crop(logprobs, y_true, classes, device, args)
                eval_metrics.append(eval_metric)
                loss = criterion(logprobs, y_true.to(device))
                iterator.set_description(f"valid loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobs.argmax(-1))
                y_score_list.append(logprobs.exp())
                field_ids_list.append(field_id)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(field_ids_list), torch.stack(eval_metrics)


def save_reference(data_loader, device, label_ids, label_names, args):
    # list of dictionaries with predictions:
    output_list=[]

    with torch.no_grad():
        with tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True) as iterator:
            for idx, batch in iterator:
                if len(args.input_data)>1: batch=batch[0]
                if args.include_extras: (_, _, fid,_), y_true = batch
                else: (_, _, fid), y_true = batch
                for i in range(y_true.size()[0]):
                    fid_i = fid[i].view(1,-1)[0]
                    output_list.append({'fid': fid_i.cpu().detach().numpy()[0],
                                    'crop_id': label_ids[y_true[i]],
                                    'crop_name': label_names[y_true[i]]})

    #  save reference into output json:
    if args.split == 'train':
        output_name = os.path.join(args.target_dir, 'reference_val.json')
        print(f'Reference for validation was saved to location: {(output_name)}')
        output_frame = pd.DataFrame.from_dict(output_list)
        output_frame.to_json(output_name)
    else:
        print(f'No reference was saved')


def save_predictions(save_model_path, model, data_loader, device, label_ids, label_names, args):
    if os.path.exists(save_model_path):
        checkpoint = torch.load(save_model_path)
        START_EPOCH = checkpoint["epoch"]
        log = checkpoint["log"]
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"INFO: Resuming from {save_model_path}, epoch {START_EPOCH}")

        # list of dictionaries with predictions:
        output_list=[]
        softmax=torch.nn.Softmax(dim=1)

        with torch.no_grad():
            with tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True) as iterator:
                for idx, batch in iterator:
                    
                    if args.use_pselatae and args.include_extras:
                        (x, mask, fid, extra_features), _ = batch
                        logits = model(((x.to(device), mask.to(device)), extra_features.to(device)))
                    # for combined model - current implementation wo extra features
                    elif len(args.input_data)>1:
                        sample_planet, sample_s1 = batch
                        for i in range(len(sample_planet)):
                            sample_planet[i] = sample_planet[i].to(device)
                            sample_s1[i] = sample_s1[i].to(device)
                        (x_p, mask_p, fid), _ = sample_planet
                        (x_s1, mask_s1, _), _ = sample_s1
                        logits = model(((x_p, mask_p), (x_s1, mask_s1)))
                    # for spatiotemporal model
                    else:
                        (x, mask, fid), _ = batch
                        if args.use_pselatae: logits = model((x.to(device), mask.to(device)))
                        else: logits = model(x.to(device))
                    for i in range(logits.size()[0]):
                        logits_i = logits[i].view(1,-1)
                        predicted_probabilities = softmax(logits_i).cpu().detach().numpy()[0]
                        fid_i = fid[i].view(1,-1)[0]
                        predicted_class = np.argmax(predicted_probabilities)
                        output_list.append({'fid': fid_i.cpu().detach().numpy()[0],
                                    'crop_id': label_ids[predicted_class],
                                    'crop_name': label_names[predicted_class],
                                    'crop_probs': np.array(predicted_probabilities)})

        #  save predictions into output json:
        if args.split == 'train':
            output_name = os.path.join(args.target_dir, 'validation.json')
            print(f'Validation was saved to location: {(output_name)}')
        else:
            output_name = os.path.join(args.target_dir, 'submission.json')
            print(f'Submission was saved to location: {(output_name)}')
        output_frame = pd.DataFrame.from_dict(output_list)
        # temporary fix for class mismatch
        
        # swap 1s and 4s
        crop_ids = output_frame['crop_id']
        crop_ids = np.array(crop_ids)
        crop_ids[crop_ids==1] = 100
        crop_ids[crop_ids==4] = 1
        crop_ids[crop_ids==100] = 4
        output_frame['crop_id'] = crop_ids.astype(np.uint8)
        # swap Wheat and Lucerne/Medics
        output_frame['crop_name']=output_frame['crop_name'].str.replace('Wheat', 'blabla')
        output_frame['crop_name']=output_frame['crop_name'].str.replace('Lucerne/Medics', 'Wheat')
        output_frame['crop_name']=output_frame['crop_name'].str.replace('blabla', 'Lucerne/Medics')

        print(output_frame.tail())
        output_frame.to_json(output_name)

    else:
        print('INFO: no best model found ...')
        
def save_predictions_majority(target_dir, model, data_loader, device, label_ids, label_names, args, num_samples, num_folds=5):
    
    not_found=0
    softmax=torch.nn.Softmax(dim=1)
    probs_array=np.zeros((num_samples, len(label_ids)))
    batch_s = args.batch_size
    
    for fold in range(num_folds):
        
        if os.path.exists(target_dir):
            save_model_path = os.path.join(args.target_dir, f'best_model_fold_{fold}.pt')
            checkpoint = torch.load(save_model_path)
            START_EPOCH = checkpoint["epoch"]
            log = checkpoint["log"]
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            print(f"INFO: Resuming from {save_model_path}, epoch {START_EPOCH}")

            # list of dictionaries with predictions:
            output_list=[]

            with torch.no_grad():
                with tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True) as iterator:
                    for idx, batch in iterator:
                        
                        if args.use_pselatae and args.include_extras:
                            (x, mask, fid, extra_features), _ = batch
                            logits = model(((x.to(device), mask.to(device)), extra_features.to(device)))
                        # for combined model - current implementation wo extra features
                        elif args.input_data>1:
                            sample_planet, sample_s1 = batch
                            for i in range(len(sample_planet)):
                                sample_planet[i] = sample_planet[i].to(device)
                                sample_s1[i] = sample_s1[i].to(device)
                            (x_p, mask_p, fid), _ = sample_planet
                            (x_s1, mask_s1, _), _ = sample_s1
                            logits = model(((x_p, mask_p), (x_s1, mask_s1)))
                        # for spatiotemporal model
                        else:
                            (x, mask, fid), _ = batch
                            if args.use_pselatae: logits = model((x.to(device), mask.to(device)))
                            else: logits = model(x.to(device))
                        
                        batch_s = x.size(0)
                        
                        for i in range(logits.size()[0]):
                            logits_i = logits[i].view(1,-1)
                            predicted_probabilities = softmax(logits_i).cpu().detach().numpy()[0]
                            probs_array[idx*batch_s+i] += predicted_probabilities # idx*batch_s:(idx*batch_s)+batch_s
                            
                            if fold==(num_folds-1):
                                probs_array_ids=probs_array[idx*batch_s+i]/(num_folds-not_found)
                                fid_i = fid[i].view(1,-1)[0]
                                predicted_class = np.argmax(probs_array_ids)
                                output_list.append({'fid': fid_i.cpu().detach().numpy()[0],
                                            'crop_id': label_ids[predicted_class],
                                            'crop_name': label_names[predicted_class],
                                            'crop_probs': np.array(probs_array_ids)})

        else:
            print(f"INFO: no best model found for fold {fold}")
            not_found+=1
            
    #  save predictions into output json:
    output_name = os.path.join(args.target_dir, 'submission.json')
    print(f'Submission was saved to location: {(output_name)}')
    output_frame = pd.DataFrame.from_dict(output_list)
    # ____________________temporary fix for class mismatch________________________
    # swap 1s and 4s
    crop_ids = output_frame['crop_id']
    crop_ids = np.array(crop_ids)
    crop_ids[crop_ids==1] = 100
    crop_ids[crop_ids==4] = 1
    crop_ids[crop_ids==100] = 4
    output_frame['crop_id'] = crop_ids.astype(np.uint8)
    # swap Wheat and Lucerne/Medics
    output_frame['crop_name']=output_frame['crop_name'].str.replace('Wheat', 'blabla')
    output_frame['crop_name']=output_frame['crop_name'].str.replace('Lucerne/Medics', 'Wheat')
    output_frame['crop_name']=output_frame['crop_name'].str.replace('blabla', 'Lucerne/Medics')

    output_frame.to_json(output_name)
