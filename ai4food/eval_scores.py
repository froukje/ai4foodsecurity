###
# This script calculates the evaluation score used in the starter notebook for the validation set
# This script does not run in the singularity container!
###


import os
import argparse
import pandas as pd
import numpy as np
import json
import tensorflow as tf

def main(args):
    reference_file = os.path.join(args.ref_path, 'reference_val.json')
    #reference_file = os.path.join(args.ref_path, 'submission_val.json')
    submission_file = os.path.join(args.pred_path, 'validation.json')
    #submission_file = os.path.join(args.pred_path, 'submission.json')

    # Opening JSON file
    with open(reference_file) as json_file:
        reference_json = json.load(json_file)
    with open(submission_file) as json_file:
        submission_json = json.load(json_file)

    submission = pd.DataFrame.from_dict(submission_json)
    reference = pd.DataFrame.from_dict(reference_json)
    print(reference.head())
    print(submission.head())
    
    score = calculate_score(reference, submission)
    print('SCORE:', score)

def calculate_score(reference, submission):
    """
    Scores the crop type classficiation using the modified categorical cross entropy.

    Parameters
    ----------
    reference
        pandas dataframe including the target crop IDs for each field.
    submission
        pandas dataframe including the predicted crop IDs for each field.

    Returns
    -------
    Score value as a float
    """

    submission.sort_values('fid', inplace=True)
    reference.sort_values('fid', inplace=True)
    
    if len(submission) != len(reference):
        print('Your submission does not satisfy the required length')
    else:
        if (submission['fid'].values == reference['fid'].values).all():
            df = pd.merge(reference, submission, on=['fid'], suffixes=('_reference', '_submission'), how='left')
            depth = np.max(df[['crop_id_reference', 'crop_id_submission']].max())
            df.crop_id_reference.loc[df.crop_id_reference > 0] -= 1
            df.crop_id_submission.loc[df.crop_id_submission > 0] -= 1
            y_true = tf.one_hot(df['crop_id_reference'], depth=depth)
            y_pred = tf.one_hot(df['crop_id_submission'], depth=depth)
            cross_entropy_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            return cross_entropy_func(y_true, y_pred).numpy()
        else:
            print('Your submission contains wrong FIDs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--ref-path', type=str, default='.')
    parser.add_argument('--pred-path', type=str, default='.')
    args = parser.parse_args()

    print('\nbegin args key: value')
    for key, value in vars(args).items():
        print(f'{key:20s}:{value}')
    print('\nend args key: value')
    main(args)


