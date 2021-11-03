import os
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm

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
    )

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
        softmax=torch.nn.Softmax()

        with torch.no_grad():
            with tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True) as iterator:
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
            print(f'Validation was saved to location: {(output_name)}')
        else:
            output_name = os.path.join(args.target_dir, '34S-20E-259N-2017-submission.json')
            output_frame = pd.DataFrame.from_dict(output_list)
            output_frame.to_json(output_name)
            print(f'Submission was saved to location: {(output_name)}')

    else:
        print('INFO: no best model found ...')
