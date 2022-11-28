import os
import argparse
import numpy as np
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import wandb

from torch.utils.data.dataloader import DataLoader
from dataset import pytorch_dataset, augmentations



@torch.no_grad()
def testing(model, dataloader, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in tqdm(dataloader):
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        outputs = model(x)
        loss = criterion(outputs, y)

        running_loss.append(loss.cpu().numpy())
        outputs = torch.sigmoid(outputs)
        y_true.append(y.squeeze(1).cpu().int())
        y_pred.append(outputs.squeeze(1).cpu())
    wandb.log({'Loss': np.mean(running_loss)})

    return np.mean(running_loss), torch.cat(y_true, 0), torch.cat(y_pred, 0)


def log_metrics(y_true, y_pred):

    test_acc = tmf.accuracy(y_pred, y_true)
    test_f1 = tmf.f1(y_pred, y_true)
    test_prec = tmf.precision(y_pred, y_true)
    test_rec = tmf.recall(y_pred, y_true)
    test_auc = tmf.auroc(y_pred, y_true)

    wandb.log({
        'Accuracy': test_acc,
        'F1': test_f1,
        'Precision': test_prec,
        'Recall': test_rec,
        'ROC-AUC score': test_auc})


def log_conf_matrix(y_true, y_pred):
    conf_matrix = tmf.confusion_matrix(y_pred, y_true, num_classes=2)
    conf_matrix = pd.DataFrame(data=conf_matrix, columns=['A', 'B'])
    cf_matrix = wandb.Table(dataframe=conf_matrix)
    wandb.log({'conf_mat': cf_matrix})


# main def:
def main():

    # initialize parser
    parser = test_parser()
    args = parser.parse_args()


    # initialize w&b
    
    wandb.init(project=args.project_name, name=args.name,
               config=vars(args), group=args.group)

    # initialize model:
    # TO DO: CREATE A CLASS OF MODEL model = ...

    # load weights:
    model.load_state_dict(torch.load(args.weights_dir, map_location='cpu'))

    model = model.eval().to(args.device)

    # defining transforms:
    transforms = augmentations.get_validation_augmentations()

    # define test dataset:
    test_dataset = pytorch_dataset.dataset2(
        args.dataset_dir, args.test_dir, transforms)

    # define data loaders:
    test_dataloader = DataLoader(test_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

    # set the criterion:
    criterion = nn.BCEWithLogitsLoss()

    # testing
    test_loss, y_true, y_pred = testing(
        model=model, dataloader=test_dataloader, criterion=criterion)

    # calculating and logging results:
    log_metrics(y_true=y_true, y_pred=y_pred)
    log_conf_matrix(y_true=y_true, y_pred=y_pred)

    print(f'Finished Testing with test loss = {test_loss}')


if __name__ == '__main__':
    main()