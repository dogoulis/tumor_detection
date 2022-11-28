import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from network import network
import torch
import torch.nn as nn
import wandb

from torch.utils.data.dataloader import DataLoader
from dataset import dataset2
import augmentations 
import timm

# define training logic
def train_epoch(model, train_dataloader, device, optimizer, criterion, scheduler=None, epoch=0, val_results={}, scheme=False):
    # to train only the classification layer:
    model.train()

    running_loss = []
    pbar = tqdm(train_dataloader, desc='epoch {}'.format(epoch), unit='iter')

    # train with clasic scheme
    if scheme:
        for batch, (x, y) in enumerate(pbar):

            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())

            # log mean loss for the last 10 batches:
            if (batch+1) % 10 == 0:
                wandb.log({'train-step-loss': np.mean(running_loss[-10:])})
                pbar.set_postfix(loss='{:.3f} ({:.3f})'.format(running_loss[-1], np.mean(running_loss)), **val_results)

        # change the position of the scheduler:
        scheduler.step()

        train_loss = np.mean(running_loss)

        wandb.log({'train-epoch-loss': train_loss})

    # train with teacher-student scheme
    else:
        for batch, (x, y) in enumerate(pbar):

            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)


            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())

            # log mean loss for the last 10 batches:
            if (batch+1) % 10 == 0:
                wandb.log({'train-step-loss': np.mean(running_loss[-10:])})
                pbar.set_postfix(loss='{:.3f} ({:.3f})'.format(running_loss[-1], np.mean(running_loss)), **val_results)

        # change the position of the scheduler:
        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(running_loss)

        wandb.log({'train-epoch-loss': train_loss})


    return train_loss


# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, device, criterion):
    print('Validating...')

    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in val_dataloader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        outputs = model(x)
        loss = criterion(outputs, y)

        # loss calculation over batch
        running_loss.append(loss.cpu().numpy())

        # accuracy calculation over batch
        outputs = torch.sigmoid(outputs)
        outputs = torch.round(outputs)
        y_true.append(y.cpu())
        y_pred.append(outputs.cpu())

    y_true = torch.cat(y_true, 0).numpy()
    y_pred = torch.cat(y_pred, 0).numpy()
    val_loss = np.mean(running_loss)
    wandb.log({'validation-loss': val_loss})
    acc = 100. * np.mean(y_true == y_pred)
    wandb.log({'validation-accuracy': acc})
    return {'val_acc': acc, 'val_loss': val_loss}


# MAIN def
def main():

    parser = argparse.ArgumentParser(description='Training Args.')

    parser.add_argument('--cf', '-config_file', required='True', type=str, metavar='config_file', help='Configuration .yaml file')

    parser_args = parser.parse_args()

    cf_file = vars(parser_args)

    # initialize parser
    with open(cf_file, 'r') as stream:
        args=yaml.safe_load(stream)


    # initialize weights and biases:
    wandb.init(project=args['project_name'], name=args['name'], group=args["group"], save_code=True, config=args, mode='disabled')

    # initialize model:
    model = network()
    # model = timm.create_model('resnet18', pretrained=False, num_classes=1)
    model = model.to(args['device'])

    train_transforms = augmentations.get_training_augmentations(args['aug'])
    valid_transforms = augmentations.get_validation_augmentations()

    # set the paths for training
    train_dataset = dataset2(
        args['train_dir'], train_transforms)
    val_dataset = dataset2(
        args['valid_dir'], valid_transforms)

    # defining data loaders:
    train_dataloader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=False)

    # setting the optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # setting the scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=5, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    # directory:
    save_dir = args['save_dir']
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set value for min-loss:
    min_loss, val_results = float('inf'), {}
    print('Training starts...')
    for epoch in range(args['epochs']):

        wandb.log({'epoch': epoch})
        train_epoch(model, train_dataloader=train_dataloader, optimizer=optimizer, criterion=criterion,
                    scheduler=None, epoch=epoch, val_results=val_results, scheme=args['scheme'], device=args["device"])
        val_results = validate_epoch(model, val_dataloader=val_dataloader, criterion=criterion, device=args["device"])

        if val_results['val_loss'] < min_loss:
            min_loss = val_results['val_loss'].copy()
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best-ckpt.pt'))


if __name__ == '__main__':
    main()