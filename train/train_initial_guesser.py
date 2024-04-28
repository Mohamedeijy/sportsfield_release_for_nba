"""
train the initial guesser on homography data
"""
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
print(os.path.abspath(os.getcwd()))
from torch.utils.data import DataLoader
from models import init_guesser, end_2_end_optimization_helper
from options import options
from datasets import aligned_dataset
from utils import metrics, utils, warp
import torch.nn as nn
from torchsummary import summary
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def train_batch(frame, homography, base_model, optimizer, loss_fn, batch_size):
    base_model.train()
    optimizer.zero_grad()
    # inference
    inferred_corners = base_model(frame)
    # get ground truth transforming default corners with homography
    lower_canon_6pts = end_2_end_optimization_helper.get_default_canon4pts(batch_size, canon4pts_type='six')
    original_corners = warp.get_six_corners(homography, lower_canon_6pts[0])
    original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
    # register loss and update weights
    loss = loss_fn(original_corners, inferred_corners)
    loss.backward()
    optimizer.step()
    return loss


def validate_batch(frame, homography, base_model, loss_fn, batch_size):
    base_model.eval()
    with torch.no_grad():
        # inference
        inferred_corners = base_model(frame)
    # get ground truth transforming default corners with homography
    lower_canon_6pts = end_2_end_optimization_helper.get_default_canon4pts(batch_size, canon4pts_type='six')
    original_corners = warp.get_six_corners(homography, lower_canon_6pts[0])
    original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
    # register loss
    loss = loss_fn(original_corners, inferred_corners)
    return loss


def main():

    # repurposing the testing script for testing end2end_optim for training the initial guesser
    utils.fix_randomness()
    opt = options.set_init_guesser_options()

    train_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    val_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'val')
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # model
    initial_guesser = init_guesser.InitialGuesserFactory.get_initial_guesser(opt)
    initial_guesser = utils.set_model_device(initial_guesser)
    summary(initial_guesser, (3, 640, 640))

    lr = 1e-4
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(params=initial_guesser.parameters(), lr=lr)

    # train & val loop
    best_loss = -1
    best_model_state = initial_guesser.state_dict()
    train_loss, val_loss = [], []
    for epoch in range(opt.train_epochs):
        print(f"Epoch {epoch + 1} out of {opt.train_epochs}.")
        epoch_train_loss, epoch_val_loss = [], []
        for _, data_batch in enumerate(train_loader):
            frame, _, gt_homography = data_batch
            loss = train_batch(frame=frame, homography=gt_homography, base_model=initial_guesser, optimizer=optim,
                               loss_fn=criterion, batch_size=opt.batch_size)
            epoch_train_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss))
        print(f"Training loss: {train_loss[-1]:.4f}")

        for _, data_batch in enumerate(val_loader):
            frame, _, gt_homography = data_batch
            loss = validate_batch(frame=frame, homography=gt_homography, base_model=initial_guesser, loss_fn=criterion,
                                  batch_size=opt.batch_size)
            epoch_val_loss.append(loss.item())
        val_loss.append(np.mean(epoch_val_loss))
        print(f"Validation loss: {val_loss[-1]:.4f}")
        if val_loss[-1] < best_loss:
            best_model_state = initial_guesser.state_dict()
            best_loss = val_loss

    initial_guesser.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(opt.out_dir, 'pretrained_init_guess', 'checkpoint.pth.tar'))
    epochs = np.arange(opt.train_epochs)+1
    plt.plot(epochs, np.float32(train_loss), 'bo', label='Training loss')
    plt.plot(epochs, np.float32(val_loss), 'r', label='Val loss')
    plt.title('Training and Validation loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.savefig(os.path.join(opt.out_dir, 'pretrained_init_guess', 'loss_fig.png'))
    plt.close()


if __name__ == '__main__':
    main()
