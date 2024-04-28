"""
train the loss surface estimator on homography data
"""
import sys
import os

import imageio

sys.path.append(os.path.abspath(os.getcwd()))
print(os.path.abspath(os.getcwd()))
from torch.utils.data import DataLoader
from models import loss_surface, end_2_end_optimization_helper
from options import options
from datasets import aligned_dataset, noise_dataset
from utils import metrics, utils, warp, image_utils
import torch.nn as nn
from torchsummary import summary
import torch
import numpy as np

import matplotlib.pyplot as plt


def train_batch(batch, loss_surface, optimizer, loss_fn, batch_size, iou):
    frame, _, gt_homography, perturbed_homography, perturbed_warped_template = batch
    # print(f"frame: {frame.shape}\ngt_homography: {gt_homography.shape}\nperturbed_homography: "
    #       f"{perturbed_homography.shape}\nperturbed_warped_template: {perturbed_warped_template.shape}")
    x = (frame, perturbed_warped_template)
    loss_surface.train()
    optimizer.zero_grad()
    inferred_IoU = loss_surface(x)
    original_IoU = iou(perturbed_homography, gt_homography) # second index is IoU whole
    _, original_IoU_whole = original_IoU
    original_IoU_whole = utils.to_torch(original_IoU_whole).reshape(inferred_IoU.shape[0],1)
    # register loss and update weights
    loss = loss_fn(original_IoU_whole, inferred_IoU)
    loss.backward()
    optimizer.step()
    return loss


def validate_batch(batch, loss_surface, loss_fn, batch_size, iou):
    frame, _, gt_homography, perturbed_homography, perturbed_warped_template = batch
    x = (frame, perturbed_warped_template)
    loss_surface.eval()
    with torch.no_grad():
        inferred_IoU = loss_surface(x)
    original_IoU = iou(perturbed_homography, gt_homography) # second index is IoU whole
    _, original_IoU_whole = original_IoU
    original_IoU_whole = utils.to_torch(original_IoU_whole).reshape(inferred_IoU.shape[0],1)
    # get ground truth transforming default corners with gt_homographies
    loss = loss_fn(original_IoU_whole, inferred_IoU)
    return loss


def main():

    utils.fix_randomness()
    opt = options.set_loss_surface_options()

    noise_train_dataset = noise_dataset.NoiseDatasetFactory.get_noise_dataset(opt=opt, dataset_type='train')
    noise_train_loader = DataLoader(noise_train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    noise_val_dataset = noise_dataset.NoiseDatasetFactory.get_noise_dataset(opt=opt, dataset_type='val')
    noise_val_loader = DataLoader(noise_val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # model
    loss_surface_model = loss_surface.ErrorModelFactory.get_error_model(opt)
    loss_surface_model = utils.set_model_device(loss_surface_model)
    # summary(loss_surface_model, (6, 640, 640))

    lr = 5e-4
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(params=loss_surface_model.parameters(), lr=lr)
    iou = metrics.IOU(opt)

    # train & val loop
    best_loss = -1
    best_model_state = loss_surface_model.state_dict()
    train_loss, val_loss = [], []
    for epoch in range(opt.train_epochs):
        print(f"Epoch {epoch + 1} out of {opt.train_epochs}.")
        epoch_train_loss, epoch_val_loss = [], []
        for _, data_batch in enumerate(noise_train_loader):
            loss = train_batch(batch=data_batch, loss_surface=loss_surface_model, optimizer=optim, loss_fn=criterion,
                               batch_size=opt.batch_size, iou=iou)
            epoch_train_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss))
        print(f"Training loss: {train_loss[-1]:.4f}")

        for _, data_batch in enumerate(noise_val_loader):
            loss = validate_batch(batch=data_batch, loss_surface=loss_surface_model, loss_fn=criterion,
                                  batch_size=opt.batch_size, iou=iou)
            epoch_val_loss.append(loss.item())
        val_loss.append(np.mean(epoch_val_loss))
        print(f"Validation loss: {val_loss[-1]:.4f}")
        if val_loss[-1] < best_loss:
            best_model_state = loss_surface_model.state_dict()
            best_loss = val_loss

    loss_surface_model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(opt.out_dir, 'pretrained_loss_surface', 'checkpoint.pth.tar'))
    epochs = np.arange(opt.train_epochs) + 1
    plt.plot(epochs, np.float32(train_loss), 'bo', label='Training loss')
    plt.plot(epochs, np.float32(val_loss), 'r', label='Val loss')
    plt.title('Training and Validation loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.savefig(os.path.join(opt.out_dir, 'pretrained_loss_surface', 'loss_fig.png'))
    plt.close()


if __name__ == '__main__':
    main()
