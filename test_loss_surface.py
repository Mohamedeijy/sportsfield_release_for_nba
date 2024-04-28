"""
test the error registration network on homography data
"""
import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
print(os.path.abspath(os.getcwd()))
from torch.utils.data import DataLoader
from models import init_guesser, end_2_end_optimization_helper, loss_surface
from options import options
from datasets import aligned_dataset, noise_dataset
from utils import metrics, utils, warp
import torch.nn as nn
from torchsummary import summary
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def evaluate_model(loader, loss_surface, loss_fn, batch_size, iou):
    loss_surface.eval()
    losses = []
    with torch.no_grad():
        for _, data_batch in enumerate(loader):
            frame, _, gt_homography, perturbed_homography, perturbed_warped_template = data_batch
            x = (frame, perturbed_warped_template)
            inferred_IoU = loss_surface(x)
            original_IoU = iou(perturbed_homography, gt_homography) # second index is IoU whole
            _, original_IoU_whole = original_IoU
            original_IoU_whole = utils.to_torch(original_IoU_whole).reshape(inferred_IoU.shape[0],1)
            # get ground truth transforming default corners with gt_homographies
            loss = loss_fn(original_IoU_whole, inferred_IoU)
            losses.append(loss.cpu())
    test_mse = np.mean(losses)
    return test_mse


def main():
    utils.fix_randomness()
    # load_weights_upstream='pretrained_init_guess',
    # imagenet_pretrain=False
    opt = options.set_loss_surface_options()

    noise_test_dataset = noise_dataset.NoiseDatasetFactory.get_noise_dataset(opt=opt, dataset_type='test')
    noise_test_loader = DataLoader(noise_test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # model
    loss_surface_model = loss_surface.ErrorModelFactory.get_error_model(opt)
    loss_surface_model = utils.set_model_device(loss_surface_model)

    lr = 1e-4
    criterion = nn.MSELoss()
    iou = metrics.IOU(opt)
    test_error = evaluate_model(loader=noise_test_loader, loss_surface=loss_surface_model, loss_fn=criterion,
                                batch_size=opt.batch_size, iou=iou)

    print(f"Test error (MSE): {test_error}")


if __name__ == '__main__':
    main()
