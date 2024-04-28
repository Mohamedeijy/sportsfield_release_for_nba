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


def evaluate_model(loader, base_model, loss_fn, batch_size):
    base_model.eval()
    losses = []
    with torch.no_grad():
        for _, data_batch in enumerate(loader):
            frame, _, gt_homography = data_batch
            inferred_corners = base_model(frame)
            # get ground truth transforming default corners with homography
            lower_canon_6pts = end_2_end_optimization_helper.get_default_canon4pts(batch_size, canon4pts_type='six')
            original_corners = warp.get_six_corners(gt_homography, lower_canon_6pts[0])
            original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
            # register loss
            loss = loss_fn(original_corners, inferred_corners)
            losses.append(loss.cpu())
    test_mse = np.mean(losses)
    return test_mse


def main():
    utils.fix_randomness()
    # load_weights_upstream='pretrained_init_guess',
    # imagenet_pretrain=False
    opt = options.set_init_guesser_options()

    test_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    print(f"test loader batch size: {test_loader.batch_size}")

    # model
    initial_guesser = init_guesser.InitialGuesserFactory.get_initial_guesser(opt)
    initial_guesser = utils.set_model_device(initial_guesser)
    summary(initial_guesser, (3, 640, 640))

    lr = 1e-4
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(params=initial_guesser.parameters(), lr=lr)

    test_error = evaluate_model(loader=test_loader, base_model=initial_guesser, loss_fn=criterion,
                                batch_size=opt.batch_size)

    print(f"Test error (MSE): {test_error}")


if __name__ == '__main__':
    main()
