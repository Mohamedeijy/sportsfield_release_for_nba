"""
train the initial guesser on homography data
"""
from torch.utils.data import DataLoader
from models import init_guesser
from options import options
from datasets import aligned_dataset
from utils import metrics, utils
import torch


def main():
    # repurposing the testing script for testing end2end_optim for training the initial guesser
    utils.fix_randomness()
    opt = options.set_end2end_optim_options()
    assert opt.iou_space == "part_and_whole"

    train_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt, 'train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    base_model =
