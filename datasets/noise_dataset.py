'''
Aligned frame, warped template, and warp params.
'''

import abc
from abc import ABC

from torch.utils import data
import imageio
import tables
import os
import pandas as pd
import numpy as np
import ast
from datasets import aligned_dataset

from utils import utils, warp, image_utils


class NoiseDatasetFactory():
    '''factory of aligned dataset
    '''

    @staticmethod
    def get_noise_dataset(opt, dataset_type):
        '''[summary]

        Arguments:
            dataset_type {[str]} -- [train/val/test]
        '''

        if opt.dataset_name == 'nba':
            dset = NBANoiseDataset(opt, dataset_type)
        else:
            raise ValueError('unknown dataset: {0}'.format(opt.dataset_name))
        return dset


class NoiseDataset(data.Dataset, abc.ABC):
    '''
    This dataset provides current frame, current warped template,
    and the homography parameters.
    '''

    def __init__(self, opt, dataset_type):
        self.opt = opt
        assert dataset_type in ['train', 'val', 'test'], 'unknown dataset type {0}'.format(dataset_type)
        self.dataset_type = dataset_type
        self.load_template()
        # self.load_h5_file(dataset_type)
        self.load_dataframe_from_initial_loader(opt, dataset_type)

    # @abc.abstractmethod
    # def load_h5_file(self, dataset_type):
    #     pass

    @abc.abstractmethod
    def load_dataframe_from_initial_loader(self, opt, dataset_type):
        pass

    def load_template(self):
        self.template_path = self.opt.template_path
        self.template = imageio.v2.imread(self.template_path, pilmode='RGB') / 255.0
        if self.opt.coord_conv_template:
            self.template = image_utils.rgb_template_to_coord_conv_template(self.template)

        self.template = utils.np_img_to_torch_img(self.template)
        if self.opt.need_single_image_normalization:
            self.template = image_utils.normalize_single_image(self.template)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = self.get_image_by_index(index)
        warped_template = self.get_warped_template_by_index(index)
        gt_homography = self.get_gt_homography_by_index(index)
        perturbed_homographies = self.get_perturbed_homography_by_index(index)
        perturbed_warped_template = self.get_perturbed_warped_template_by_index(index)
        return img, warped_template, gt_homography, perturbed_homographies, perturbed_warped_template

    def get_image_by_index(self, index):
        # img = self.raw_data.root.frames[index]
        img = self.df.loc[index, 'frame']
        img = utils.np_img_to_torch_img(img)
        return img

    def get_gt_homography_by_index(self, index):
        # homography = self.raw_data.root.homographies[index]
        gt_homography = self.df.loc[index, 'gt_homography']
        gt_homography = utils.to_torch(gt_homography)
        gt_homography = gt_homography / gt_homography[2:3, 2:3]
        return gt_homography

    def get_warped_template_by_index(self, index):
        warped_template = self.df.loc[index, 'warped_template']
        warped_template = utils.to_torch(warped_template)
        return warped_template

    def get_perturbed_homography_by_index(self, index):
        perturbed_homography = self.df.loc[index, 'perturbed_homography']
        perturbed_homography = utils.to_torch(perturbed_homography)
        perturbed_homography = perturbed_homography / perturbed_homography[2:3, 2:3]
        return perturbed_homography

    def get_perturbed_warped_template_by_index(self, index):
        perturbed_warped_template = self.df.loc[index, 'perturbed_warped_template']
        perturbed_warped_template = utils.to_torch(perturbed_warped_template)
        perturbed_warped_template = perturbed_warped_template.permute(2, 0, 1)
        return perturbed_warped_template


class NBADataset(abc.ABC):

    def load_dataframe_from_initial_loader(self, opt, dataset_type):
        number_perturbations = 3
        # create df with frame, warped_template, gt_homographies, perturbed_homographies, perturbed_warped_template
        self.df = pd.DataFrame(
            columns=['frame', 'warped_template', 'gt_homography', 'perturbed_homography', 'perturbed_warped_template'])
        # load initial dataloader for dataset_type (retrieves data from actual csv) with batch size 1
        initial_dataset = aligned_dataset.AlignedDatasetFactory.get_aligned_dataset(opt=opt, dataset_type=dataset_type)
        initial_loader = data.DataLoader(initial_dataset, batch_size=1, shuffle=False, num_workers=0)

        # go through each frame, generate perturbed homography from ground truth homography, and corresponding
        # transformation of the template
        for _, batch in enumerate(initial_loader):
            frame, warped_template, gt_homography = batch
            for _ in range(number_perturbations):
                perturbed_homography = warp.perturbe_homography(gt_homography)
                perturbed_warped_template = warp.warp_image(frame, perturbed_homography, out_shape=frame.shape[-2:])[0]
                row = pd.DataFrame({'frame': [utils.torch_img_to_np_img(frame[0])],
                                    'warped_template': [utils.torch_img_to_np_img(warped_template[0])],
                                    'gt_homography': [utils.to_numpy(gt_homography[0])],
                                    'perturbed_homography': [utils.to_numpy(perturbed_homography)],
                                    'perturbed_warped_template': [utils.torch_img_to_np_img(perturbed_warped_template)], })
                self.df = pd.concat([self.df, row], ignore_index=True)
        self.num_samples = len(self.df)


class NBANoiseDataset(NBADataset, NoiseDataset):
    def __init__(self, opt, dataset_type):
        super().__init__(opt, dataset_type)
