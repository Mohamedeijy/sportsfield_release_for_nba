'''
Aligned frame, warped template, and warp params.
'''

import abc

from torch.utils import data
import imageio
import tables
import os
import pandas as pd
import numpy as np
import ast

from utils import utils, warp, image_utils


class AlignedDatasetFactory():
    '''factory of aligned dataset
    '''

    @staticmethod
    def get_aligned_dataset(opt, dataset_type):
        '''[summary]

        Arguments:
            dataset_type {[str]} -- [train/val/test]
        '''

        if opt.dataset_name == 'nba':
            dset = NBAAlignedDataset(opt, dataset_type)
        else:
            raise ValueError('unknown dataset: {0}'.format(opt.dataset_name))
        return dset


class AlignedDataset(data.Dataset, abc.ABC):
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
        self.load_dataframe(dataset_type)

    # @abc.abstractmethod
    # def load_h5_file(self, dataset_type):
    #     pass

    @abc.abstractmethod
    def load_dataframe(self, dataset_type):
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
        gt_homography = self.get_homography_by_index(index)
        warped_template = warp.warp_image(self.template, gt_homography, out_shape=img.shape[-2:])[0]
        return img, warped_template, gt_homography

    def get_homography_by_index(self, index):
        # homography = self.raw_data.root.homographies[index]
        gt_homography = ast.literal_eval(self.df.loc[index, 'gt_homography'])
        gt_homography = np.float32(gt_homography).flatten().reshape(3, 3)
        gt_homography = utils.to_torch(gt_homography)
        gt_homography = gt_homography / gt_homography[2:3, 2:3]
        return gt_homography

    def get_image_by_index(self, index):
        # img = self.raw_data.root.frames[index]
        img_path = os.path.join(self.root_dir_path, self.df.loc[index, 'image_path'])
        img = imageio.v2.imread(img_path, pilmode='RGB')
        img = img[..., [2, 1, 0]]
        img = img / 255.0
        img = utils.to_torch(img).permute(2, 0, 1)
        if self.opt.need_single_image_normalization:
            img = image_utils.normalize_single_image(img)
        return img


class NBADataset(abc.ABC):
    def load_h5_file(self, dataset_type):
        if dataset_type == 'test':
            self.dataset_path = self.opt.test_dataset_path
        else:
            raise ValueError('unknown dataset type:{0}'.format(dataset_type))
        self.raw_data = tables.open_file(self.dataset_path, mode='r')
        assert len(self.raw_data.root.frames) == len(
            self.raw_data.root.homographies)
        self.num_samples = len(self.raw_data.root.frames)

    def load_dataframe(self, dataset_type):
        self.root_dir_path = os.path.join(self.opt.root_dir_path, dataset_type)
        print(f"root dir path: {self.root_dir_path}")
        self.df = pd.read_csv(os.path.join(self.root_dir_path, 'annotations_' + dataset_type + '.csv'))
        self.num_samples = len(self.df)


class NBAAlignedDataset(NBADataset, AlignedDataset):
    def __init__(self, opt, dataset_type):
        super().__init__(opt, dataset_type)
