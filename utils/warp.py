import random

import cv2
import imageio
import torch
import numpy as np

from models import end_2_end_optimization_helper
from utils import utils, image_utils


def warp_image(img, H, out_shape=None, input_grid=None):
    if out_shape is None:
        out_shape = img.shape[-2:]
    if len(img.shape) < 4:
        img = img[None]
    if len(H.shape) < 3:
        H = H[None]
    assert img.shape[0] == H.shape[0], 'batch size of images do not match the batch size of homographies'
    batchsize = img.shape[0]
    # create grid for interpolation (in frame coordinates)
    if input_grid is None:
        y, x = torch.meshgrid([
            torch.linspace(-utils.BASE_RANGE, utils.BASE_RANGE,
                           steps=out_shape[-2]),
            torch.linspace(-utils.BASE_RANGE, utils.BASE_RANGE,
                           steps=out_shape[-1])
        ])
        x = x.to(img.device)
        y = y.to(img.device)
    else:
        x, y = input_grid
    x, y = x.flatten(), y.flatten()

    # append ones for homogeneous coordinates
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy = xy.repeat([batchsize, 1, 1])  # shape: (B, 3, N)
    # warp points to model coordinates
    xy_warped = torch.matmul(H, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # build grid
    grid = torch.stack([
        x_warped.view(batchsize, *out_shape[-2:]),
        y_warped.view(batchsize, *out_shape[-2:])
    ],
        dim=-1)

    # sample warped image
    warped_img = torch.nn.functional.grid_sample(
        img, grid, mode='bilinear', padding_mode='zeros')

    if utils.hasnan(warped_img):
        print('nan value in warped image! set to zeros')
        warped_img[utils.isnan(warped_img)] = 0

    return warped_img


def get_four_corners(homo_mat, canon4pts=None):
    '''
    calculate the 4 corners after transformation, from frame to template
    assuming the original 4 corners of the frame are [+-0.5, +-0.5]
    note: this function supports batch processing
    Arguments:
        homo_mat {[type]} -- [homography, shape: (B, 3, 3) or (3, 3)]

    Return:
        xy_warped -- torch.Size([B, 2, 4])
    '''
    # append ones for homogeneous coordinates
    if homo_mat.shape == (3, 3):
        homo_mat = homo_mat[None]
    assert homo_mat.shape[1:] == (3, 3)
    if canon4pts is None:
        canon4pts = utils.to_torch(utils.FULL_CANON4PTS_NP())
    assert canon4pts.shape == (4, 2)
    x, y = canon4pts[:, 0], canon4pts[:, 1]
    xy = torch.stack([x, y, torch.ones_like(x)])
    # warp points to model coordinates
    xy_warped = torch.matmul(homo_mat, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)
    xy_warped = xy_warped / (z_warped + 1e-8)
    return xy_warped

def get_six_corners(homo_mat, canon6pts=None):
    '''
    calculate the 6 corners after transformation, from frame to template
    assuming the original 6 points of the frame are [-0.5,-0.3;−0.5,0.1;−0.5,0.5;0.5,-0.3;0.5,0.5; 0.5,0.1;]
    i.e. corners of the lower 2/5ths and lower 4/5ths of the frame
    note: this function supports batch processing
    Arguments:
        homo_mat {[type]} -- [homography, shape: (B, 3, 3) or (3, 3)]

    Return:
        xy_warped -- torch.Size([B, 2, 6])
    '''
    # append ones for homogeneous coordinates
    if homo_mat.shape == (3, 3):
        homo_mat = homo_mat[None]
    assert homo_mat.shape[1:] == (3, 3)
    if canon6pts is None:
        canon6pts = utils.to_torch(utils.CANON6PTS_NP())
    assert canon6pts.shape == (6, 2)
    x, y = canon6pts[:, 0], canon6pts[:, 1]
    xy = torch.stack([x, y, torch.ones_like(x)])
    # warp points to model coordinates
    xy_warped = torch.matmul(homo_mat, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)
    xy_warped = xy_warped / (z_warped + 1e-8)
    return xy_warped


def perturbe_homography(gt_homography, global_translation_param=0.05, local_translation_param=0.02):
    # get canon 6 points
    lower_canon_6pts = end_2_end_optimization_helper.get_default_canon4pts(1, canon4pts_type='six')
    # compute destination vector from canon 6 points and homography matrix
    original_corners = get_six_corners(gt_homography, lower_canon_6pts[0])
    original_corners = torch.flatten(original_corners.permute(0, 2, 1), start_dim=1)
    # add a global noise to the whole vector
    corners_with_noise = original_corners + random.uniform(-global_translation_param, global_translation_param)
    # add local noise to each element the resulting vector
    corners_with_noise = corners_with_noise + utils.set_tensor_device(torch.FloatTensor(corners_with_noise.size())\
        .uniform_(-local_translation_param, local_translation_param))
    corners_with_noise = torch.reshape(corners_with_noise, (-1, 6, 2))
    # generate  new homography matrix from canon points and new vector
    perturbed_homography, _ = cv2.findHomography(utils.to_numpy(lower_canon_6pts), utils.to_numpy(corners_with_noise), cv2.RANSAC)
    # print(f"original_corners:\n{original_corners}")
    # print(f"corners_with_noise:\n{corners_with_noise}")
    # print(f"gt_homography:\n{gt_homography}")
    # print(f"perturbed_homography:\n{perturbed_homography}")
    return utils.to_torch(perturbed_homography)
