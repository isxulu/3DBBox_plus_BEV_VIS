#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import l1_loss, l2_loss
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from collections import OrderedDict
# from render_video import phi2matrix

def rt_loss(rt_optimized, rt_gt, skip_train, skip_test):
    loss_R = 0
    loss_t = 0
    count = 0
    for track_id in rt_gt.keys():
        for t in rt_gt[track_id]['B2W'].keys():
            if skip_train and int(t) % 2 == 0:
                continue
            if skip_test and int(t) % 2 == 1:
                continue
            try:
                B2W_optimized = torch.Tensor(rt_optimized[track_id]['B2W'][t]).float().cuda()
            except KeyError:
                print(f'{track_id} did not occur in prediction on timestamp{t}')
                continue
            B2W_gt = torch.Tensor(rt_gt[track_id]['B2W'][t]).float().cuda()

            # if -1 <= ((torch.trace(B2W_gt[:3, :3] @ torch.inverse(B2W_optimized[:3, :3])) - 1) / 2) <= 1:
            error_R = torch.arccos((torch.trace(B2W_gt[:3, :3] @ torch.inverse(B2W_optimized[:3, :3])) - 1) / 2)
            print(error_R)
            # else:
            #     error_R = 0
            error_t = l2_loss(B2W_gt[:3, 3], B2W_optimized[:3, 3])
            if error_R > 0.000000000001:
                loss_R += error_R 
            loss_t += error_t
            count += 1.0
    
    print(count)
    return loss_R/count, loss_t/count

def evaluate(model_path, optimized_Rt_path, Rtgt_path, train_view):
    # import ipdb; ipdb.set_trace()
    with open(optimized_Rt_path[0]) as json_file:
    # with open(os.path.join(model_path[0], 'B2W.json')) as json_file:
        B2W_optimized = json.load(json_file)
        rt_optimized = B2W_optimized['bbox']
    
    # if ('phi' in rt_optimized['1'].keys()) or ('phi_tilde' in rt_optimized['1'].keys()):
    #     rt_optimized = phi2matrix(rt_optimized)


    with open(Rtgt_path[0]) as json_file:
        B2W_gt = json.load(json_file)
        rt_gt = B2W_gt['bbox']
    error_R, error_t = rt_loss(rt_optimized, rt_gt, train_view)

    rt_loss_dict = {'error_R': error_R.item(), 'error_t': error_t.item()}
    print(rt_loss_dict)
    if train_view:
        with open(model_path[0] + "/train_view_rt_loss.json", 'w') as fp:
            json.dump(rt_loss_dict, fp, indent=True)
    else:
        with open(model_path[0] + "/test_view_rt_loss.json", 'w') as fp:
            json.dump(rt_loss_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gt_Rt_path', '-gt', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--optimized_Rt_path', '-op', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--train_view", action="store_true", default=False)
    args = parser.parse_args()
    evaluate(args.model_paths, args.optimized_Rt_path, args.gt_Rt_path, args.train_view)