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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def phi2Tmatrix(phi: torch.Tensor):
    T = torch.zeros((4, 4)).float().cuda()
    T[0, 0] += torch.cos(phi[0]) * torch.cos(phi[2]) + torch.sin(phi[0]) * torch.sin(phi[1]) * torch.sin(phi[2])
    T[0, 1] += torch.cos(phi[2]) * torch.sin(phi[0]) * torch.sin(phi[1]) - torch.cos(phi[0]) * torch.sin(phi[1])
    T[0, 2] += torch.cos(phi[1]) * torch.sin(phi[0])
    T[1, 0] += torch.cos(phi[1]) * torch.sin(phi[2])
    T[1, 1] += torch.cos(phi[1]) * torch.cos(phi[2])
    T[1, 2] -= torch.sin(phi[1])
    T[2, 0] += torch.cos(phi[0]) * torch.sin(phi[1]) * torch.sin(phi[2]) - torch.sin(phi[0]) * torch.cos(phi[2])
    T[2, 1] += torch.sin(phi[0]) * torch.sin(phi[2]) + torch.cos(phi[0]) * torch.cos(phi[2]) * torch.sin(phi[1])
    T[2, 2] += torch.cos(phi[0]) * torch.cos(phi[1])
    
    T[:3, 3] += phi[3:]
    T[3, 3] = 1
    return T

def velocity_loss_phi(track_phi, timestamp):
    ones_point = torch.ones((4, 1)).float().cuda()
    if 'phi' in track_phi.keys():
        B2W_pre = phi2Tmatrix(track_phi['phi'][str(timestamp-1)])
        B2W_now = phi2Tmatrix(track_phi['phi'][str(timestamp)])
        B2W_next = phi2Tmatrix(track_phi['phi'][str(timestamp+1)])
    if 'phi_tilde' in track_phi.keys():
        P_inv_pre = track_phi['P_inv'][str(timestamp-1)]
        P_inv_now = track_phi['P_inv'][str(timestamp)]
        P_inv_next = track_phi['P_inv'][str(timestamp+1)]

        phi_tilde_pre = track_phi['phi_tilde'][str(timestamp-1)]
        phi_tilde_now = track_phi['phi_tilde'][str(timestamp)]
        phi_tilde_next = track_phi['phi_tilde'][str(timestamp+1)]

        phi_pre = torch.mm(P_inv_pre, phi_tilde_pre).squeeze()
        phi_now = torch.mm(P_inv_now, phi_tilde_now).squeeze()
        phi_next = torch.mm(P_inv_next, phi_tilde_next).squeeze()

        B2W_pre = phi2Tmatrix(phi_pre)
        B2W_now = phi2Tmatrix(phi_now)
        B2W_next = phi2Tmatrix(phi_next)

    point_pre = torch.mm(B2W_pre, ones_point)[:3]
    point_now = torch.mm(B2W_now, ones_point)[:3]
    point_next = torch.mm(B2W_next, ones_point)[:3]

    velocity1 = point_now - point_pre
    velocity2 = point_next - point_now

    norm1 = torch.norm(velocity1)
    norm2 = torch.norm(velocity2)

    cosine_similarity = torch.dot(velocity1.squeeze(), velocity2.squeeze()) / (norm1 * norm2)

    norm_loss = l1_loss(norm1, norm2)

    loss = 1 - cosine_similarity**2
    loss = 50 * loss + 10 * norm_loss
    return loss

def velocity_loss(track_B2W, timestamp):
    ones_point = torch.ones((4, 1)).float().cuda()
    B2W_pre = track_B2W[str(timestamp-1)]
    B2W_now = track_B2W[str(timestamp)]
    B2W_next = track_B2W[str(timestamp+1)]

    point_pre = torch.mm(B2W_pre, ones_point)[:3]
    point_now = torch.mm(B2W_now, ones_point)[:3]
    point_next = torch.mm(B2W_next, ones_point)[:3]

    velocity1 = point_now - point_pre
    velocity2 = point_next - point_now

    norm1 = torch.norm(velocity1)
    norm2 = torch.norm(velocity2)

    cosine_similarity = torch.dot(velocity1.squeeze(), velocity2.squeeze()) / (norm1 * norm2)

    norm_loss = l1_loss(norm1, norm2)

    loss = 1 - cosine_similarity**2
    loss = 5000 * loss + 500 * norm_loss
    return loss


def bbox_loss(xyz, vertices, lambda1=0.02):
    up_loss = torch.sum((xyz - vertices.max(0)[0]).clamp(min=0))
    down_loss = torch.sum((-xyz + vertices.min(0)[0]).clamp(min=0))
    loss = up_loss + down_loss
    return loss * lambda1

def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.abs((network_output - gt)[:, mask]).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

