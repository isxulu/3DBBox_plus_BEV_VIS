from arguments import ModelParams
import os
import json
import numpy as np
import open3d as o3d
from scene import GaussianModel
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
from transforms3d.euler import euler2mat, mat2euler

def rot2Euler(R):
    sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2,1] , R[2,2])
        y = torch.atan2(-R[2,0], sy)
        z = torch.atan2(R[1,0], R[0,0])
    else:
        x = torch.atan2(-R[1,2], R[1,1])
        y = torch.atan2(-R[2,0], sy)
        z = 0

    return torch.stack([x,y,z])

class SinglePoint(torch.nn.Module):
    def __init__(self, c) -> None:
        super(SinglePoint, self).__init__()
        self.c = nn.Parameter(c)
        self.phi = nn.Parameter(torch.tensor(0.))
        self.h = nn.Parameter(torch.tensor(1.5))

    def forward(self):
        return self.c, self.phi, self.h


class unicycle(torch.nn.Module):

    def __init__(self, c0, phi0, heights, visible_timestamps, all_timestamps):
        super(unicycle, self).__init__()
        self.c0 = nn.Parameter(c0)
        self.v, self.phi = nn.Parameter(torch.tensor(0.1)), nn.Parameter(phi0)
        self.acc = nn.Parameter(torch.zeros_like(all_timestamps).float())
        self.omg = nn.Parameter(torch.zeros_like(all_timestamps).float())
        self.heights = nn.Parameter(heights)
        self.visible_timestamps = visible_timestamps
        self.all_timestamps = all_timestamps
    
    def forward(self, timestamps):
        pred_V = torch.ones(self.all_timestamps.shape[0], device='cuda').float() * self.v
        pred_V[1:] += torch.cumsum(self.acc, dim=0)[:-1]
        pred_phi = torch.ones(self.all_timestamps.shape[0], device='cuda').float() * self.phi
        pred_phi[1:] += torch.cumsum(self.omg, dim=0)[:-1]
        pred_motion = torch.stack([pred_V * torch.cos(pred_phi), pred_V * torch.sin(pred_phi)], dim=1)
        pred_c = torch.ones(self.all_timestamps.shape[0], 2, device='cuda').float() * self.c0
        pred_c[1:] += torch.cumsum(pred_motion, dim=0)[:-1]
        iids = timestamps - self.visible_timestamps[0]
        preds = pred_c[iids]
        if preds.dim() == 2:
            a, b = preds[:, 0], preds[:, 1]
        else:
            a, b = preds[0], preds[1]
        return a, b, pred_V[iids], pred_phi[iids], self.heights[iids]
    
    def interpolate(self,timestamp):
        pred_V = torch.ones(self.all_timestamps.shape[0], device='cuda').float() * self.v
        pred_V[1:] += torch.cumsum(self.acc, dim=0)[:-1]
        # pred_V = pred_V.clamp(max=0)
        pred_phi = torch.ones(self.all_timestamps.shape[0], device='cuda').float() * self.phi
        pred_phi[1:] += torch.cumsum(self.omg, dim=0)[:-1]
        pred_motion = torch.stack([pred_V * torch.cos(pred_phi), pred_V * torch.sin(pred_phi)], dim=1)
        pred_c = torch.ones(self.all_timestamps.shape[0], 2, device='cuda').float() * self.c0
        pred_c[1:] += torch.cumsum(pred_motion, dim=0)[:-1]
        prev_iid = int(timestamp - self.visible_timestamps[0])
        pred_c, pred_h, pred_phi = pred_c[prev_iid], self.heights[prev_iid], pred_phi[prev_iid]
        pred_c += (timestamp - int(timestamp)) * pred_motion[prev_iid]
        return pred_c, pred_h, pred_phi
    
    def capture(self):
        return (
            self.c0,
            self.v,
            self.phi,
            self.acc,
            self.omg,
            self.heights,
            self.visible_timestamps,
            self.all_timestamps
        )
    
    def restore(self, model_args):
        (
            self.c0,
            self.v,
            self.phi,
            self.acc,
            self.omg,
            self.heights,
            self.visible_timestamps,
            self.all_timestamps
        ) = model_args

    def visualize(self, save_path, noise_centers=None, gt_centers=None):
        a, b, _, _, _ = self.forward(self.visible_timestamps)
        plt.scatter(a.detach().cpu().numpy(), b.detach().cpu().numpy(), marker='x', color='b')
        if noise_centers is not None:
            noise_centers = noise_centers.detach().cpu().numpy()
            plt.scatter(noise_centers[:, 0], noise_centers[:, 1], marker='o', color='r')
        if gt_centers is not None:
            gt_centers = gt_centers.detach().cpu().numpy()
            plt.scatter(gt_centers[:, 0], gt_centers[:, 1], marker='v', color='g')
        plt.axis('equal')
        plt.savefig(save_path)
        plt.close()

    def reg_loss(self):
        return torch.mean(torch.abs(self.acc - torch.mean(self.acc))) + torch.mean(torch.abs(self.omg - torch.mean(self.omg)))


class unicycle2(torch.nn.Module):

    def __init__(self, train_timestamp, centers=None, heights=None, phis=None):
        super(unicycle2, self).__init__()
        self.train_timestamp = train_timestamp
        self.delta = torch.diff(self.train_timestamp)

        if centers is None:
            self.a = nn.Parameter(torch.zeros_like(train_timestamp).float())
            self.b = nn.Parameter(torch.zeros_like(train_timestamp).float())
        else:
            self.a = nn.Parameter(centers[:, 0])
            self.b = nn.Parameter(centers[:, 1])
        
        diff_a = torch.diff(centers[:, 0]) / self.delta
        diff_b = torch.diff(centers[:, 1]) / self.delta
        v = torch.sqrt(diff_a ** 2 + diff_b**2)
        self.v = nn.Parameter(F.pad(v, (0, 1), 'constant', v[-1].item()))
        self.phi = nn.Parameter(phis)

        if heights is None:
            self.h = nn.Parameter(torch.zeros_like(train_timestamp).float())
        else:
            self.h = nn.Parameter(heights)

    def acc_omega(self):
        acc = torch.diff(self.v) / self.delta
        omega = torch.diff(self.phi) / self.delta
        acc = F.pad(acc, (0, 1), 'constant', acc[-1].item())
        omega = F.pad(omega, (0, 1), 'constant', omega[-1].item())
        return acc, omega

    def forward(self, timestamps):
        idx = torch.searchsorted(self.train_timestamp, timestamps, side='left')
        invalid = (idx == self.train_timestamp.shape[0])
        idx[invalid] -= 1
        idx[self.train_timestamp[idx] != timestamps] -= 1
        idx[invalid] += 1
        prev_timestamps = self.train_timestamp[idx]
        delta_t = timestamps - prev_timestamps
        prev_a, prev_b = self.a[idx], self.b[idx]
        prev_v, prev_phi = self.v[idx], self.phi[idx]
        
        acc, omega = self.acc_omega()
        v = prev_v + acc[idx] * delta_t
        phi = prev_phi + omega[idx] * delta_t
        a = prev_a + prev_v * ((torch.sin(phi) - torch.sin(prev_phi)) / (omega[idx] + 1e-6))
        b = prev_b - prev_v * ((torch.cos(phi) - torch.cos(prev_phi)) / (omega[idx] + 1e-6))
        h = self.h[idx]
        return a, b, v, phi, h

    # def forward(self, timestamps):
    #     return self.a, self.b, self.v, self.phi, self.h

    def capture(self):
        return (
            self.a,
            self.b,
            self.v,
            self.phi,
            self.h,
            self.train_timestamp,
            self.delta
        )
    
    def restore(self, model_args):
        (
            self.a,
            self.b,
            self.v,
            self.phi,
            self.h,
            self.train_timestamp,
            self.delta
        ) = model_args

    def visualize(self, save_path, noise_centers=None, gt_centers=None):
        a, b, _, phi, _ = self.forward(self.train_timestamp)
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        phi = phi.detach().cpu().numpy()
        plt.scatter(a, b, marker='x', color='b')
        plt.quiver(a, b, np.ones_like(a) * np.cos(phi), np.ones_like(b) * np.sin(phi), scale=20, width=0.005)
        if noise_centers is not None:
            noise_centers = noise_centers.detach().cpu().numpy()
            plt.scatter(noise_centers[:, 0], noise_centers[:, 1], marker='o', color='gray')
        if gt_centers is not None:
            gt_centers = gt_centers.detach().cpu().numpy()
            plt.scatter(gt_centers[:, 0], gt_centers[:, 1], marker='v', color='g')
        plt.axis('equal')
        plt.savefig(save_path)
        plt.close()

    def reg_loss(self):
        reg = 0
        acc, omega = self.acc_omega()
        reg += torch.mean(torch.abs(torch.diff(acc))) * 1
        reg += torch.mean(torch.abs(torch.diff(omega))) * 1
        reg_a_motion = self.v[:-1] * ((torch.sin(self.phi[1:]) - torch.sin(self.phi[:-1])) / (omega[:-1] + 1e-6)) 
        reg_b_motion = -self.v[:-1] * ((torch.cos(self.phi[1:]) - torch.cos(self.phi[:-1])) / (omega[:-1] + 1e-6))
        reg_a = self.a[:-1] + reg_a_motion
        reg_b = self.b[:-1] + reg_b_motion
        reg += torch.mean((reg_a - self.a[1:])**2 + (reg_b - self.b[1:])**2) * 1
        return reg


def create_unicycle_model(track_B2W, fit_id=None, stage1=True):
    unicycle_models = {}
    for track_id, v in track_B2W.items():
        if fit_id is not None and track_id not in fit_id:
            continue
        b2ws = v['B2W']
        vertices = v['vertices']
        verts = torch.tensor(vertices)
        h_verts = torch.ones(verts.shape[0], 4)
        h_verts[:, :3] = verts
        h_verts = h_verts.cuda()
        centers, heights, phis, visible_timestamps, all_timestamps = [], [], [], [], []
        for t, b2w in b2ws.items():
            t = int(t)
            # b2w = torch.tensor(b2w)
            # w_verts = (b2w @ h_verts.T).T[:, :3]
            # center = torch.mean(w_verts, dim=0)
            centers.append(b2w[[0, 2], 3])
            heights.append(b2w[1, 3])
            eulers = rot2Euler(b2w[:3, :3])
            phis.append(eulers[1])
            visible_timestamps.append(t)
            # fill missing timestamps
            if len(all_timestamps) == 0 or t == all_timestamps[-1] + 1:
                all_timestamps.append(t)
            else:
                for iit in range(all_timestamps[-1]+1, t+1):
                    all_timestamps.append(iit)

        centers = torch.stack(centers, dim=0).cuda()
        visible_timestamps = torch.tensor(visible_timestamps).cuda()
        all_timestamps = torch.tensor(all_timestamps).cuda()
        heights = torch.tensor(heights).cuda()
        phis = torch.tensor(phis).cuda() + torch.pi

        model = unicycle2(visible_timestamps, centers=centers.clone(), heights=heights.clone(), phis=phis.clone())
        l = [
            {'params': [model.a], 'lr': 1e-2, "name": "a"},
            {'params': [model.b], 'lr': 1e-2, "name": "b"},
            {'params': [model.v], 'lr': 1e-3, "name": "v"},
            {'params': [model.phi], 'lr': 1e-4, "name": "phi"},
            {'params': [model.h], 'lr': 0, "name": "h"}
        ]

        # model = unicycle(centers[0], phis[0], heights, visible_timestamps, all_timestamps)
        # l = [
        #     {'params': [model.c0], 'lr': 1e-2, "name": "c0"},
        #     {'params': [model.phi], 'lr': 1e-2, "name": "phi"},
        #     {'params': [model.v], 'lr': 1e-2, "name": "v"},
        #     {'params': [model.acc], 'lr': 1e-3, "name": "acc"},
        #     {'params': [model.omg], 'lr': 1e-3, "name": "omg"},
        #     {'params': [model.heights], 'lr': 0, "name": "h"}
        # ]

        optimizer = optim.Adam(l, lr=0.0)

        if stage1:
            t_range = tqdm(range(3000), desc=f"Fitting {track_id}")
            for iter in t_range:
                # loss = torch.mean((model(model.visible_timestamps)[0] - centers) ** 2)
                a, b, _, _, _ = model(visible_timestamps)
                loss = torch.mean((a - centers[:, 0]) ** 2 + (b - centers[:, 1]) ** 2) * 0.5
                loss += model.reg_loss()
                t_range.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        unicycle_models[track_id] = {'model': model, 
                                    'optimizer': optimizer,
                                    'input_centers': centers}

    return unicycle_models


def test_optimize_unicycle(model, uni_optimizer, views, track_id, gt):
    VERT = torch.tensor(gt['vertices']).cuda()
    VERT = torch.cat([VERT, torch.ones(8, 1, device='cuda')], dim=1)
    W, H = views[0].W, views[0].H
    track_id = int(track_id)
    
    cam_info = {}
    for view in views:
        mask = view.dynamic_mask.clone()
        mask[mask != (track_id + 1)] = 0
        mask[mask == (track_id + 1)] = 1
        if torch.sum(mask) == 0:
            continue
        mask_coord = torch.nonzero(mask).float()
        mask_min = torch.min(mask_coord, dim=0)[0].cuda()
        mask_max = torch.max(mask_coord, dim=0)[0].cuda()
        cam_info[view.timestamp] = [mask, mask_min, mask_max, view.world_view_transform, view.full_proj_transform]

    timestamps, masks, mask_mins, mask_maxs, w2vs, full_projs = [], [], [], [], [], []
    for t, (mask, mask_min, mask_max, w2v, full) in sorted(cam_info.items()):
        timestamps.append(t)
        masks.append(mask)
        mask_mins.append(mask_min)
        mask_maxs.append(mask_max)
        w2vs.append(w2v)
        full_projs.append(full)

    timestamps = torch.tensor(timestamps).cuda()

    t_range = tqdm(range(1, 10001))
    p0, pm = SinglePoint(torch.tensor([0., 100.]).cuda()), SinglePoint(torch.tensor([0., 100.]).cuda())
    optimizer_0 = optim.Adam(p0.parameters(), lr=1e-1)
    optimizer_m = optim.Adam(pm.parameters(), lr=1e-1)
    for ep in t_range:
        
        bbox_loss = 0

        if ep <= 3000:
            train_ids = [0]
            pred_xz, pred_phi, pred_h = p0.forward()
            optimizer = optimizer_0
        elif ep <= 6000:
            train_ids = [timestamps.shape[0] // 2]
            pred_xz, pred_phi, pred_h = pm.forward()
            optimizer = optimizer_m
        else:
            train_ids = list(range(timestamps.shape[0]))
            pred_xz, pred_h, pred_phi = model(timestamps)
            optimizer = uni_optimizer

        cos = torch.cos(-pred_phi)
        sin = torch.sin(-pred_phi)

        if pred_xz.dim() == 2:
            preds_RT = torch.eye(4)[None, ...].repeat(pred_xz.shape[0], 1, 1).float().cuda()
            preds_RT[:, [0,2], 3] = pred_xz
            preds_RT[:, 1, 3] = pred_h
            preds_RT[:, 0,0] = cos
            preds_RT[:, 0,2] = sin
            preds_RT[:, 2,0] = -sin
            preds_RT[:, 2,2] = cos
            preds = torch.einsum("nij,kj->nki", preds_RT, VERT)
        else:
            preds_RT = torch.eye(4).float().cuda()
            preds_RT[[0,2], 3] = pred_xz
            preds_RT[1, 3] = pred_h
            preds_RT[0,0] = cos
            preds_RT[0,2] = sin
            preds_RT[2,0] = -sin
            preds_RT[2,2] = cos
            preds = torch.einsum("ij,kj->ki", preds_RT, VERT)

        for i in train_ids:
            # if i in [16,17,18]:
            #     continue
            if preds.dim() == 3:
                point = preds[i]
            else:
                point = preds
            w2v = w2vs[i]
            full = full_projs[i]
            pred_u, pred_v = proj(w2v, full, point, W, H)
            if pred_u.numel() == 0 or pred_v.numel() == 0:
                min_u, min_v, max_u, max_v = 0., 0., 0., 0.
            else:
                min_u, min_v = torch.min(pred_u).clamp(min=0), torch.min(pred_v).clamp(min=0)
                max_u, max_v = torch.max(pred_u).clamp(max=W), torch.max(pred_v).clamp(max=H)
            gt_min_u, gt_min_v = mask_mins[i][1], mask_mins[i][0]
            gt_max_u, gt_max_v = mask_maxs[i][1], mask_maxs[i][0]

            if ep == 9999:
                plt.imshow(masks[i].detach().cpu().numpy())
                plt.scatter(min_u.detach().cpu(), min_v.detach().cpu(), marker=".", color='red', s=20)
                plt.scatter(max_u.detach().cpu(), max_v.detach().cpu(), marker=".", color='red', s=20)
                plt.savefig(f"/data0/hyzhou/outputs/temp/{i}.png")
                plt.close()

            bbox_loss += torch.abs(min_u - gt_min_u) + torch.abs(min_v - gt_min_v) + torch.abs(max_u - gt_max_u) + torch.abs(max_v - gt_max_v)
        
        bbox_loss /= len(train_ids)
        loss = bbox_loss
        optimizer.zero_grad()
        loss.backward()
        t_range.set_postfix({'bbox': bbox_loss.item()})
        optimizer.step()

        if ep == 6000:
            print(p0.c, pm.c, p0.h.item(), pm.h.item())
            N = model.all_timestamps.shape[0]
            model.c0 = nn.Parameter(p0.c)
            delta = pm.c - p0.c
            phi = torch.atan(delta[1]/delta[0])
            if delta[0] < 0:
                phi += torch.pi
            model.phi = nn.Parameter(phi)
            model.v = nn.Parameter(torch.norm(pm.c - p0.c) / (N // 2))
            heights = F.pad(torch.linspace(p0.h.item(), pm.h.item(), (N // 2)), (0, N - N // 2), 'constant', pm.h.item())
            model.heights = nn.Parameter(heights.cuda())

        if ep > 5000 and ep % 1000 == 0:
            for g in optimizer.param_groups:
                g['lr'] /= 2

    for g in optimizer.param_groups:
        if g['name'] == 'c0':
            g['lr'] = 3e-1
        elif g['name'] == 'phi':
            g['lr'] = 2e-1
        elif g['name'] == 'v':
            g['lr'] = 3e-2
        elif g['name'] == 'h':
            g['lr'] = 3e-2
        elif g['name'] == 'acc':
            g['lr'] = 1e-3
        elif g['name'] == 'omg':
            g['lr'] = 1e-3

    return model


def proj(viewmatrix, projmatrix, homo_point, W, H):
    p_view = homo_point @ (viewmatrix[:, :3])
    mask1 = p_view[:, 2] > 0.1 # near plane

    p_proj = homo_point[mask1] @ projmatrix
    p_proj = (p_proj / (p_proj[:, 3][:, None] + 1e-6))[:, :3]

    u, v = ((p_proj[:, 0] + 1) * W - 1) * 0.5, ((p_proj[:, 1] + 1) * H - 1) * 0.5
    # mask2 = (0 <= u) & (u <= W) & (0 <= v) & (v <= H)
    # u, v = u[mask2], v[mask2] 
    return u, v


def read_bbox_info(source_path):
    if os.path.exists(os.path.join(source_path, "bbox_phi.json")):
        bbox_fn = "bbox_phi.json"
    else:
        bbox_fn = "bbox.json"
    with open(os.path.join(source_path, bbox_fn)) as json_file:
        bbox_data = json.load(json_file)
        track = bbox_data['track']
        track_time = bbox_data['track_time']
        bbox = bbox_data['bbox']
    bbox = Rt_list2tensor(bbox)
    # b2w = bbox['81']['B2W']
    # for t, m in b2w.items():
    #     m[[0, 2], 3] += torch.rand(2) - 0.5
    #     b2w[t] = m
    # bbox['81']['B2W'] = b2w
    return track, track_time, bbox


def create_mult_dynamic_gaussians(args : ModelParams, cameras_extent : float, opt, feat_mode, train_cam, noise):
    muti_dynamic_gaussians = {}
    
    _, track_time, bbox = read_bbox_info(args.source_path)

    train_timestamps = set([cam.timestamp for cam in train_cam])
    print(train_timestamps)

    train_track_cnt = defaultdict(int)
    
    for t in train_timestamps:
        if str(t) not in track_time:
            continue
        for track_id in track_time[str(t)]: 
            train_track_cnt[track_id] += 1
        # train_track_time[str(t)] = track_time[str(t)]

    print(train_track_cnt)
    train_track = set()
    for trackid, cnt in train_track_cnt.items():
        if cnt > 4:
            train_track.add(trackid)

    train_bbox = {}
    train_track_time = {}
    for trackid, v in bbox.items():
        if int(trackid) in train_track:
            b2w = v['B2W']
            train_b2w = {}
            for t in b2w:
                if int(t) in train_timestamps:
                    # train_b2w[t] = b2w[t]
                    train_b2w[t] = b2w[t].clone()
                    if noise:
                        train_b2w[t][[0,2], 3] += (torch.rand(1).item() -0.5) * 2
            v['B2W'] = train_b2w
            train_bbox[trackid] = v

    for t, track_list in track_time.items():
        if int(t) in train_timestamps:
            train_track_time[t] = [track_id for track_id in track_list if track_id in train_track]
    
    train_track = sorted(list(train_track))

    for track_id in train_track:
        vertices = np.array(train_bbox[str(track_id)]["vertices"]) * 1.0
        volume = (vertices[:, 0].max() - vertices[:, 0].min()) * (vertices[:, 1].max() - vertices[:, 1].min()) * (vertices[:, 2].max() - vertices[:, 2].min())
        xyz = np.zeros((int(volume*5000), 3))
        colors = np.random.uniform(low=0, high=1, size=xyz.shape)
        for i in range(3):
            xyz[:, i] = np.random.uniform(low=vertices[:, i].min(), high=vertices[:, i].max(), size=(xyz.shape[0],))
        bound = None
        # if track_id == 81:
        #     bound = 1
        gaussians = GaussianModel(args.sh_degree, feat_mode=feat_mode, bound=None)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        gaussians.create_from_pcd(pcd, cameras_extent)
        gaussians.training_setup(opt)
        if opt is not None:
            gaussians.training_setup(opt)
        muti_dynamic_gaussians[track_id] = gaussians
        
    return muti_dynamic_gaussians, train_track_time, train_bbox, track_time, bbox


def optimize_Rt_intialize(track_B2W, lr=0.0005):
    l = []
    for track_id in track_B2W.keys():
        for t in track_B2W[track_id]['B2W'].keys():
            track_B2W[track_id]['B2W'][t] = torch.nn.Parameter(track_B2W[track_id]['B2W'][t].requires_grad_(True))
            if int(track_id) != 137:
                continue
            parameter = {'params': [track_B2W[track_id]['B2W'][t]], 'lr': lr, 'name': f'track{track_id} time{t}'}
            l.append(parameter)
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    return track_B2W, optimizer

def Rt_list2tensor(track_B2W, device='cuda'):
    for track_id in track_B2W.keys():
        for t in track_B2W[track_id]['B2W'].keys():
            track_B2W[track_id]['B2W'][t] = torch.Tensor(track_B2W[track_id]['B2W'][t]).float().to(device)

    return track_B2W

def Rt_optimized2json(dataset, iteration, track_B2W):
    print("\n[ITER {}] Saving B2W_optimized".format(iteration))
    bbox_path = os.path.join(dataset.model_path, "B2W_optimized/iteration_{}".format(iteration))
    os.makedirs(bbox_path, exist_ok=True)
    with open(os.path.join(dataset.source_path, 'bbox.json')) as json_file:
        bbox_data = json.load(json_file)
        bbox_saved = {}
        bbox_saved['track'] = bbox_data['track']
        bbox_saved['track_time'] = bbox_data['track_time']

    bbox_saved['bbox'] = {}
    for track_id in track_B2W.keys():
        bbox_saved['bbox'][track_id] = {}
        bbox_saved['bbox'][track_id]['vertices'] = track_B2W[track_id]['vertices']
        bbox_saved['bbox'][track_id]['B2W'] = {}
        # if 'phi' in track_B2W[track_id].keys():
        #     bbox_saved['bbox'][track_id]['phi'] = {}
        # if 'phi_tilde' in track_B2W[track_id].keys():
        #     bbox_saved['bbox'][track_id]['phi_tilde'] = {}
        #     bbox_saved['bbox'][track_id]['P_inv'] = {}
        for t in track_B2W[track_id]['B2W'].keys():
            bbox_saved['bbox'][track_id]['B2W'][t] = track_B2W[track_id]['B2W'][t].tolist()
            # if 'phi' in track_B2W[track_id].keys():
            #     bbox_saved['bbox'][track_id]['phi'][t] = track_B2W[track_id]['phi'][t].tolist()
            # if 'phi_tilde' in track_B2W[track_id].keys():
            #     bbox_saved['bbox'][track_id]['phi_tilde'][t] = track_B2W[track_id]['phi_tilde'][t].tolist()
            #     bbox_saved['bbox'][track_id]['P_inv'][t] = track_B2W[track_id]['P_inv'][t].tolist()

    with open(os.path.join(bbox_path, "bbox.json"), 'w') as file:
        json.dump(bbox_saved, file)

def create_track_onehot_id(track):
    track_onehot_id = {}
    instance_track_index = {}
    for i, track_id in enumerate(track):
        onehot_i = torch.zeros((1, 20))
        onehot_i[:, i+1] = 1
        track_onehot_id[track_id] = onehot_i
        instance_track_index[track_id] = i + 1
    return track_onehot_id, instance_track_index

def instance_recode(instance, instance_track_index):
    # remove static cars from instance mask
    instance_track_index_tensor = torch.tensor(list(instance_track_index.keys())) + 1
    instance[~torch.isin(instance, instance_track_index_tensor)] = 0
    for track_id in instance_track_index.keys():
        instance = torch.where(instance == track_id+1, instance_track_index[track_id]*torch.ones_like(instance), instance)
    
    return instance