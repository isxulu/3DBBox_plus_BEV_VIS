import os
import cv2
import torch
import matplotlib
import numpy as np
from tqdm import tqdm
from os import makedirs
from matplotlib import cm
from copy import deepcopy
import matplotlib.pyplot as plt
from os.path import join, exists
from utils.Rt_loss import rt_loss
from argparse import ArgumentParser
from pyquaternion import Quaternion
from utils.semantic_utils import colorize
from matplotlib import patches as mpatches
from matplotlib.ticker import MultipleLocator

from scene import Scene
import utils.plot_utils as pu
import utils.tracking_utils as tu
from utils.dynamic_utils import rot2Euler
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from gaussian_renderer import render, render_with_dynamic
from utils.dynamic_utils import read_bbox_info, unicycle, unicycle2
from arguments import ModelParams, PipelineParams, get_combined_args
from torchmetrics.functional import structural_similarity_index_measure as ssim
from utils.dynamic_utils import create_unicycle_model, create_mult_dynamic_gaussians


def remap_bbox(orig_bbox, map={2: 408, 3: 411, 7: 409, 8: 410, 4: 412, 9: 9, 5: 415, 12: 414, 6: 419, 10: 420}):
    rmap_bbox = {}
    for k, v in orig_bbox.items(): rmap_bbox[str(map[int(k)])] = v
    return rmap_bbox

def unicycle_b2w(track_id, timestamp, unicycle_models, b2w):
    if unicycle_models is not None and str(track_id) in unicycle_models:
        pred_a, pred_b, pred_v, pred_phi, pred_h = unicycle_models[str(track_id)]['model'](timestamp)
        cos, sin = torch.cos(-pred_phi), torch.sin(-pred_phi)
        B2W = torch.eye(4).float().cuda()
        B2W[1, 3], B2W[0, 3], B2W[2, 3] = pred_h, pred_a, pred_b
        B2W[0, 0], B2W[0, 2], B2W[2, 0], B2W[2, 2] = cos, sin, -sin, cos
        return B2W
    else: return b2w

def get_bound_corners(points, device):
    bounds = [torch.min(points, dim=0)[0], torch.max(points, dim=0)[0]]
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = torch.tensor([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ], dtype=torch.float).to(device)
    return corners_3d

def project_bbox(viewpoint_camera, track_B2W: dict, unicycle_models: dict, dynamic_gaussians, kitti360: bool = False):
    # Prepare timestamp, camera extrinsics and intrinsic outside
    timestamp = viewpoint_camera.timestamp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ixt = torch.as_tensor(viewpoint_camera.K[:3, :3]).float().to(device)  # (3, 3)
    w2c = torch.as_tensor(viewpoint_camera.w2c).float()[:3, :].to(device)  # (3, 4)

    visible_cnt, track_bboxs = 0, {}
    for idx, info in track_B2W.items():    
        track_id, b2ws = int(idx), info['B2W']

        # Determine the visibility situation of the current truck
        visible_times, prev_t, next_t = [int(t) for t in b2ws.keys()], -1e8, 1e8
        for t in visible_times:
            if t < timestamp and (timestamp - t < timestamp - prev_t): prev_t = t
            if t > timestamp and (t - timestamp < next_t - timestamp): next_t = t
        prev_b2w, next_b2w = None, None
        if prev_t >= -1e6: prev_b2w = b2ws[str(prev_t)]
        if next_t <   1e6: next_b2w = b2ws[str(next_t)]
        if timestamp in visible_times:
            b2w = b2ws[str(timestamp)]
            visible_cnt += 1
        elif prev_b2w is not None and next_b2w is not None:
            b2w = (prev_b2w + next_b2w) / 2
            visible_cnt += 1
        else: continue

        # Actual projection for the bbox under current frame
        B2W = unicycle_b2w(track_id, timestamp, unicycle_models, b2w)  # (4, 4)
        # Project the bbox of the current track into the image coordinate
        if kitti360: o_xyz = get_bound_corners(dynamic_gaussians[track_id].get_xyz, device)  # (8, 3)
        else: o_xyz = torch.as_tensor(info['vertices'], dtype=torch.float, device=device)  # (8, 3)
        o_xyz = torch.cat([o_xyz, torch.ones_like(o_xyz[:, :1], dtype=torch.float, device=device)], dim=-1)  # (8, 4)
        o_xyz = torch.cat([o_xyz, torch.mean(o_xyz, dim=0, keepdim=True)], dim=0)  # (9, 4)
        # Actual projection
        w_xyz = (B2W @ o_xyz.mT).mT  # (9, 4)
        c_xyz = (w2c @ w_xyz.mT).mT  # (9, 4)
        i_xyz = (ixt @ c_xyz.mT).mT  # (9, 3)
        uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-5)  # (9, 2)

        # Save the projected results
        track_bboxs[str(track_id)] = {'w_xyz': w_xyz, 'c_xyz': c_xyz, 'uv': uv, 'color': info['color']}
    return track_bboxs

def visualize_3d_bbox(image: np.ndarray, bounds: dict, save_path: str = None):
    connections = [
        [1, 3], [1, 4], [3, 6], [4, 6],  # Lower plane parallel to Z=0 plane
        [0, 2], [0, 5], [2, 7], [5, 7],  # Upper plane parallel to Z=0 plane
        [0, 1], [2, 3], [4, 5], [6, 7],  # Connections between upper and lower planes
    ]
    for _, v in bounds.items():
        for connection in connections:
            corners = v['uv'].detach().cpu().numpy().astype(np.int32)  # (8, 2)
            cv2.line(image, corners[connection[0]], corners[connection[1]], color=v['color'], thickness=2)
    cv2.imwrite(save_path, (image[..., [2, 1, 0]] * 255.).astype(np.uint8))

def draw_bev_canvas(ax, x_min=-55, x_max=55, y_min=0, y_max=100, interval=10):
    # Set x, y limit and mark border
    ax.set_facecolor('white')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.tick_params(axis='both', labelbottom=False, labelleft=False)
    ax.xaxis.set_minor_locator(MultipleLocator(interval))
    ax.yaxis.set_minor_locator(MultipleLocator(interval))

    # for radius in range(y_max, -1, -interval):
    #     # Mark all around sector
    #     ax.add_patch(mpatches.Wedge(center=[0, 0], alpha=0.1, aa=True, r=radius, theta1=-180, theta2=180, fc="black"))
    #     # Mark range
    #     if radius / np.sqrt(2) + 8 < x_max: ax.text(radius / np.sqrt(2) + 3, radius / np.sqrt(2) - 5, f'{radius}m', rotation=-45, color='darkblue', fontsize='xx-large')

    # Mark visible sector
    # ax.add_patch(mpatches.Wedge(center=[0, 0], alpha=0.1, aa=True, r=y_max, theta1=45, theta2=135, fc="cyan"))
    # # Mark ego-vehicle
    # ax.arrow(0, 0, 0, 3, color='black', width=0.5, overhang=0.3)
    return ax

def orthogonalize_rotation_matrix(rotation_matrix):
    U, _, Vt = np.linalg.svd(rotation_matrix)
    corrected_rotation_matrix = np.dot(U, Vt)
    
    # 使用特殊正交群的投影操作
    det = np.linalg.det(corrected_rotation_matrix)
    reflection_matrix = np.eye(3)
    reflection_matrix[2, 2] = det
    orthogonal_rotation_matrix = np.dot(corrected_rotation_matrix, reflection_matrix)

    return orthogonal_rotation_matrix

def visualize_bev(view, track_B2W_o, track_B2W_g, unicycles, bounds_o: dict, bounds_g: dict,
                  save_path: str = None, kitti360=False, fig_size=10, dpi=100):
    # Create the canvas
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax = draw_bev_canvas(ax)

    # Draw unicycle optimized bbox
    for k, v in bounds_o.items():
        if str(view.timestamp) not in track_B2W_g[k]['B2W'].keys(): continue

        c_center = v['w_xyz'][-1, [0, 2]].detach().cpu().numpy()  # (2,)
        quat_cam_rot_t = Quaternion(matrix=orthogonalize_rotation_matrix(view.R).T)  # (4,)
        yaw_hist_pd = []
        if k in unicycles.keys(): _, _, _, quat_yaw_world_pd, _ = unicycles[k]['model'](view.timestamp)
        else: quat_yaw_world_pd = rot2Euler(track_B2W_o[k]['B2W'][str(view.timestamp)][:3, :3])[1]
        quat_yaw_world_pd = -quat_yaw_world_pd.detach().cpu().numpy()
        rotation_cam = quat_cam_rot_t * quat_yaw_world_pd
        vtrans = np.dot(rotation_cam.rotation_matrix, np.array([1, 0, 0]))
        yaw_hist_pd.append(-np.arctan2(vtrans[2], vtrans[0]).tolist())
        yaw_hist_pd = np.vstack(yaw_hist_pd)

        if kitti360: L_ratio, _, W_ratio = np.linalg.norm(track_B2W_g[k]['B2W'][str(view.timestamp)][:3, :3].cpu().numpy(), axis=0)
        else: L_ratio, _, W_ratio = 1.0, 1.0, 1.0
        L, _, W = (np.array(track_B2W_g[k]['vertices']).max(axis=0) - np.array(track_B2W_g[k]['vertices']).min(axis=0))
        L, W = L * L_ratio, W * W_ratio
        pu.plot_bev_obj(ax, c_center, [c_center], quat_yaw_world_pd, yaw_hist_pd, L, W, v['color'], 'PD', line_width=2)

    # Draw GT bbox
    for k, v in bounds_g.items():
        if str(view.timestamp) not in track_B2W_g[k]['B2W'].keys(): continue

        c_center = v['c_xyz'][-1, [0, 2]].detach().cpu().numpy()  # (2,)
        quat_cam_rot_t = Quaternion(matrix=orthogonalize_rotation_matrix(view.R))  # (4,)
        yaw_hist_pd = []
        quat_yaw_world_pd = -rot2Euler(track_B2W_g[k]['B2W'][str(view.timestamp)][:3, :3])[1].detach().cpu().numpy()
        rotation_cam = quat_cam_rot_t * quat_yaw_world_pd
        vtrans = np.dot(rotation_cam.rotation_matrix, np.array([1, 0, 0]))
        yaw_hist_pd.append(-np.arctan2(vtrans[2], vtrans[0]).tolist())
        yaw_hist_pd = np.vstack(yaw_hist_pd)

        if kitti360: L_ratio, _, W_ratio = np.linalg.norm(track_B2W_g[k]['B2W'][str(view.timestamp)][:3, :3].cpu().numpy(), axis=0)
        else: L_ratio, _, W_ratio = 1.0, 1.0, 1.0
        L, _, W = (np.array(track_B2W_g[k]['vertices']).max(axis=0) - np.array(track_B2W_g[k]['vertices']).min(axis=0))
        L, W = L * L_ratio, W * W_ratio
        pu.plot_bev_obj(ax, c_center, [c_center], quat_yaw_world_pd, yaw_hist_pd, L, W, v['color'], 'PD', line_width=1)

    cv2.imwrite(save_path, pu.fig2data(fig))
    plt.close()

def visualize_bev_trace(view, track_B2W_o, track_B2W_g, unicycles, bounds_o: dict, bounds_g: dict, traces: dict,
                        save_path: str = None, kitti360=False, fig_size=10, dpi=100):
    # Create the canvas
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax = draw_bev_canvas(ax)

    # Draw unicycle optimized bbox
    for k, v in bounds_o.items():
        if str(view.timestamp) not in track_B2W_g[k]['B2W'].keys(): continue

        c_center = v['w_xyz'][-1, [0, 2]].detach().cpu().numpy()  # (2,)
        quat_cam_rot_t = Quaternion(matrix=orthogonalize_rotation_matrix(view.R).T)  # (4,)
        yaw_hist_pd = []
        if k in unicycles.keys(): _, _, _, quat_yaw_world_pd, _ = unicycles[k]['model'](view.timestamp)
        else: quat_yaw_world_pd = rot2Euler(track_B2W_o[k]['B2W'][str(view.timestamp)][:3, :3])[1]
        quat_yaw_world_pd = -quat_yaw_world_pd.detach().cpu().numpy()
        rotation_cam = quat_cam_rot_t * quat_yaw_world_pd
        vtrans = np.dot(rotation_cam.rotation_matrix, np.array([1, 0, 0]))
        yaw_hist_pd.append(-np.arctan2(vtrans[2], vtrans[0]).tolist())
        yaw_hist_pd = np.vstack(yaw_hist_pd)

        if kitti360: L_ratio, _, W_ratio = np.linalg.norm(track_B2W_g[k]['B2W'][str(view.timestamp)][:3, :3].cpu().numpy(), axis=0)
        else: L_ratio, _, W_ratio = 1.0, 1.0, 1.0
        L, _, W = (np.array(track_B2W_g[k]['vertices']).max(axis=0) - np.array(track_B2W_g[k]['vertices']).min(axis=0))
        L, W = L * L_ratio, W * W_ratio
        pu.plot_bev_obj(ax, c_center, [c_center], quat_yaw_world_pd, yaw_hist_pd, L, W, v['color'], 'PD', line_width=2)

        if k not in traces['o'].keys(): traces['o'][k] = []
        for i in range(len(traces['o'][k])): pu.plot_bev_center(ax, traces['o'][k][i][0], [traces['o'][k][i][0]], traces['o'][k][i][1], traces['o'][k][i][2], L, W, v['color'], 'PD', line_width=1.0)
        traces['o'][k].append([c_center, quat_yaw_world_pd, yaw_hist_pd])

    # Draw GT bbox
    for k, v in bounds_g.items():
        if str(view.timestamp) not in track_B2W_g[k]['B2W'].keys(): continue

        c_center = v['c_xyz'][-1, [0, 2]].detach().cpu().numpy()  # (2,)
        quat_cam_rot_t = Quaternion(matrix=orthogonalize_rotation_matrix(view.R))  # (4,)
        yaw_hist_pd = []
        quat_yaw_world_pd = -rot2Euler(track_B2W_g[k]['B2W'][str(view.timestamp)][:3, :3])[1].detach().cpu().numpy()
        rotation_cam = quat_cam_rot_t * quat_yaw_world_pd
        vtrans = np.dot(rotation_cam.rotation_matrix, np.array([1, 0, 0]))
        yaw_hist_pd.append(-np.arctan2(vtrans[2], vtrans[0]).tolist())
        yaw_hist_pd = np.vstack(yaw_hist_pd)

        if kitti360: L_ratio, _, W_ratio = np.linalg.norm(track_B2W_g[k]['B2W'][str(view.timestamp)][:3, :3].cpu().numpy(), axis=0)
        else: L_ratio, _, W_ratio = 1.0, 1.0, 1.0
        L, _, W = (np.array(track_B2W_g[k]['vertices']).max(axis=0) - np.array(track_B2W_g[k]['vertices']).min(axis=0))
        L, W = L * L_ratio, W * W_ratio
        pu.plot_bev_obj(ax, c_center, [c_center], quat_yaw_world_pd, yaw_hist_pd, L, W, v['color'], 'PD', line_width=1)

        if k not in traces['g'].keys(): traces['g'][k] = []
        for i in range(len(traces['g'][k])):
            o_center = (view.w2c[:3, :] @ traces['g'][k][i].T).T[-1, [0, 2]]
            pu.plot_bev_center(ax, o_center, [o_center], quat_yaw_world_pd, yaw_hist_pd, L, W, v['color'], 'PD', line_width=0.4)
        traces['g'][k].append(v['w_xyz'].detach().cpu().numpy())

    cv2.imwrite(save_path, pu.fig2data(fig))
    plt.close()
    return traces

def visualize_set(model_path, name, iteration, views_o, views_g, gaussians, pipeline, background, multi_dynamic_gaussians=None,
                  track_time=None, track_B2W_o=None, track_B2W_g=None, unicycles=None, optical=False, kitti360=False):
    # Paths to save the two types of the visualizations
    img_path = join(model_path, name, f'ours_{iteration}/imgs')
    bev_path = join(model_path, name, f'ours_{iteration}/bevs')
    bev_trace_path = join(model_path, name, f'ours_{iteration}/bev_traces')
    makedirs(img_path, exist_ok=True)
    makedirs(bev_path, exist_ok=True)
    makedirs(bev_trace_path, exist_ok=True)

    traces = {'o': {}, 'g': {}}
    for idx, (view_o, view_g) in enumerate(tqdm(zip(views_o, views_g), desc="Rendering progress")):
        # Perform forward to render the result
        if multi_dynamic_gaussians is None: render_pkg = render(view_o, gaussians, pipeline, background)
        else: render_pkg = render_with_dynamic(view_o, gaussians, multi_dynamic_gaussians, track_B2W_o, unicycles, pipeline, background, optical)
        # Fetch rendering result out
        render = render_pkg['render'].detach().cpu().permute(1, 2, 0).numpy()
        render = view_o.image.detach().cpu().permute(1, 2, 0).numpy()
        # Project the unicycle optimized bounds
        bounds_o = project_bbox(view_o, track_B2W_o, unicycles, multi_dynamic_gaussians, kitti360)
        # Project the ground truth bounds
        bounds_g = project_bbox(view_g, track_B2W_g, None, multi_dynamic_gaussians, False)

        # Visualize the projected bbox
        visualize_3d_bbox(render, bounds_g, save_path=join(img_path, view_o.image_name + '.png'))
        visualize_bev(view_o, track_B2W_o, track_B2W_g, unicycles, bounds_o, bounds_g, save_path=join(bev_path, view_o.image_name + '.png'), kitti360=kitti360)
        traces = visualize_bev_trace(view_o, track_B2W_o, track_B2W_g, unicycles, bounds_o, bounds_g, traces, save_path=join(bev_trace_path, view_o.image_name + '.png'), kitti360=kitti360)

def visualize_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams,
                   gt_path: str, skip_train : bool, skip_test : bool, dynamic : bool, optical : bool,
                   is_remap: bool, kitti360: bool):
    with torch.no_grad():
        # Create the Gaussian model and the scene
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Create the ground truth scene for comparison in BEV visualization
        dataset_t = deepcopy(dataset); dataset_t.source_path = gt_path
        gaussians_t = GaussianModel(dataset_t.sh_degree)
        scene_t = Scene(dataset_t, gaussians_t, shuffle=False)

        # Create the background filling
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dynamic:
            all_track, all_track_time, all_bbox = read_bbox_info(dataset.source_path)
            # Load ground truth tracks and its corresponding bboxes
            _, _, all_bbox_t_woremap = read_bbox_info(dataset_t.source_path)
            if is_remap: all_bbox_t = remap_bbox(all_bbox_t_woremap)
            else: all_bbox_t = all_bbox_t_woremap

            multi_dynamic_gaussians, unicycles, track, track_time = {}, {}, set(), {}
            for track_id in all_track:
                # Load pretrained dynamic Gaussian Model
                gaussian_path = join(dataset.model_path, f"ckpts/track_id_{track_id}_chkpnt{iteration}.pth")
                if exists(gaussian_path):
                    multi_dynamic_gaussians[track_id] = GaussianModel(dataset.sh_degree)
                    multi_dynamic_gaussians[track_id].restore(torch.load(gaussian_path)[0], None)
                    track.add(track_id)

                # Load pretrained unicycle model for every truck
                unicycle_path = join(dataset.model_path, f"ckpts/unicycle_{track_id}_chkpnt{iteration}.pth")
                if exists(unicycle_path):
                    unicycle_model = unicycle2(torch.randn(10), torch.randn(10,2), torch.randn(10), torch.randn(10))
                    unicycle_model.restore(torch.load(unicycle_path))
                    unicycles[str(track_id)] = {"model": unicycle_model}
                # unicycles[str(track_id)] = {"model": unicycle_models[str(track_id)]['model']}

            for t, track_list in all_track_time.items():
                track_time[t] = [track_id for track_id in track_list if track_id in track]
            track = sorted(list(track))

            # Add a specific color for each track
            for i, k in enumerate(all_bbox.keys()): all_bbox[k]['color'] = matplotlib.colors.to_rgb(matplotlib.colormaps['tab20'](i))
            for i, k in enumerate(all_bbox_t.keys()): all_bbox_t[k]['color'] = matplotlib.colors.to_rgb(matplotlib.colormaps['tab20'](i))
        else:
            multi_dynamic_gaussians, unicycles, track, track_time, all_bbox, all_bbox_t = None, None, None, None, None, None

        # Visualize train views
        if not skip_train:
            visualize_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene_t.getTrainCameras(), gaussians, pipeline,
                          background, multi_dynamic_gaussians, track_time, all_bbox, all_bbox_t, unicycles, optical, kitti360)

        # Visualize test views
        if not skip_test:
            visualize_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene_t.getTestCameras(), gaussians, pipeline,
                          background, multi_dynamic_gaussians, track_time, all_bbox, all_bbox_t, unicycles, optical, kitti360)
        
        # compute pose 's loss
        fit_bbox = {}
        for track_id, track_B2W in all_bbox.items():
            for t, value in track_B2W['B2W'].items():

                B2W = unicycle_b2w(track_id, int(t), unicycles, value)
                try: 
                    fit_bbox[str(track_id)]['B2W'][str(t)] = B2W
                except:
                    fit_bbox[str(track_id)] = {}
                    fit_bbox[str(track_id)]['B2W'] = {}
                    fit_bbox[str(track_id)]['B2W'][str(t)] = B2W
        # print(fit_bbox)
        loss_R, loss_t = rt_loss(fit_bbox, all_bbox_t, skip_train, skip_test)
        print(loss_R, loss_t)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--dynamic", action='store_true', default=False)
    parser.add_argument("--optical", action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gt_path", type=str, default='/data3/xulu/data/gt_65_f56')
    parser.add_argument("--is_remap", action='store_true', default=False)
    parser.add_argument("--kitti360", action='store_true', default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    visualize_sets(model.extract(args), args.iteration, pipeline.extract(args), args.gt_path,
                   args.skip_train, args.skip_test, args.dynamic, args.optical, args.is_remap,
                   args.kitti360)
