
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import cv2

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
import PIL.Image
import pyspng
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
)
import trimesh

opt = tyro.cli(AllConfigs)

def _file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def rotatedx(original_vertices, angle=90):

    vertices = original_vertices


    theta = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])


    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def rotatedz(original_vertices, angle=90):

    vertices = original_vertices


    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    rotated_vertices = np.dot(rotation_matrix, vertices.T).T

    return rotated_vertices

def prepare_default_rays(input_size, view_num=4, device='cpu', elevation=0):
    
    from kiui.cam import orbit_camera
    from core.utils import get_rays
    
    cam_poses = np.stack([
        orbit_camera(elevation, 0, radius=opt.cam_radius),
        orbit_camera(elevation, 180, radius=opt.cam_radius),
        orbit_camera(elevation, 90, radius=opt.cam_radius),
        orbit_camera(elevation, 270, radius=opt.cam_radius),
    ], axis=0) # [4, 4, 4]

    cam_poses = cam_poses[0:view_num]
    
    cam_poses = torch.from_numpy(cam_poses)

    rays_embeddings = []
    for i in range(cam_poses.shape[0]):
        rays_o, rays_d = get_rays(cam_poses[i], input_size, input_size, opt.fovy) # [h, w, 3]
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        rays_embeddings.append(rays_plucker)


    rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
    
    return rays_embeddings


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws



def ood_prepare_obj(image_path, smpl_path, idx):

    images = []
    masks = []
    vids = [3]
    for vid in vids:

        # Normal Correspond RGB for Mask
        # image_path_rgb = os.path.join(image_path, f'{vid:03d}_rgb.png')
        image_path_rgb = os.path.join(image_path)

        try:
            with open(image_path_rgb, 'rb') as f:
                if _file_ext(image_path_rgb) == '.png':
                    ori_img_rgb = pyspng.load(f.read())
                else:
                    ori_img_rgb = np.array(PIL.Image.open(f))
        except:
            print(image_path_rgb)

        image_rgb = torch.from_numpy(ori_img_rgb.astype(np.float32) / 255) # [512, 512, 4] in [0, 1]

        image_rgb = image_rgb.permute(2, 0, 1) # [4, 512, 512]

        mask_rgb = image_rgb[3:4] # [1, 512, 512]
        image_rgb = image_rgb[:3] * mask_rgb + (1 - mask_rgb) # [3, 512, 512], to white bg

    
        
        images.append(image_rgb)
        masks.append(mask_rgb)



    images = torch.stack(images, dim=0) # [V, C, H, W]
    masks = torch.stack(masks, dim=0) # [V, H, W]

    images_input = F.interpolate(images[:1].clone(), size=(opt.input_size//2, opt.input_size//2), mode='bilinear', align_corners=False) # [V, C, H, W]
    masks_input = F.interpolate(masks[:1].clone(), size=(opt.input_size//2, opt.input_size//2), mode='bilinear', align_corners=False) # [V, C, H, W]

    images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = prepare_default_rays(opt.input_size//2, 1)
    final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]


    images_input = F.interpolate(images[:1].clone(), size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]

    smpl_ori = trimesh.load(smpl_path) # 保存为OBJ文件 mesh.export('output.obj') 

    smpl_ori.vertices = rotatedx(smpl_ori.vertices)
    smpl_ori.vertices = rotatedz(smpl_ori.vertices)



    def new_mesh(mesh_ori, ratio):

        try:
            bb_max = mesh_ori.vertices.max(axis=0)
            bb_min = mesh_ori.vertices.min(axis=0)
        except:
            mesh_ori = mesh_ori.geometry[[i for i in mesh_ori.geometry][0]]
            bb_max = mesh_ori.vertices.max(axis=0)
            bb_min = mesh_ori.vertices.min(axis=0)
        centers = (
            (bb_min[0] + bb_max[0]) / 2,
            (bb_min[1] + bb_max[1]) / 2,
            (bb_min[2] + bb_max[2]) / 2
        )
        total_size = (bb_max - bb_min).max()
        scale = total_size / (0.5 + ratio)
        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        scales_inv = (
            2/scale, 2/scale, 2/scale
        )
        mesh_ori.vertices = mesh_ori.vertices+translation
        mesh_ori.vertices = mesh_ori.vertices*scales_inv

        return mesh_ori, translation, scales_inv

    ratio = 0
    smpl_ori, translation, scales_inv = new_mesh(smpl_ori, ratio)

    
    first = 0

    azimuths = np.concatenate([np.array([first, (first+180)%360, (first+90)%360, (first+270)%360])], axis=0)
    elevations = np.concatenate([np.array([0, 0, 0, 0])], axis=0)

    azimuths = np.array(azimuths).astype(float)
    elevations = np.array(elevations).astype(float)

    radius = [1.5 for i in range(1 + 3)]
    radius = np.array(radius).astype(float)

    poses = []
    for i in range(1 + 3):
        pose = orbit_camera(elevations[i], azimuths[i], radius[i])
        poses.append(pose)

    poses = np.array(poses).astype(float)

    c2ws = spherical_camera_pose(azimuths, elevations, radius)

    data = {
        "input":final_input,
        "images_input":images_input,

        "smpl_ori_v":torch.from_numpy(smpl_ori.vertices).float(),
        "smpl_ori_f":torch.from_numpy(smpl_ori.faces).int(),
        'input_poses': poses[:1],               # (6, 4, 4)
        'target_poses': poses[1:],               # (6, 4, 4)
        'input_c2ws': c2ws[:1],               # (6, 4, 4)
        'target_c2ws': c2ws[1:],              # (V, 4, 4)

    }

    return data



IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# model
model = LGM(opt)
model.extra()

# resume
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    
    attn_names = []
    attn_shapes = []
    for key in ckpt:
        if 'attn.qkv.weight' in key:
            attn_names.append(key)
            attn_shapes.append(ckpt[key].shape)

    state_dict = model.state_dict()
    for k, v in ckpt.items():


        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                # accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
                pass
        else:
            # accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
            pass


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1



args_input_path = opt.infer_img_path
extra_test_list_front_image = [ os.path.join(args_input_path, f) for f in sorted(os.listdir(args_input_path)) if f.endswith(('png', 'jpg'))]

folder_path = opt.infer_smpl_path
extra_test_list_smpl =  [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('obj', 'ply'))]
extra_test_list_smpl = sorted(extra_test_list_smpl)



epoch = 'infer'
for idx in range(len(extra_test_list_front_image)):
    data = ood_prepare_obj(extra_test_list_front_image[idx], extra_test_list_smpl[idx], idx)
    for d in data:
        if isinstance(data[d], np.ndarray):
            data[d] = torch.from_numpy(data[d]).to(device).unsqueeze(0)#
        else:
            data[d] = data[d].to(device).unsqueeze(0)#

    i = str(idx)
    with torch.no_grad():
        out = model.forward_ood(data)

    images_input = data['images_input'].float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    images_input = images_input.transpose(0, 3, 1, 4, 2).reshape(-1, images_input.shape[1] * images_input.shape[3], 3) # [B*output_size, V*output_size, 3]
    kiui.write_image(f'{opt.workspace}/ood/ood_input_images_{epoch}_{i}.jpg', images_input)

    images_input = data['images_input_normals_smpl'].float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    images_input = images_input.transpose(0, 3, 1, 4, 2).reshape(-1, images_input.shape[1] * images_input.shape[3], 3) # [B*output_size, V*output_size, 3]
    kiui.write_image(f'{opt.workspace}/ood/ood_input_images_{epoch}_{i}_normals_smpl.jpg', images_input)

    # render 360 video 
    images = []
    elevation = 0

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1


    azimuth = np.arange(0, 360, 2, dtype=np.int32)
    elevation_offset = 0
    azi_offset = 0
    radius_offset = 0
    for azi in azimuth:
        
        cam_poses = torch.from_numpy(orbit_camera(elevation + elevation_offset, azi+azi_offset, radius=opt.cam_radius+radius_offset, opengl=True)).unsqueeze(0).to('cuda')

        cam_poses[:, :3, 3] *=  1.5 / opt.cam_radius # 1.5 is the default scale

        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        scale = min(azi / 360, 1)

        image = model.gs.render(out['gaussians'], cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']

        images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().detach().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(os.path.join(opt.workspace, f'ood/render_pred_{epoch}_{i}' + '.mp4'), images, fps=30)
    model.gs.save_ply(out['gaussians'], os.path.join(opt.workspace, f'ood/gaussian_pred_{epoch}_{i}' + '.ply'))




