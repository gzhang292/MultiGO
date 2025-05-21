import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet, MVAttention
from core.options import Options
from core.gs import GaussianRenderer
import cv2
import copy

import torchvision
import random
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.geometry.camera.perspective_camera import PerspectiveCamera
from src.models.geometry.render.neural_render import NeuralRender

import torchvision.transforms.functional as TF
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
import pickle
import os
import tinyobjloader

from kaolin.ops.mesh import check_sign
from kaolin import _C

import copy

from Models.autoencoders.michelangelo_autoencoder import get_embedder

import math

from torch.nn.init import trunc_normal_
import torch.nn.functional as tfunc

import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds



def calc_face_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1) 
    return face_normals #F,3

def calc_vertex_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        face_normals:torch.Tensor=None, #F,3, not normalized
        )->torch.Tensor: #F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)


class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # gs renderer
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy), dtype=np.float32)
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1


    def extra(self):
        #--------------input-----------------
        embed_type = 'fourier'
        num_freqs = 8
        include_pi = False

        self.embedder = get_embedder(embed_type=embed_type, num_freqs=num_freqs, include_pi=include_pi)

        self.conv_in_shape = nn.Conv2d(54+6, 64, kernel_size=3, stride=1, padding=1)


    def prepare_render_images_infer(self, batch):
        camera = PerspectiveCamera(fovy=self.opt.fovy, device=batch['input'].device)
        renderer = NeuralRender(device=batch['input'].device, camera_model=camera)


        c2ws = torch.cat([batch['input_c2ws'],batch['target_c2ws']],dim=1)

        batch_size = c2ws.size(0)

        normal_list_smpl = []
        alpha_list_smpl = []

        concat3d2d_list = []

        bg_white = torch.ones(3, dtype=torch.float32, device=c2ws.device)

        bg_white_3d2d = torch.ones(54, dtype=torch.float32, device=c2ws.device)


        threeD_feat = self.forward_shape_(batch)


        for idx in range(batch_size):

            camera_mv_bx4x4 = torch.linalg.inv(c2ws[idx])

            smpl_v_nx3 = batch['smpl_ori_v'][idx]
            smpl_f_fx3 = batch['smpl_ori_f'][idx]

            out_smpl = renderer.render_mesh(
                        smpl_v_nx3.unsqueeze(dim=0),
                        smpl_f_fx3,
                        camera_mv_bx4x4,
                        smpl_v_nx3.unsqueeze(dim=0),
                        resolution=512,
                        device=batch['input_c2ws'].device,
                        hierarchical_mask=False,
                        uv=None,
                        uv_idx=None,
                        tex=None,
                        threeD_feat=threeD_feat[idx],
                    )

            _, _, hard_mask_smpl, _, _, _, _, normal_smpl, concat3d2d = out_smpl

            alpha_smpl = hard_mask_smpl
            normal_smpl = normal_smpl * alpha_smpl + bg_white * (1-alpha_smpl)
            concat3d2d = concat3d2d * alpha_smpl + bg_white_3d2d * (1-alpha_smpl)

            normal_smpl = normal_smpl.permute(0, 3, 1, 2).contiguous().float()
            alpha_smpl = alpha_smpl.permute(0, 3, 1, 2).contiguous().float()
            concat3d2d = concat3d2d.permute(0, 3, 1, 2).contiguous().float()
            
            
            normal_list_smpl.append(normal_smpl)
            alpha_list_smpl.append(alpha_smpl)
            concat3d2d_list.append(concat3d2d)


        normals_smpl = torch.stack(normal_list_smpl, dim=0).float()
        alphas_smpl = torch.stack(alpha_list_smpl, dim=0).float()
        concat3d2ds = torch.stack(concat3d2d_list, dim=0).float()

        batch['input_smpl_normals'] = normals_smpl[:,:4]
        batch['input_smpl_alphas'] = alphas_smpl[:,:4]

        batch['input_concat3d2ds'] = concat3d2ds
        
        return batch


    def for_render_gs_obj_infer(self, data):
        

        c2ws = torch.cat([data['input_c2ws'],data['target_c2ws']],dim=1).float()
        batch_size = c2ws.size(0)

        results = {}
        results['cam_view'] = []
        results['cam_view_proj'] = []
        results['cam_pos'] = []
        results['c2ws'] = []
        results['images_input_normals_smpl'] = []
        results['input_normals_smpl'] = []


        cam_radius = 1.5
        for idx in range(batch_size):

            cam_poses = c2ws[idx]

            # normalized camera feats as in paper (transform the first pose to a fixed position)
            transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, cam_radius], [0, 0, 0, 1]], dtype=torch.float32).cuda() @ torch.inverse(cam_poses[0])
            cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

            cam_poses[:, :3, 3] *=  1.5 / cam_radius # 1.5 is the default scale


            cam_poses_input_smpl = cam_poses[:4].clone()

            images_input_normals_smpl = F.interpolate(data['input_smpl_normals'][idx][:4].clone(), size=(self.opt.input_size//2, self.opt.input_size//2), mode='bilinear', align_corners=False) # [V, C, H, W]
            results['images_input_normals_smpl'].append(images_input_normals_smpl)
            images_input_normals_smpl = TF.normalize(images_input_normals_smpl, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

            # build rays for input views
            rays_embeddings_half_size = []
            for i in range(4):
                rays_o, rays_d = get_rays(cam_poses_input_smpl[i], self.opt.input_size//2, self.opt.input_size//2, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings_half_size.append(rays_plucker)

            rays_embeddings_half_size = torch.stack(rays_embeddings_half_size, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input_normals_smpl = torch.cat([images_input_normals_smpl, rays_embeddings_half_size], dim=1) # [V=4, 9, H, W]
            results['input_normals_smpl'].append(final_input_normals_smpl)

            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses.float()).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ self.proj_matrix.to(device=data['input_c2ws'].device)#.double() # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            results['cam_view'].append(cam_view)
            results['cam_view_proj'].append(cam_view_proj)
            results['cam_pos'].append(cam_pos)
            results['c2ws'].append(cam_poses)

        data['cam_view'] = torch.stack(results['cam_view'])
        data['cam_view_proj'] = torch.stack(results['cam_view_proj'])
        data['cam_pos'] = torch.stack(results['cam_pos'])
        data['c2ws'] = torch.stack(results['c2ws'])

        data['images_input_normals_smpl'] = torch.stack(results['images_input_normals_smpl'])
        data['input_normals_smpl'] = torch.stack(results['input_normals_smpl'])

        return data


    def forward_shape_(self, data):

        B, _, _ = data['smpl_ori_v'].shape
        normals = []
        for i in range(B):
            normal = calc_vertex_normals(data['smpl_ori_v'][i], data['smpl_ori_f'][i].long())
            normals.append(normal)
        normals = torch.stack(normals)

        fourier = self.embedder(data['smpl_ori_v'])

        shape_latents = torch.cat([fourier, normals], dim=-1)

        return shape_latents

    def forward_gaussians_(self, data):

        images = data['input']

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        shape_latents_other = torch.cat([data['input_concat3d2ds'][:,1:4], data['input_normals_smpl'][:,1:,3:]],dim=2)

        b, v, c, h, w = shape_latents_other.shape
        shape_latents_other = shape_latents_other.reshape(b*v, c, h, w)
        shape_latents_other = self.conv_in_shape(shape_latents_other)
        shape_latents_other = shape_latents_other.view(b, v, 64, h, w)


        x_mid = self.unet(images, True, shape=shape_latents_other) # [B*4, 14, h, w]
        x_mid = self.unet.conv_out(x_mid)
        x = self.conv(x_mid)

        x = x.view( -1, 14, self.opt.splat_size//2, self.opt.splat_size//2)
        x = x.reshape(B, 4, 14, self.opt.splat_size//2, self.opt.splat_size//2)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)


        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]   
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]

        return gaussians


    def forward_ood(self, data):
        results = {}
        
        data = self.prepare_render_images_infer(data)
        data = self.for_render_gs_obj_infer(data)

        gaussians = self.forward_gaussians_(data)

        results['gaussians'] = gaussians

        return results
