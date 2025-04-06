import sys
import pickle
import torch
import glob
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.transforms import matrix_to_quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import Delaunay
# import pyrender
# import trimesh
import imageio
# import open3d as o3d
import time
import os

# from FLAME_PyTorch.flame_pytorch import FLAME, get_config
from src.JunLi_2D import DeformNet2D, FLAMEToFeature, FLAMEToFeature3D
from src.UNet3D.UNet import UNet2D
from src.cUNet.cunet import Conditional_UNet, ExpandDimNet

# 在根目录下的那个 UNet3D
sys.path.append('./')
from UNet3D_only_3D import UNet3D



from flame import FLAME_mica, parse_args
from utils.general_utils import Pytorch3dRasterizer, \
                                Embedder, \
                                load_binary_pickle, \
                                a_in_b_torch, \
                                face_vertices_gen, \
                                get_expon_lr_func

# from YiHao_utils import WassersteinGaussian

SAVE_POINT_IMAGE = False # 查看点云和 mesh，保存图像，训练时False
COARSE_INPUT = True # 使用 coarse input 取代 canonical input

Unet_Concat = False # 使用 concat 的 FLAME 2D 网络取代 MLP 网络当做 deform net
UNet_AdaIN = False # 使用 AdaIN 形式的 Unet 网络取代 MLP 网络当做 deform net
# 这两个 Unet 不能同时为 True

Unet_3D = False # 和 Unet_Concat 一起使用，使用 3D Unet 网络
MLP_Deform = True # 使用 MLP 网络当做 deform net

INITIAL_ROT_ALONG_FACE = True # 是否yihao初始化旋转


# config = get_config()
# radian = np.pi / 180.0
# flamelayer = FLAME(config)

def visualize_flame(flame_model, geometry, output_dir='./output', device='cuda'):
    """
    Visualizes FLAME outputs and saves the visualization as an image.

    Args:
        flamelayer (FLAME layer): The FLAME model layer.
        geometry (torch.Tensor): The geometry output from the FLAME model.
        faces (numpy.ndarray): The face indices from the FLAME model.
        landmarks (torch.Tensor): The landmarks output from the FLAME model.
        output_dir (str): The directory where the visualization images will be saved.
        device (str): The device to run the visualization on.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(geometry.shape[0]):
        vertices = geometry[i].detach().cpu().numpy().squeeze()
        faces = flame_model.faces.to(device).detach().cpu().numpy()

        # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        # Create a mesh object
        # tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        # sm = trimesh.creation.uv_sphere(radius=0.005)
        # sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        # scene = trimesh.Scene([tri_mesh])
        # scene.export(file_obj=f'./output/model_of_deform_flame.ply', file_type='ply')


        # Create a visualization of the mesh in a 3D plot, saved as an image
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = vertices.T
        ax.scatter(x, y, z)
        ax.plot_trisurf(x, y, z, triangles=faces, linewidth=0.2, antialiased=True, color='grey', shade=True)
        ax.view_init(elev=90, azim=-90)

        x_center, y_center, z_center = np.mean(vertices, axis=0)
        max_span = np.max(np.ptp(vertices, axis=0)) / 2
        ax.set_xlim(x_center - max_span, x_center + max_span)
        ax.set_ylim(y_center - max_span, y_center + max_span)
        ax.set_zlim(z_center - max_span, z_center + max_span)
        ax.set_axis_off()

        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        plt.savefig(os.path.join(output_dir, f'model_visualization_{time_stamp}.png'), bbox_inches='tight', pad_inches=0)
        # print(f"Image saved to: {os.path.join(output_dir, f'model_visualization_{time_stamp}.png')}")
        
        plt.close(fig)


def visualize_verts_final(verts_final, output_dir='./output', color='grey', point_size=0.1):
    """
    Visualizes the final vertices and saves the visualization as an image.

    Args:
        verts_final (torch.Tensor): The final vertices tensor with shape [1, N, 3].
        output_dir (str): The directory where the visualization images will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Remove batch dimension and convert to numpy
    verts_final_np = verts_final.squeeze().detach().cpu().numpy()

    # Create a new figure and a 3D subplot
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of verts_final points
    ax.scatter(verts_final_np[:, 0], verts_final_np[:, 1], verts_final_np[:, 2], color=color, s=point_size)

    # Set the view angle
    ax.view_init(elev=90, azim=-90)

    # Set axis scales to be equal
    max_range = np.max(np.ptp(verts_final_np, axis=0)) / 2.0
    mid_x = np.mean(verts_final_np[:, 0])
    mid_y = np.mean(verts_final_np[:, 1])
    mid_z = np.mean(verts_final_np[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Hide the axes
    ax.set_axis_off()

    # Save the image with a timestamp
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_path = os.path.join(output_dir, f'verts_final_visualization_{time_stamp}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # print(f"Image saved to: {output_path}")
    
    plt.close(fig)

# Example usage
# visualize_verts_final(verts_final_tensor)


class Deform_Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        mica_flame_config = parse_args()
        self.flame_model = FLAME_mica(mica_flame_config).to(self.device)
        self.default_shape_code = torch.zeros(1, 300, device=self.device)
        self.default_expr_code = torch.zeros(1, 100, device=self.device)
        
        # positional encoding
        self.canonical_pts_freq = 8
        self.pts_embedder_canonical = Embedder(self.canonical_pts_freq)


        self.coarse_pts_freq = 3
        self.pts_embedder_coarse = Embedder(self.coarse_pts_freq)    

        
        _, faces, aux = load_obj('flame/FlameMesh.obj', load_textures=False)

        # 官方文档：https://pytorch3d.org/docs/meshes_io
        # import pdb; pdb.set_trace()
        # faces.verts_idx.shape: torch.Size([10006, 3]) # 面的每个角的顶点索引的 （F，3） 张量
        # faces.textures_idx.shape: torch.Size([10006, 3]) # 面的每个角的uv索引的 （F，3） 张量
        # aux.verts_uvs.shape: torch.Size([5150, 2]) # 顶点的uv坐标 （V，2） 张量
        # uv coordinate per vertex. If a vertex is shared between two faces, it can have a different uv value for each instance.

        uv_coords = aux.verts_uvs[None, ...]
        uv_coords = uv_coords * 2 - 1 # 使其范围从[0, 1]变为[-1, 1]，
        uv_coords[..., 1] = - uv_coords[..., 1] # y轴翻转
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)
        self.uvfaces = faces.textures_idx[None, ...].to(self.device) # torch.Size([1, 10006, 3])
        self.tri_faces = faces.verts_idx[None, ...].to(self.device) # torch.Size([1, 10006, 3])
        
        # rasterizer
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
        
        # flame mask
        # 获取脖子和头部的顶点索引
        flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"   
        flame_mask_dic = load_binary_pickle(flame_mask_path) 
        boundary_id = flame_mask_dic['boundary']
        full_id = np.array(range(5023)).astype(int)
        neckhead_id_list = list(set(full_id)-set(boundary_id))
        self.neckhead_id_list = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks()

        if SAVE_POINT_IMAGE == True:
            ## 删除../output文件夹，生成新的图像存放进去
            if os.path.exists('./output'):
                os.system('rm -r ./output')
                print("Delete ./output")
            else:
                print("No ../output")

    def init_networks(self):       
        ## full mica 
        self.deformNet = MLP(
            # input_dim=self.pts_embedder_canonical.dim_embeded + self.pts_embedder_coarse.dim_embeded + 120,
            input_dim=self.pts_embedder_canonical.dim_embeded + 120,
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6
        )
        
        if Unet_Concat:
            # self.deformNet = DeformNet2D()
            self.deformNet = UNet2D(in_channels=self.pts_embedder_canonical.dim_embeded + self.pts_embedder_coarse.dim_embeded, 
                                    out_channels=10, 
                                    final_sigmoid=False,
                                    layer_order = 'cr',
                                    is_segmentation=False,
                                    )
            # import pdb; pdb.set_trace()
            print("Using UNet2D network as deform net")

            self.flame_to_feature = FLAMEToFeature()

            if Unet_3D:
                self.deformNet = UNet3D(
                                    # in_channels=self.pts_embedder_canonical.dim_embeded + self.pts_embedder_coarse.dim_embeded * 3, 
                                    in_channels= self.pts_embedder_coarse.dim_embeded,
                                    out_channels=10, 
                                    final_sigmoid=False,
                                    layer_order = 'cr',
                                    is_segmentation=False,
                                    conv_kernel_size=(2, 3, 3),
                                    )

                self.flame_to_feature = FLAMEToFeature3D()



        if UNet_AdaIN:
            self.deformNet = Conditional_UNet(input_channel = \
                                              self.pts_embedder_canonical.dim_embeded \
                                              + self.pts_embedder_coarse.dim_embeded \
                                              + 1 , # 1 为 mask
                                              num_classes = 240, # FLAME参数升维的个数
                                              output_channel = 10
                                              )
            
            self.flame_to_feature = ExpandDimNet(input_dim=120, 
                                                 hidden_dim=180, 
                                                 output_dim=240)


        # self.wasserstein_loss = WassersteinGaussian()

        
    def example_init(self, codedict):
        # speed up
        shape_code = codedict['shape'].detach()
        batch_size = shape_code.shape[0]
        geometry_shape = self.flame_model.forward_geo(
            shape_code,
            expression_params = self.default_expr_code
        ) # torch.Size([1, 5023, 3])

        face_vertices_shape = face_vertices_gen(geometry_shape, self.tri_faces.expand(batch_size, -1, -1)) # torch.Size([1, 10006, 3, 3])
        # 生成每个面对应的顶点坐标。每个面有三个顶点，每个顶点有三个坐标
        # import pdb; pdb.set_trace()

        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1),
                                         face_vertices_shape)
        
        ## rast_out: torch.Size([1, 128, 128, 4])
        ## pix_to_face: torch.Size([1, 128, 128, 1])
        ## bary_coords: torch.Size([1, 128, 128, 1, 3])
        ## 获得了uv坐标系下的rasterize结果，包括每个像素对应的面片重心位置 ，每个像素对应的面片的索引，每个像素对应的面片的重心权重
        
        self.pix_to_face_ori = pix_to_face # torch.Size([1, 128, 128, 1])
        self.bary_coords = bary_coords # torch.Size([1, 128, 128, 1, 3]

        uvmask = rast_out[:, -1].unsqueeze(1) # torch.Size([1, 1, 128, 128]) # 对于 uv平面可见的面片索引有哪些
        self.uvmask = uvmask
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
        self.uvmask_flaten_idx = (uvmask_flaten[:]>0)

        pix_to_face_flaten = pix_to_face[0].clone().view(-1) # batch=1    # torch.Size([16384]) # 每个像素对应的面片的索引
        self.pix_to_face = pix_to_face_flaten[self.uvmask_flaten_idx] # pix to face idx    # torch.Size([14876]) # 可见的面片的索引
        self.pix_to_v_idx = self.tri_faces[0, self.pix_to_face, :] # pix to vert idx    # torch.Size([14876, 3]) # 14876个面片的三个顶点的索引

        # import pdb; pdb.set_trace() 
        uv_vertices_shape = rast_out[:, :3] # torch.Size([1, 3, 128, 128]) # 每个像素对应的面片的重心位置 xyz坐标值，作为初始化的gaussian点坐标

        self.canonical_pixel_vals = uv_vertices_shape.clone() # torch.Size([1, 3, 128, 128]) # 每个像素对应的面片的重心位置 xyz坐标值，作为初始化的gaussian点坐标

        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1  # torch.Size([16384, 3])  # 每个像素对应的面片的重心 xyz坐标值，有重复点   
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0) # torch.Size([1, 14876, 3]) # 可见的面片的像素重心的 xyz 坐标值，无重复点

        self.uv_vertices_shape = uv_vertices_shape # for cano init  # torch.Size([1, 14876, 3]) # 经检查里面没有重复的点
        ## 检查self.uv_vertices_shape 里面有没有重复的点
        # import pdb; pdb.set_trace()

        self.uv_vertices_shape_embeded = self.pts_embedder_canonical(uv_vertices_shape) # torch.Size([1, 14876, 51])
        self.v_num = self.uv_vertices_shape_embeded.shape[1] # 14876

        # mask
        self.uv_head_idx = (
            a_in_b_torch(self.pix_to_v_idx[:,0], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,1], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,2], self.neckhead_id_tensor)
        ) # torch.Size([14876])
    
    def decode(self, codedict):

        if Unet_3D:
            ## codedict是一个list

            param_list = [] # 把 1 10 3 128 128 转为所需要的格式
            
            debug_tensors = {}  # Dictionary to collect tensors for debugging

            flame_conditions = []
            coarse_embeddings = []

            # import pdb; pdb.set_trace()

            for single_codedict in codedict:

                shape_code = single_codedict['shape'].detach()
                expr_code = single_codedict['expr'].detach()
                jaw_pose = single_codedict['jaw_pose'].detach()
                eyelids = single_codedict['eyelids'].detach()
                eyes_pose = single_codedict['eyes_pose'].detach()
                batch_size = shape_code.shape[0]
                condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1)

                # condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
                

                uv_vertices, uv_vertices_flaten, pixel_vals, pixel_face_vals = self.process_geometry(shape_code, 
                                                                                                     expr_code, 
                                                                                                     jaw_pose, 
                                                                                                     eyes_pose, 
                                                                                                     eyelids,
                                                                                                     batch_size,
                                                                                                     )

                if INITIAL_ROT_ALONG_FACE:
                    # import pdb; pdb.set_trace()

                    rot_quat_0 = self.calculate_rotation(pixel_face_vals, uv_vertices, uv_vertices_flaten)[:, self.uv_head_idx, :]

                uv_vertices_coarse_embeded_condition = self.pts_embedder_coarse(
                                                            pixel_vals[:, :, :, 0].permute(0, 3, 1, 2))  
                # torch.Size([1, 3, 128, 128]) --> torch.Size([1, N, 128, 128])

                # print("len(flame_conditions)",len(flame_conditions))
                flame_conditions.append(condition) 
                coarse_embeddings.append(uv_vertices_coarse_embeded_condition)

            ## 组织成 1 3 120 的 tensor
                # 使用 torch.stack 将所有的 condition 堆叠在新的维度上
            condition = torch.stack(flame_conditions, dim=1) # 1 3 120

            uv_vertices_shape_embeded_condition = torch.stack(coarse_embeddings, dim=2) # 1 3 128 128

            # import pdb; pdb.set_trace()
            # uv_vertices_shape_embeded_condition = torch.cat((
            #                                             # self.pts_embedder_canonical(self.canonical_pixel_vals), 
            #                                             coarse_embeddings, 
            #                                                 ), 
            #                                             dim=1) # torch.Size([1, 14876, 171])
            
                



            # the offset network takes tracked expression code and the corresponding position of the Gaussian center on canonical mesh as input

            if MLP_Deform:
                uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2) # torch.Size([1, 14876, 171]) origin

            ####################### 以下代码是为了验证uv_vertices_shape_embeded_condition ，获取粗糙形变#######################

            
            ####################### 以上代码是为了验证uv_vertices_shape_embeded_condition ，获取粗糙形变#######################
            # import pdb; pdb.set_trace()
            if COARSE_INPUT:
                # print("使用粗糙形变后的uv_vertices_shape_embeded_condition")    
                if MLP_Deform:  
                    uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, \
                                                                    self.pts_embedder_coarse(uv_vertices), \
                                                                    condition), dim=2) # torch.Size([1, 14876, 171])

                if Unet_Concat:


                    if UNet3D:
                        # import pdb; pdb.set_trace()
                        flame_feature = self.flame_to_feature(condition) # 1 3 120 --> output_flame_tensor
                        
                         # 每个像素是 coarse 形变后的gaussian点位置
                        



                    else:
                        # 使用 UNet2D 的情况
                        flame_feature = self.flame_to_feature(torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1))

                        uv_vertices_coarse_embeded_condition = self.pts_embedder_coarse(pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置

                        ### 把这幅图像画出来看看什么情况

                        ## 要加嵌入，要加 embedding，input 信息量太少
                        uv_vertices_shape_embeded_condition = torch.cat(( \
                                                                        uv_vertices_coarse_embeded_condition, \
                                                                        self.pts_embedder_canonical(self.canonical_pixel_vals)
                                                                        ), dim=1) 
                        # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置

                        # uv_vertices_shape_embeded_condition = torch.cat((pixel_vals[:, :, :, 0].permute(0, 3, 1, 2), flame_feature),dim=1) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置


                if UNet_AdaIN:
                    flame_feature = self.flame_to_feature(torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1))

                    uv_vertices_coarse_embeded_condition = self.pts_embedder_coarse(pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置

                    uv_vertices_shape_embeded_condition = torch.cat((self.uvmask, \
                                                                    uv_vertices_coarse_embeded_condition, \
                                                                    self.pts_embedder_canonical(self.canonical_pixel_vals)
                                                                    ), dim=1)



            if Unet_Concat:
                # import pdb; pdb.set_trace()

                deforms = self.deformNet(uv_vertices_shape_embeded_condition, 
                                        flame_feature)['final'] # torch.Size([1, 10, 128, 128])
                #  应为 1 10 3 128 128
                
                for i in range(deforms.shape[2]):
                    frame_deforms = deforms[:, :, i, :, :]
                    # import pdb; pdb.set_trace()

                    uv_vertices_deforms, rot_delta, scale_coef = self.process_deforms(frame_deforms)

                    # import pdb; pdb.set_trace()
                    # print("i:",i)
                    # print("uv_vertices", uv_vertices.shape)
                    # print("uv_vertices_deforms", uv_vertices_deforms.shape)
                    verts_final = uv_vertices + uv_vertices_deforms

                    # conduct mask
                    # print("self.uv_head_idx", self.uv_head_idx.shape)
                    # print("rot_delta", rot_delta.shape)
                    # print("scale_coef", scale_coef.shape)

                    try:
                        verts_final = verts_final[:, self.uv_head_idx, :] # 1 13453 3
                        rot_final = rot_delta[:, self.uv_head_idx, :] # 1 13453 4
                        scale_coef = scale_coef[:, self.uv_head_idx, :] # 1 13453 3
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

                    if INITIAL_ROT_ALONG_FACE:
                        # import pdb; pdb.set_trace()

                        # rot_quat_0 = rot_quat_0[:, self.uv_head_idx, :] # 13453 4 # 取颈部和脸部的点

                        rot_final = rot_final / (torch.norm(rot_final, dim=-1, keepdim=True) + 1e-6)

                        rot_final = rot_quat_0 + rot_final
                        rot_final = rot_final / (torch.norm(rot_final, dim=-1, keepdim=True) + 1e-6)


                    # import pdb; pdb.set_trace()
                    # print("rot_final shape: ", rot_final.shape)
                    param_list.append(( verts_final, rot_final, scale_coef, uv_vertices_deforms[:, self.uv_head_idx, :], debug_tensors )) # origin: rot_delta ))


            
            elif UNet_AdaIN:
                # import pdb; pdb.set_trace()

                deforms = self.deformNet(uv_vertices_shape_embeded_condition,
                                        flame_feature)

                # 应用激活函数和转换
                # import pdb; pdb.set_trace()
                uv_vertices_deforms = torch.tanh(deforms[:, :3, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1) * 0.02 # 约束offset 漂移小一点 # batch=1  # torch.Size([16384, 3])  # XYZ偏移，保持形状不变
                uv_vertices_deforms = uv_vertices_deforms[self.uvmask_flaten_idx].unsqueeze(0) # torch.Size([1, 14876, 3]) # 保持形状不变

                # 旋转偏移处理
                rot_delta_0 = deforms[:, 3:7, :, :][0].reshape(deforms.shape[2] * deforms.shape[3], -1) # 提取四元数表示的旋转偏移
                rot_delta_0 = rot_delta_0[self.uvmask_flaten_idx].unsqueeze(0)  # 保持形状不变
                rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
                rot_delta_v = rot_delta_0[..., 1:]
                rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)

                # 缩放偏移处理
                scale_coef = torch.exp(deforms[:, 7:, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1)  # 应用指数转换
                scale_coef = scale_coef[self.uvmask_flaten_idx].unsqueeze(0)  # 保持形状不变

                # 注意：以上操作均保持输出张量的形状为原始形状，即 (1, 10, 128, 128) 或相应部分的形状

            else:
                # origin code , MLP
                deforms = self.deformNet(uv_vertices_shape_embeded_condition) # torch.Size([1, 14876, 10])
                deforms = torch.tanh(deforms) 
                uv_vertices_deforms = deforms[..., :3]
                rot_delta_0 = deforms[..., 3:7]
                rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
                rot_delta_v = rot_delta_0[..., 1:]
                rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
                scale_coef = deforms[..., 7:]
                scale_coef = torch.exp(scale_coef)



            # At points where you want to collect tensors for debugging:
            debug_tensors['uvmask'] = self.uvmask.clone()
            debug_tensors['uv_vertices_coarse'] = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2).clone()
            debug_tensors['canonical_pixel_vals_embedded'] = self.canonical_pixel_vals.clone()

            # print("uvmask pixel range: ", debug_tensors['uvmask'].min().item(), "-", debug_tensors['uvmask'].max().item())
            # print("uv_vertices_coarse pixel range: ", debug_tensors['uv_vertices_coarse'].min().item(), "-", debug_tensors['uv_vertices_coarse'].max().item())
            # print("canonical_pixel_vals_embedded pixel range: ", debug_tensors['canonical_pixel_vals_embedded'].min().item(), "-", debug_tensors['canonical_pixel_vals_embedded'].max().item())

            # import pdb; pdb.set_trace()

            # 得到 1 3 3 128 128 的张量
            # 再重新组织
            if SAVE_POINT_IMAGE:
                visualize_verts_final(uv_vertices[:, self.uv_head_idx, :], 
                                    output_dir='./output/visualize_verts_coarse_deform', 
                                    color='blue', 
                                    point_size=0.1)
                visualize_verts_final(verts_final, 
                                    output_dir='./output/visualize_verts_final', 
                                    color='grey',
                                    point_size=0.1)

            # import pdb; pdb.set_trace()
            return param_list


        else: # 如果不是UNet3D

            debug_tensors = {}  # Dictionary to collect tensors for debugging

            shape_code = codedict['shape'].detach()
            expr_code = codedict['expr'].detach()
            jaw_pose = codedict['jaw_pose'].detach()
            eyelids = codedict['eyelids'].detach()
            eyes_pose = codedict['eyes_pose'].detach()
            batch_size = shape_code.shape[0]
            condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1)

            # MLP
            condition = condition.unsqueeze(1).repeat(1, self.v_num, 1) # torch.Size([1, 120]) --> torch.Size([1, 14876, 120])

            # the offset network takes tracked expression code and the corresponding position of the Gaussian center on canonical mesh as input



            ####################### 以下代码是为了验证uv_vertices_shape_embeded_condition ，获取粗糙形变#######################
            geometry = self.flame_model.forward_geo(
                shape_code,
                expression_params=expr_code,
                jaw_pose_params=jaw_pose,
                eye_pose_params=eyes_pose,
                eyelid_params=eyelids,
            ) # torch.Size([1, 5023, 3])

            if SAVE_POINT_IMAGE:
                visualize_flame(self.flame_model, geometry, output_dir='./output/visualize_flame', device=self.device)

            face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1)) # torch.Size([1, 10006, 3, 3]) # 世界坐标系的面片顶点坐标

            # rasterize face_vertices to uv space
            D = face_vertices.shape[-1] # 3
            attributes = face_vertices.clone()
            attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
            N, H, W, K, _ = self.bary_coords.shape
            idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            # 根据可见的顶点索引去取顶点的世界坐标
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D) # torch.Size([1, 128, 128, 1, 3, 3]) # 每个uv像素对应的面片三顶点的世界坐标
            pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2) # torch.Size([1, 128, 128, 1, 3]) # 每个uv像素对应的的gaussian点的世界坐标系位置

            uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2) # torch.Size([1, 3, 128, 128])
            uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1 # torch.Size([16384, 3]) # 筛选出可见的索引
            uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0) # torch.Size([1, 14876, 3])


            if INITIAL_ROT_ALONG_FACE:
                uv_vertices_tangent_0 = pixel_face_vals.view(1,uv_vertices_flaten.shape[0], 3,3)[:,self.uvmask_flaten_idx,0,:] - uv_vertices 
                # torch.Size([1, 14876, 3])
                uv_vertices_tangent_1 = pixel_face_vals.view(1,uv_vertices_flaten.shape[0], 3,3)[:,self.uvmask_flaten_idx,1,:] - uv_vertices

                uv_vertices_normal = torch.cross(uv_vertices_tangent_0, uv_vertices_tangent_1, dim=-1) 

                uv_vertices_normal = uv_vertices_normal / (torch.norm(uv_vertices_normal, dim=-1, keepdim=True)+1e-6)

                uv_vertices_tangent_1 = torch.cross(uv_vertices_normal, uv_vertices_tangent_0, dim=-1)

                rotmat_0 = torch.cat([uv_vertices_tangent_0.unsqueeze(2), uv_vertices_tangent_1.unsqueeze(2), uv_vertices_normal.unsqueeze(2)], dim=2) # torch.Size([1, 14876, 3, 3])

                # rotmat_0 = rotmat_0.transpose(2, 3) # torch.Size([1, 14876, 3, 3]) #### Yihao: Not sure if this is necessary

                rot_quat_0 = matrix_to_quaternion(rotmat_0) # torch.Size([1, 14876, 4])

                

            
            ####################### 以上代码是为了验证uv_vertices_shape_embeded_condition ，获取粗糙形变#######################
            
            if COARSE_INPUT:
                # print("使用粗糙形变后的uv_vertices_shape_embeded_condition")    
                if MLP_Deform:  
                    # uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, \
                    #                                                 self.pts_embedder_coarse(uv_vertices), \
                    #                                                 condition), 
                    #                                                 dim=2) # torch.Size([1, 14876, 192])

                    uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, 
                                                                    condition), 
                                                                    dim=2) # torch.Size([1, 14876, 171]) origin

                if Unet_Concat:

                    flame_feature = self.flame_to_feature(torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1))

                    uv_vertices_coarse_embeded_condition = self.pts_embedder_coarse(pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置

                    ### 把这幅图像画出来看看什么情况

                    ## 要加嵌入，要加 embedding，input 信息量太少
                    uv_vertices_shape_embeded_condition = torch.cat(( \
                                                                    uv_vertices_coarse_embeded_condition, \
                                                                    self.pts_embedder_canonical(self.canonical_pixel_vals)), dim=1) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置

                    # uv_vertices_shape_embeded_condition = torch.cat((pixel_vals[:, :, :, 0].permute(0, 3, 1, 2), flame_feature),dim=1) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置


                if UNet_AdaIN:
                    flame_feature = self.flame_to_feature(torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1))

                    uv_vertices_coarse_embeded_condition = self.pts_embedder_coarse(pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)) # torch.Size([1, 3, 128, 128]) # 每个像素是 coarse 形变后的gaussian点位置

                    uv_vertices_shape_embeded_condition = torch.cat((self.uvmask, \
                                                                    uv_vertices_coarse_embeded_condition, \
                                                                    self.pts_embedder_canonical(self.canonical_pixel_vals)
                                                                    ), dim=1)



            if Unet_Concat:
                deforms = self.deformNet(uv_vertices_shape_embeded_condition, 
                                        flame_feature)['final'] # torch.Size([1, 10, 128, 128])

                # 假设 `deforms` 是2D网络的直接输出：torch.Size([1, 10, 128, 128])
                # deforms已经包含了所有需要的属性，按通道分布

                # 应用激活函数和转换
                # import pdb; pdb.set_trace()
                uv_vertices_deforms = torch.tanh(deforms[:, :3, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1) * 0.02 # 约束offset 漂移小一点 # batch=1  # torch.Size([16384, 3])  # XYZ偏移，保持形状不变
                uv_vertices_deforms = uv_vertices_deforms[self.uvmask_flaten_idx].unsqueeze(0) # torch.Size([1, 14876, 3]) # 保持形状不变

                # 旋转偏移处理
                rot_delta_0 = deforms[:, 3:7, :, :][0].reshape(deforms.shape[2] * deforms.shape[3], -1) # 提取四元数表示的旋转偏移
                rot_delta_0 = rot_delta_0[self.uvmask_flaten_idx].unsqueeze(0)  # 保持形状不变
                rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
                rot_delta_v = rot_delta_0[..., 1:]
                rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)

                # 缩放偏移处理
                scale_coef = torch.exp(deforms[:, 7:, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1)  # 应用指数转换
                scale_coef = scale_coef[self.uvmask_flaten_idx].unsqueeze(0)  # 保持形状不变

                # 注意：以上操作均保持输出张量的形状为原始形状，即 (1, 10, 128, 128) 或相应部分的形状

            
            elif UNet_AdaIN:
                # import pdb; pdb.set_trace()

                deforms = self.deformNet(uv_vertices_shape_embeded_condition,
                                        flame_feature)

                # 应用激活函数和转换
                # import pdb; pdb.set_trace()
                uv_vertices_deforms = torch.tanh(deforms[:, :3, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1) * 0.02 # 约束offset 漂移小一点 # batch=1  # torch.Size([16384, 3])  # XYZ偏移，保持形状不变
                uv_vertices_deforms = uv_vertices_deforms[self.uvmask_flaten_idx].unsqueeze(0) # torch.Size([1, 14876, 3]) # 保持形状不变

                # 旋转偏移处理
                rot_delta_0 = deforms[:, 3:7, :, :][0].reshape(deforms.shape[2] * deforms.shape[3], -1) # 提取四元数表示的旋转偏移
                rot_delta_0 = rot_delta_0[self.uvmask_flaten_idx].unsqueeze(0)  # 保持形状不变
                rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
                rot_delta_v = rot_delta_0[..., 1:]
                rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)

                # 缩放偏移处理
                scale_coef = torch.exp(deforms[:, 7:, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1)  # 应用指数转换
                scale_coef = scale_coef[self.uvmask_flaten_idx].unsqueeze(0)  # 保持形状不变

                # 注意：以上操作均保持输出张量的形状为原始形状，即 (1, 10, 128, 128) 或相应部分的形状

            else:
                # origin code , MLP
                deforms = self.deformNet(uv_vertices_shape_embeded_condition) # torch.Size([1, 14876, 10])
                deforms = torch.tanh(deforms) 
                uv_vertices_deforms = deforms[..., :3]
                rot_delta_0 = deforms[..., 3:7]
                rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
                rot_delta_v = rot_delta_0[..., 1:]
                rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
                scale_coef = deforms[..., 7:]
                scale_coef = torch.exp(scale_coef)

            verts_final = uv_vertices + uv_vertices_deforms

            # conduct mask
            verts_final = verts_final[:, self.uv_head_idx, :] # 1 13453 3

            if SAVE_POINT_IMAGE:
                visualize_verts_final(uv_vertices[:, self.uv_head_idx, :], 
                                    output_dir='./output/visualize_verts_coarse_deform', 
                                    color='blue', 
                                    point_size=0.1)
                visualize_verts_final(verts_final, 
                                    output_dir='./output/visualize_verts_final', 
                                    color='grey',
                                    point_size=0.1)

            rot_delta = rot_delta[:, self.uv_head_idx, :] # 1 13453 4

            if INITIAL_ROT_ALONG_FACE:
                # import pdb; pdb.set_trace()

                rot_quat_0 = rot_quat_0[:, self.uv_head_idx, :] # 13453 4 # 取颈部和脸部的点

                rot_delta = rot_delta / (torch.norm(rot_delta, dim=-1, keepdim=True) + 1e-6)

                rot_final = rot_quat_0 + rot_delta
                rot_final = rot_final / (torch.norm(rot_final, dim=-1, keepdim=True) + 1e-6)

            scale_coef = scale_coef[:, self.uv_head_idx, :] # 1 13453 3


            # At points where you want to collect tensors for debugging:
            debug_tensors['uvmask'] = self.uvmask.clone()
            debug_tensors['uv_vertices_coarse'] = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2).clone()
            debug_tensors['canonical_pixel_vals_embedded'] = self.canonical_pixel_vals.clone()

            # print("uvmask pixel range: ", debug_tensors['uvmask'].min().item(), "-", debug_tensors['uvmask'].max().item())
            # print("uv_vertices_coarse pixel range: ", debug_tensors['uv_vertices_coarse'].min().item(), "-", debug_tensors['uv_vertices_coarse'].max().item())
            # print("canonical_pixel_vals_embedded pixel range: ", debug_tensors['canonical_pixel_vals_embedded'].min().item(), "-", debug_tensors['canonical_pixel_vals_embedded'].max().item())

            # import pdb; pdb.set_trace()

            return verts_final, rot_final, scale_coef, uv_vertices_deforms[:, self.uv_head_idx, :], debug_tensors, self.flame_model # origin: rot_delta 


    def process_geometry(self, shape_code, expr_code, jaw_pose, eyes_pose, eyelids, batch_size):
        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            jaw_pose_params=jaw_pose,
            eye_pose_params=eyes_pose,
            eyelid_params=eyelids,
        )

        if SAVE_POINT_IMAGE:
            visualize_flame(self.flame_model, geometry, output_dir='./output/visualize_flame', device=self.device)

        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        D = face_vertices.shape[-1]
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)

        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0)
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        return uv_vertices, uv_vertices_flaten, pixel_vals, pixel_face_vals,

    def calculate_rotation(self, pixel_face_vals, uv_vertices, uv_vertices_flaten):
        # import pdb; pdb.set_trace()
        # uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0)
        uv_vertices_tangent_0 = pixel_face_vals.view(1, uv_vertices_flaten.shape[0], 3, 3)[:, self.uvmask_flaten_idx, 0, :] - uv_vertices
        uv_vertices_tangent_1 = pixel_face_vals.view(1, uv_vertices_flaten.shape[0], 3, 3)[:, self.uvmask_flaten_idx, 1, :] - uv_vertices

        uv_vertices_normal = torch.cross(uv_vertices_tangent_0, uv_vertices_tangent_1, dim=-1)
        uv_vertices_normal = uv_vertices_normal / (torch.norm(uv_vertices_normal, dim=-1, keepdim=True) + 1e-6)
        uv_vertices_tangent_1 = torch.cross(uv_vertices_normal, uv_vertices_tangent_0, dim=-1)

        rotmat_0 = torch.cat([uv_vertices_tangent_0.unsqueeze(2), uv_vertices_tangent_1.unsqueeze(2), uv_vertices_normal.unsqueeze(2)], dim=2)

        rot_quat_0 = matrix_to_quaternion(rotmat_0)

        return rot_quat_0

    def process_deforms(self, deforms):
        uv_vertices_deforms = torch.tanh(deforms[:, :3, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1) * 0.02
        uv_vertices_deforms = uv_vertices_deforms[self.uvmask_flaten_idx].unsqueeze(0)

        rot_delta_0 = deforms[:, 3:7, :, :][0].reshape(deforms.shape[2] * deforms.shape[3], -1)
        rot_delta_0 = rot_delta_0[self.uvmask_flaten_idx].unsqueeze(0)
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)

        scale_coef = torch.exp(deforms[:, 7:, :, :])[0].reshape(deforms.shape[2] * deforms.shape[3], -1)
        scale_coef = scale_coef[self.uvmask_flaten_idx].unsqueeze(0)

        return uv_vertices_deforms, rot_delta, scale_coef

    
    def capture(self):
        if Unet_Concat or UNet_AdaIN:
            return (
                self.deformNet.state_dict(),
                self.flame_to_feature.state_dict(),
                self.optimizer.state_dict(),
            )
        else:
            return (
                self.deformNet.state_dict(),
                # self.flame_to_feature.state_dict(),
                self.optimizer.state_dict(),
            )
    
    def restore(self, model_args):
        if Unet_Concat or UNet_AdaIN:
            (net_dict,
            flame_to_feature_dict,
            opt_dict) = model_args
            self.deformNet.load_state_dict(net_dict)
            self.flame_to_feature.load_state_dict(flame_to_feature_dict)
            self.training_setup()
            self.optimizer.load_state_dict(opt_dict)
        else:
            (net_dict,
            # flame_to_feature_dict,
            opt_dict) = model_args
            self.deformNet.load_state_dict(net_dict)
            # self.flame_to_feature.load_state_dict(flame_to_feature_dict)
            self.training_setup()
            self.optimizer.load_state_dict(opt_dict)


    def training_setup(self):
        if Unet_Concat or UNet_AdaIN:
            params_group = [
                {'params': self.deformNet.parameters(), 'lr': 1e-4, 'name': 'deform'},
                {'params': self.flame_to_feature.parameters(), 'lr': 1e-4, 'name': 'flame_to_feature'},
            ]
        else:
            params_group = [
                {'params': self.deformNet.parameters(), 'lr': 1e-4, 'name': 'deform'},
            ]
        self.optimizer = torch.optim.AdamW(params_group, betas=(0.9, 0.999))

        ## schedule copy from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/arguments/__init__.py
        self.deform_scheduler_args = get_expon_lr_func(lr_init = 0.00016 * 5,
                                                       lr_final = 0.0000016,
                                                       lr_delay_mult = 0.01,
                                                       max_steps= 150_000)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform" or 'flame_to_feature':
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def get_template(self):
        geometry_template = self.flame_model.forward_geo(
            self.default_shape_code,
            self.default_expr_code,
            self.default_jaw_pose,
            eye_pose_params=self.default_eyes_pose,
        )

        return geometry_template


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output
    


