import os, sys 
import random
import numpy as np
import torch
import argparse
import cv2
import time
import datetime
import pickle

from tqdm import tqdm
from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

import torch
from moviepy.editor import ImageSequenceClip

from flame import FLAME_mica, parse_args



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def get_expression_params(flame_params, index):
    """Retrieves expression parameters for the given index."""

    if index < 0 or index >= flame_params['expression'].shape[0]:
        raise ValueError("Index out of bounds for expression parameters.")

    exp_params = flame_params['expression'][index]
    jaw_pose = flame_params['jaw_pose'][index]
    # Ensure they are in the format expected by your model (e.g., float32)
    expression = torch.tensor(exp_params, dtype=torch.float32).unsqueeze(0).to('cuda')  # Add a batch dimension
    jaw_pose = torch.tensor(jaw_pose, dtype=torch.float32).unsqueeze(0).to('cuda')  # Add a batch dimension

    rotation_matrices = batch_rodrigues(jaw_pose)
    rotation_6d = matrix_to_rotation_6d(rotation_matrices)


    return expression, rotation_6d



def pca(X, n_components):
    """
    对给定的数据集执行PCA并返回前n个主成分。
    
    :param X: 数据集，形状为(num_samples, num_features)
    :param n_components: 保留的主成分数目
    :return: 主成分矩阵，形状为(num_features, n_components)
    """
    # 中心化数据
    X_centered = X - X.mean(dim=0)
    
    # 计算协方差矩阵
    covariance_matrix = torch.mm(X_centered.t(), X_centered) / (X_centered.size(0) - 1)
    
    # 求解特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
    
    # 选择主成分
    idx = eigenvalues.argsort(descending=True)[:n_components]
    principal_components = eigenvectors[:, idx]
    
    return principal_components

def project_to_basis(X, basis):
    """
    将数据集X投影到基向量basis上，并计算投影权重。
    
    :param X: 要投影的数据集，形状为(num_samples, num_features)
    :param basis: 基向量，形状为(num_features, n_components)
    :return: 投影权重，形状为(num_samples, n_components)
    """
    X_centered = X - X.mean(dim=0)
    weights = torch.mm(X_centered, basis)
    return weights

def reconstruct_from_weights(weights, basis):
    """
    使用给定的权重和基向量重构数据。
    
    :param weights: 投影权重，形状为(num_samples, n_components)
    :param basis: 基向量，形状为(num_features, n_components)
    :return: 重构后的数据，形状为(num_samples, num_features)
    """
    return torch.mm(weights, basis.t())

#################################### 单帧数据的情况下

def project_to_basis_single(X, basis, mean):
    """
    将单帧数据X投影到基向量basis上，并计算投影权重。
    
    :param X: 要投影的单帧数据，形状为(1, num_features)或(num_features,)
    :param basis: 基向量，形状为(num_features, n_components)
    :param mean: 数据集的均值，用于中心化，形状为(num_features,)
    :return: 投影权重，形状为(1, n_components)
    """
    # 确保X是二维的
    if X.dim() == 1:
        X = X.unsqueeze(0)
    X_centered = X - mean
    weights = torch.mm(X_centered, basis)
    return weights

def reconstruct_from_weights_single(weights, basis, mean):
    """
    使用给定的权重和基向量重构单帧数据。
    
    :param weights: 投影权重，形状为(1, n_components)
    :param basis: 基向量，形状为(num_features, n_components)
    :param mean: 数据集的均值，用于重构后的数据中心化，形状为(num_features,)
    :return: 重构后的单帧数据，形状为(1, num_features)
    """
    reconstruction = torch.mm(weights, basis.t()) + mean
    return reconstruction


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='RD_Radio1_000_corrected', help='id name')
    ## 用其他 id 的 flame 参数驱动，实现表情迁移
    parser.add_argument('--driven_idname', type=str, default='RD_Radio1_000_corrected', help='flame id name')
    # justin duda
    parser.add_argument('--logname', type=str, default='logname_base写在下面', help='log name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default = '写在下面')
    parser.add_argument("--region", type=str, default = 'mouth') # mouth, eyes, both
    parser.add_argument("--noise_level", type=float, default = 0.1) # 0.001 , if 0, no noise
    parser.add_argument("--Unet_2D", type=bool, default = False) # 选择是否使用 Unet_2D 或 Unet_3D, 只能有一个为 True
    parser.add_argument("--Unet_3D", type=bool, default = False)
    parser.add_argument("--MLP", type=bool, default = True)
    parser.add_argument("--PCA", type=bool, default = False)

    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    # 选择是否使用 Unet_2D 或 Unet_3D, 只能有一个为 True
    PCA = args.PCA # 是否使用 PCA
    Unet_2D = args.Unet_2D
    Unet_3D = args.Unet_3D
    MLP = args.MLP
    noise_level = args.noise_level


    # 根据 idname 构造 logname 和 start_checkpoint
    logname_base = "INITIAL_ROT_ALONG_FACE_更新Rotation_fuxian"  ###########################!!!!!!!!
    args.logname = f"{logname_base}"
    # args.start_checkpoint = f"./dataset/{args.idname}/{logname_base}/ckpt/chkpnt150000.pth"
    args.start_checkpoint = f"./dataset/{args.idname}/{logname_base}/ckpt/chkpnt150000.pth"
    print("use chkpnt150000 for test")
    print("driven_idname: ", args.driven_idname)
    print("source_idname: ", args.idname)    
    print(f"noise level : {noise_level}")
    print(f"add noise to {args.region} region")

    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    print(f"start_checkpoint: {args.start_checkpoint}")

    batch_size = 1
    # set_random_seed(args.seed)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()
    print("DeformModel 注释掉了 eval()，观察 gn 层的影响")

    ## dataloader
    data_dir_s = os.path.join('dataset', args.idname)
    data_dir_d = os.path.join('dataset', args.driven_idname) # driven_id的表情

    mica_datadir_source = os.path.join('metrical-tracker/output', args.idname)
    mica_datadir_driven = os.path.join('metrical-tracker/output', args.driven_idname)

    logdir = data_dir_s+'/'+args.logname
    scene = Scene_mica(data_dir_d, mica_datadir_source, mica_datadir_driven, train_type=3, white_background=lpt.white_background, device = args.device)
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)


    if PCA:
        source_flame = torch.load("./metrical-tracker/output/WDA_TedLieu_000_25fps/source_frame_tensor_WDA_TedLieu_000_25fps.pt").to(args.device)
        # torch.size([248, 120])
        n_components = 60
        # 步骤1: 对A的FLAME参数执行PCA
        mean_source_flame = source_flame.mean(dim=0) # torch.size([120])
        source_flame_principal_components = pca(source_flame, n_components) # # torch.size([120, 28])

    flame_list = []

    # pkl_file_path = '/root/autodl-tmp/inferno/inferno_apps/results/EMOTE_v2-junli-all-2/M003_Neutral_2/flame/flame_M003_Neutral_2.pkl'
    # with open(pkl_file_path, 'rb') as f:
    #     flame_params_from_emote = pickle.load(f)


    if Unet_3D:
        frames = []  # 存储帧的列表
        for iteration in tqdm(range(len(viewpoint)), desc='Rendering'):
            viewpoint_cam = viewpoint[iteration]
            frame_id = viewpoint_cam.uid

            # deform gaussians
            codedict['expr'] = viewpoint_cam.exp_param # torch.size([1, 100])
            codedict['eyes_pose'] = viewpoint_cam.eyes_pose # torch.size([1, 12])
            codedict['eyelids'] = viewpoint_cam.eyelids # torch.size([1, 2])
            codedict['jaw_pose'] = viewpoint_cam.jaw_pose # torch.size([1, 6])

            # 如果使用别的flame参数驱动
            # codedict['expr'], codedict['jaw_pose'] = get_expression_params(flame_params_from_emote, frame_id)

            # print("codedict['expr']: ", codedict['expr'].shape)
            # print("codedict['eyes_pose']: ", codedict['eyes_pose'].shape)
            # print("codedict['eyelids']: ", codedict['eyelids'].shape)
            # print("codedict['jaw_pose']: ", codedict['jaw_pose'].shape)

            # print("iteration before flame_list.append: ", iteration)
            flame_list.append(codedict)


            if PCA:
                ## 用投影到 source id的 pca 上
                # 每次一帧 torch.size([1, 120])
                B_flame = torch.cat((viewpoint_cam.exp_param, viewpoint_cam.eyes_pose, viewpoint_cam.eyelids, viewpoint_cam.jaw_pose), dim=1 ).to(args.device)

                # torch.size([1, 120]) -> torch.size([1, 28])
                B_weights = project_to_basis_single(B_flame, source_flame_principal_components, mean_source_flame)

                # torch.size([1, 28]) -> torch.size([1, 120])
                B_reconstructed = reconstruct_from_weights_single(B_weights, source_flame_principal_components, mean_source_flame) 

                codedict['expr'] = B_reconstructed[:, :viewpoint_cam.exp_param.shape[1]] # torch.size([1, 100])
                codedict['eyes_pose'] = B_reconstructed[:, viewpoint_cam.exp_param.shape[1]:viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]] # torch.size([1, 12])
                codedict['eyelids'] = B_reconstructed[:, viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]:viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]+viewpoint_cam.eyelids.shape[1]] # torch.size([1, 2])
                codedict['jaw_pose'] = B_reconstructed[:, viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]+viewpoint_cam.eyelids.shape[1]:] # torch.size([1, 6])


            if len(flame_list) == 3:

                # print("flame_list",len(flame_list))

                # import pdb; pdb.set_trace()

                param_list = DeformModel.decode(flame_list)

                for j, (verts_final, rot_delta, scale_coef, verts_offset, debug_tensors) in enumerate(param_list):

                    # print("verts_final: ", verts_final[0])
                    # print("rot_delta: ", rot_delta[0])
                    # print("scale_coef: ", scale_coef[0])

                    # 检查verts_final是否和上一个一模一样
                    if j > 0:
                        if verts_final is prev_verts_final:
                            print("verts_final has not changed.")
                        if rot_delta is prev_rot_delta:
                            print("rot_delta has not changed.")
                        if scale_coef is prev_scale_coef:
                            print("scale_coef has not changed.")
                        if verts_offset is prev_verts_offset:
                            print("verts_offset has not changed.")
                        # if debug_tensors is prev_debug_tensors:
                        #     print("debug_tensors has not changed.")
                    
                    # Store previous values for the next iteration
                    prev_verts_final = verts_final
                    prev_rot_delta = rot_delta
                    prev_scale_coef = scale_coef
                    prev_verts_offset = verts_offset
                    prev_debug_tensors = debug_tensors


                    viewpoint_cam = viewpoint[iteration-2+j]


                    gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])


                    # if j > 0:  # Skip the first iteration
                    #     if viewpoint_cam is prev_viewpoint_cam:
                    #         print("viewpoint_cam has not changed.")
                    #     if gaussians is prev_gaussians:
                    #         print("gaussians has not changed.")
                    #     if ppt is prev_ppt:
                    #         print("ppt has not changed.")
                    #     if background is prev_background:
                    #         print("background has not changed.")

                    # Store previous values for the next iteration
                    prev_viewpoint_cam = viewpoint_cam
                    prev_gaussians = gaussians  # Assuming it's directly modifiable
                    prev_ppt = ppt
                    prev_background = background

                    # Render
                    render_pkg = render(viewpoint_cam, gaussians, ppt, background)
                    image= render_pkg["render"]
                    image = image.clamp(0, 1)

                    gt_image = viewpoint_cam.original_image
                    save_image = np.zeros((args.image_res, args.image_res*2, 3))
                    gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
                    image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

                    save_image[:, :args.image_res, :] = gt_image_np
                    save_image[:, args.image_res:, :] = image_np
                    save_image = save_image.astype(np.uint8)
                    save_image = save_image[:,:,[2,1,0]]

                    
                    ## 保存图片
                    # print("iteration: ", iteration)
                    # print("frame_id: ", frame_id)
                    
                    # cv2.imwrite(f"{logdir}/{args.driven_idname}_to_{args.idname}_{args.logname}.png", save_image)

                    # print(f"Image saved to {logdir}/{args.driven_idname}_to_{args.idname}_{args.logname}.png")

                    # import pdb; pdb.set_trace()

                    # save_image_rgb = cv2.cvtColor(save_image, cv2.COLOR_BGR2RGB)
                    # frames.append(save_image_rgb)
            # print(f"length: {len(viewpoint)}, iteration: {iteration}")
                #flame_list.clear()

        # 使用 moviepy 创建视频
        # clip = ImageSequenceClip(frames, fps=25)
        # vid_save_path = os.path.join(logdir, f'{args.driven_idname}_to_{args.idname}_{args.logname}.mp4')
        # clip.write_videofile(vid_save_path, codec='libx264')
        # print(f"Video saved to {vid_save_path}")
        # print("flame of driven_idname: ", args.driven_idname)
        # print("gaussian idname: ", args.idname)
        

    if Unet_2D or MLP:
        frames = []  # 存储帧的列表

        for iteration in tqdm(range(len(viewpoint)), desc='Rendering'):
            viewpoint_cam = viewpoint[iteration]
            frame_id = viewpoint_cam.uid

            # deform gaussians
            codedict['expr'] = viewpoint_cam.exp_param
            codedict['eyes_pose'] = viewpoint_cam.eyes_pose
            codedict['eyelids'] = viewpoint_cam.eyelids
            codedict['jaw_pose'] = viewpoint_cam.jaw_pose


            # 添加随机噪声
            # codedict['expr'] += torch.randn_like(codedict['expr']) * noise_level
            # codedict['jaw_pose'] += torch.randn_like(codedict['jaw_pose']) * noise_level
            # codedict['eyes_pose'] += torch.randn_like(codedict['eyes_pose']) * noise_level
            # codedict['eyelids'] += torch.randn_like(codedict['eyelids']) * noise_level



            ## 如果使用外来的flame参数
            # codedict['expr'], codedict['jaw_pose'] = get_expression_params(flame_params_from_emote, frame_id)

            if PCA:
                ## 用投影到 source id的 pca 上
                # 每次一帧 torch.size([1, 120])
                B_flame = torch.cat((viewpoint_cam.exp_param, viewpoint_cam.eyes_pose, viewpoint_cam.eyelids, viewpoint_cam.jaw_pose), dim=1 ).to(args.device)

                # torch.size([1, 120]) -> torch.size([1, 28])
                B_weights = project_to_basis_single(B_flame, source_flame_principal_components, mean_source_flame)

                # torch.size([1, 28]) -> torch.size([1, 120])
                B_reconstructed = reconstruct_from_weights_single(B_weights, source_flame_principal_components, mean_source_flame) 

                codedict['expr'] = B_reconstructed[:, :viewpoint_cam.exp_param.shape[1]] # torch.size([1, 100])
                codedict['eyes_pose'] = B_reconstructed[:, viewpoint_cam.exp_param.shape[1]:viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]] # torch.size([1, 12])
                codedict['eyelids'] = B_reconstructed[:, viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]:viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]+viewpoint_cam.eyelids.shape[1]] # torch.size([1, 2])
                codedict['jaw_pose'] = B_reconstructed[:, viewpoint_cam.exp_param.shape[1]+viewpoint_cam.eyes_pose.shape[1]+viewpoint_cam.eyelids.shape[1]:] # torch.size([1, 6])


            verts_final, rot_delta, scale_coef, _, _, flame_model = DeformModel.decode(codedict)

            ''''''
            # 获取嘴部顶点的掩码
            mouth_mask = flame_model.get_facial_region_mask(verts_final, region=args.region)
            

            # 只对嘴部顶点添加噪声
            noise = torch.randn_like(verts_final) * noise_level
            verts_final[mouth_mask] += noise[mouth_mask]
            ''''''

            gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

            # Render
            render_pkg = render(viewpoint_cam, gaussians, ppt, background)
            image = render_pkg["render"]
            image = image.clamp(0, 1)

            gt_image = viewpoint_cam.original_image
            image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()
            
            # 创建目录结构
            if noise_level > 0: 
                base_dir = os.path.join(logdir, f'noise_{noise_level:.2e}_{args.driven_idname}_to_{args.idname}_{args.logname}_{args.start_checkpoint.split("/")[-1]}')
            else:
                base_dir = os.path.join(logdir, f'{args.driven_idname}_to_{args.idname}_{args.logname}_{args.start_checkpoint.split("/")[-1]}')
            
            # 创建GT和render子目录
            gt_dir = os.path.join(base_dir, 'GT')
            render_dir = os.path.join(base_dir, 'render')
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(render_dir, exist_ok=True)
            
            # 保存GT图像
            gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
            gt_frame_path = os.path.join(gt_dir, f'frame_{iteration:04d}.png')
            cv2.imwrite(gt_frame_path, cv2.cvtColor(gt_image_np, cv2.COLOR_BGR2RGB))
            
            # 保存渲染图像
            render_frame_path = os.path.join(render_dir, f'frame_{iteration:04d}.png')
            cv2.imwrite(render_frame_path, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            
            # 只将渲染结果添加到视频帧中
            # frame_rgb = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # frames.append(frame_rgb)
            frames.append(image_np.astype(np.uint8))

        # 使用 moviepy 创建视频
        clip = ImageSequenceClip(frames, fps=25)
        if noise_level > 0:
            vid_save_path = os.path.join(logdir, f'noise_{noise_level:.2e}_{args.driven_idname}_to_{args.idname}_{args.logname}_{args.start_checkpoint.split("/")[-1]}.mp4')
        else:
            vid_save_path = os.path.join(logdir, f'{args.driven_idname}_to_{args.idname}_{args.logname}_{args.start_checkpoint.split("/")[-1]}.mp4')
        clip.write_videofile(vid_save_path, codec='libx264')

        print(f"Video saved to {vid_save_path}")
        print("flame of driven_idname: ", args.driven_idname)
        print("gaussian idname: ", args.idname)

    else:
        print("Unet_2D 和 Unet_3D 不能同时为 True, 也不能同时为 False")
        exit()
    
   
        

           