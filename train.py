import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import lpips
import wandb
from tqdm import tqdm
from collections import deque
from pytorch3d.transforms import quaternion_to_matrix
from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep
from YiHao_utils import WassersteinGaussian

def parse_args():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument('--log_dir', type=str, default='INITIAL_ROT_ALONG_FACE_更新Rotation_fuxian', help='Base directory for logging')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='RD_Radio1_000_corrected', help='id name')
    parser.add_argument('--driven_idname', type=str, default='RD_Radio1_000_corrected', help='flame id name') # same as idname
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--train_type', type=int, default=0, help='train type')
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)
    return args, lpt, opt, ppt

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

def init_wandb(args, lpt, opt, ppt):
    r"""Initialize Weights and Biases.

    Args:
        args (argparse.Namespace): Command line arguments.
        lpt (ModelParams): Model parameters.
        opt (OptimizationParams): Optimization parameters.
        ppt (PipelineParams): Pipeline parameters.
    """
    wandb.init(project=f"WassersteinGS_{args.idname}_{args.log_dir}", config={
        "model": lpt.__dict__,
        "optimization": opt.__dict__,
        "pipeline": ppt.__dict__,
        "args": args.__dict__
    })

def initialize_training(args, lpt, opt, ppt):
    # Set random seed
    set_random_seed(args.seed)
    # Initialize perception module
    percep_module = lpips.LPIPS(net='vgg').to(args.device)
    # Initialize deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    # Initialize Wasserstein loss
    wasserstein_loss = WassersteinGaussian()
    # Set up directories
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir_source = os.path.join('metrical-tracker/output', args.idname)
    mica_datadir_driven = os.path.join('metrical-tracker/output', args.driven_idname)
    log_dir = os.path.join(data_dir, args.log_dir)
    print(f"Log directory: {log_dir}")
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return DeformModel, percep_module, wasserstein_loss, data_dir, mica_datadir_source, mica_datadir_driven, train_dir, model_dir


if __name__ == "__main__":
    args, lpt, opt, ppt = parse_args()
    init_wandb(args, lpt, opt, ppt)
    DeformModel, percep_module, wasserstein_loss, data_dir, \
    mica_datadir_source, mica_datadir_driven, train_dir, model_dir = initialize_training(args, lpt, opt, ppt)
        
    ###
    # import pdb; pdb.set_trace()
    ###
    
    scene = Scene_mica(data_dir, 
                        mica_datadir_source, 
                        mica_datadir_driven, 
                        train_type=0, 
                        white_background=lpt.white_background, 
                        device = args.device)
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_stack = None
    first_iter += 1
    mid_num = 15000 # 15000 # 50000

    # 初始化 prev_gauss_params 为 None，表示第一帧没有前一帧的参数
    prev_gauss_params = []

    # 初始化Wasserstein loss队列
    w_loss_queue = deque(maxlen=5)
    # 初始化图像损失累加变量
    accumulated_img_loss = torch.tensor(0.0).to(args.device)
    offset_reg_loss = torch.tensor(0.0).to(args.device)
    avg_w_loss = torch.tensor(0.0).to(args.device)
    
    lambda_wasserstein = 0.0005 # 0.0005 # 0 # 0.0005
    # print("delete wasserstein!!!!!!!!!!!")
    lambda_reg = 0.0015 # 0.0015
    lambda_mouth = 40
    lambda_percep = 0.05   

    REVERSE = -1

    # 用于 Unet3d
    flame_list = deque(maxlen=3)
    gaussian_initialized = False  


    # 使用UNet3d时候
    # for iteration in tqdm(range(first_iter, opt.iterations + 1), desc=f"Training {args.idname}"):
    #     # Every 500 its we increase the levels of SH up to a maximum degree
    #     if iteration % 500 == 0:
    #         gaussians.oneupSHdegree()

    #     # random Camera
    #     if not viewpoint_stack:
    #         viewpoint_stack = scene.getCameras().copy()
    #         # random.shuffle(viewpoint_stack)
    #         if len(viewpoint_stack)>3000: # 2000
    #             viewpoint_stack = viewpoint_stack[:3000] # 2000

            
    #     # viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1)) 
    #     # REVERSE *= -1
    #     # if REVERSE == 1:
    #     #     viewpoint_cam = viewpoint_stack.pop(0) # 时间正顺序逐帧读
    #     # else:
    #     #     viewpoint_cam = viewpoint_stack.pop(-1) # 时间负顺序逐帧读

    #     # import pdb; pdb.set_trace()
                
    #     uid = iteration % (len(viewpoint_stack) if len(viewpoint_stack)<2999 else 2999)
        
    #     # print("uid:", uid)
    #     viewpoint_cam = viewpoint_stack[uid] # 时间正顺序逐帧读

    #     frame_id = viewpoint_cam.uid

    #     # deform gaussians
    #     codedict['expr'] = viewpoint_cam.exp_param
    #     codedict['eyes_pose'] = viewpoint_cam.eyes_pose
    #     codedict['eyelids'] = viewpoint_cam.eyelids
    #     codedict['jaw_pose'] = viewpoint_cam.jaw_pose 

    #     # print("gaussians.get_xyz():",gaussians.get_xyz.shape)


    #     flame_list.append(codedict)

        
    #     if len(flame_list) == 3 and uid >= 3:
    #         # import pdb; pdb.set_trace()

    #         param_list = DeformModel.decode(flame_list)

    #         for j, (verts_final, rot_delta, scale_coef, verts_offset, debug_tensors) in enumerate(param_list):

    #             # print(f"frame_id: {frame_id}, j: {j}")
    #             # print("verts_final:", verts_final.shape)
    #             # print("rot_delta:", rot_delta.shape)
    #             # print("scale_coef:", scale_coef.shape)
    #             # print("verts_offset:", verts_offset.shape)
    #             # print("debug_tensors:", debug_tensors.shape)


    #     # verts_final, rot_delta, scale_coef, verts_offset, debug_tensors = DeformModel.decode(codedict)
                
                
    #             # if iteration == 1 :
    #             if iteration in [0, 1, 2, 3] and not gaussian_initialized:
    #                 print("iteration:", iteration)
    #                 print("执行了 gaussian 初始化")
    #                 gaussians.create_from_verts(verts_final[0])
    #                 gaussians.training_setup(opt)
    #                 gaussian_initialized = True

                
    #             # print("verts_final:", verts_final.shape)
    #             # print("rot_delta:", rot_delta.shape)
    #             # print("scale_coef:", scale_coef.shape)

    #             # import pdb; pdb.set_trace()

    #             gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

    #             # 这里加入推土机损失计算
    #             # 如果prev_gauss_params不是None，即不是第一帧，计算Wasserstein损失
    #             w_loss = 0.
    #             # if not iteration == 1 :
    #             # print("iteration:", iteration)
    #             if iteration not in [0, 1, 2, 3]:
    #                 # 计算Wasserstein损失
    #                 # 提前放 prev_gaussian进来
    #                 # import pdb; pdb.set_trace()
    #                 rot_matrix = quaternion_to_matrix(gaussians.get_rotation)
    #                 w_loss = wasserstein_loss(gaussians.get_xyz, 
    #                                         gaussians.get_scaling, 
    #                                         rot_matrix, 
    #                                         prev_gauss_params[0],
    #                                         prev_gauss_params[1],
    #                                         quaternion_to_matrix(prev_gauss_params[2])).mean()
    #             # if iteration % 50 == 0:
    #             #     print("w_loss: ", w_loss.item())
    #             w_loss_queue.append(w_loss)
    #             # import pdb; pdb.set_trace()

    #             viewpoint_cam = viewpoint_stack[uid-2+j]
    #             # print("uid-2+j:", uid-2+j)

    #             # Render
    #             render_pkg = render(viewpoint_cam, gaussians, ppt, background)
    #             image = render_pkg["render"]

    #             # Loss
    #             gt_image = viewpoint_cam.original_image
    #             mouth_mask = viewpoint_cam.mouth_mask
                
    #             loss_huber = huber_loss(image, gt_image, 0.1) + lambda_mouth * huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)
                
    #             loss_G = 0.
    #             head_mask = viewpoint_cam.head_mask
    #             image_percep = normalize_for_percep(image*head_mask)
    #             gt_image_percep = normalize_for_percep(gt_image*head_mask)
    #             if iteration>mid_num:
    #                 loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * lambda_percep

    #                 # 约束顶点损失
    #                 reg_loss = torch.norm(verts_offset, p=2)
    #                 offset_reg_loss = lambda_reg * reg_loss

    #             # 将Wasserstein损失加到总损失中
    #             # loss = loss_huber*1 + loss_G*1 + w_loss * wasserstein_weight
    #             # loss.backward()

    #             # 累加图像损失
    #             current_img_loss = loss_huber + loss_G + offset_reg_loss
    #             accumulated_img_loss += current_img_loss

    #             # import pdb; pdb.set_trace()
    #             prev_gauss_params = [gaussians.get_xyz, 
    #                                 gaussians.get_scaling, 
    #                                 gaussians.get_rotation]

    #             # with torch.no_grad():
    #             #     # Optimizer step
    #             #     if iteration < opt.iterations :
    #             #         gaussians.optimizer.step()
    #             #         DeformModel.optimizer.step()
    #             #         gaussians.optimizer.zero_grad(set_to_none = True)
    #             #         DeformModel.optimizer.zero_grad(set_to_none = True)

    #             # 当队列填满时，计算队列中所有Wasserstein loss的平均值，并加到总损失中
    #             # with torch.no_grad():

                    
    #                 # print loss
    #             if iteration % 1000 == 0:
    #                 if iteration<=mid_num:
    #                     print("step: %d, huber: %.5f, w_loss: %.5f " %(iteration, loss_huber.item(), w_loss.item()))
    #                 else:
    #                     print("step: %d, huber: %.5f,  w_loss: %.5f , percep: %.5f" %(iteration, 
    #                                                                                 loss_huber.item(),  
    #                                                                                 w_loss.item(), 
    #                                                                                 loss_G.item()))
                
    #             # visualize results
    #             with torch.no_grad():
    #                 if iteration % 500 == 0 or iteration in [0,1,2,3,4,5]:
    #                     save_image = np.zeros((args.image_res, args.image_res*2, 3))
    #                     gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    #                     image = image.clamp(0, 1)
    #                     image_np = (image.detach().cpu()*255.).permute(1,2,0).numpy().astype(np.uint8)
    #                     save_image[:, :args.image_res, :] = gt_image_np
    #                     save_image[:, args.image_res:, :] = image_np
    #                     cv2.imwrite(os.path.join(train_dir, f"{iteration}.jpg"), save_image[:,:,[2,1,0]])
    #                     wandb_image = wandb.Image(save_image, caption=f"Iteration {iteration}")
    #                     wandb.log({"Image": wandb_image})

    #                     # import pdb; pdb.set_trace()

    #                     for name, tensor in debug_tensors.items():
    #                         if len(tensor.shape) == 4:  # Assuming it's an image tensor BxCxHxW
    #                             wandb.log({name: [wandb.Image(tensor[i].permute(1,2,0).cpu().numpy(), caption=name) for i in range(tensor.shape[0])]})
    #                         else:
    #                             wandb.log({name: wandb.Histogram(tensor.cpu().numpy())})
                    
    #             # save checkpoint
    #         with torch.no_grad():
    #             if iteration in [0,1,2,3] or iteration % 5000 == 0:
    #                 print("\n[ITER {}] Saving Checkpoint".format(iteration), "to", model_dir)
    #                 torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")
    #                 gaussians.save_ply(model_dir + "/point_cloud_3dgs" + str(iteration) + ".ply") # save gaussian ply

    #         if len(w_loss_queue) == 5:
    #             if iteration>mid_num:
                        
    #                 avg_w_loss = sum(w_loss_queue) / 5

    #             total_loss = accumulated_img_loss + avg_w_loss * lambda_wasserstein  # 总损失包括图像损失和Wasserstein loss
    #             total_loss = accumulated_img_loss  # 总损失包括图像损失和Wasserstein loss
                    
    #             # 反向传播总损失
    #             total_loss.backward()
                
    #             # 更新参数
    #             gaussians.optimizer.step()
    #             DeformModel.optimizer.step()
                
    #             # 清零优化器
    #             gaussians.optimizer.zero_grad(set_to_none=True)
    #             DeformModel.optimizer.zero_grad(set_to_none=True)

    #             learning_rate = DeformModel.update_learning_rate(iteration)


    #             # 在末尾，更新 prev_gauss_params 为当前帧的参数，用于下一次迭代的比较
    #             if iteration > 1 and iteration % 250 == 0:
    #                 wandb.log({
    #                         f"嘴巴：{str(lambda_mouth)}, 全图:{1}, Huber Loss": loss_huber.item() * lambda_mouth, 
    #                         f"{str(lambda_wasserstein)}, Wasserstein Loss": avg_w_loss.item() * lambda_wasserstein, 
    #                         f"{str(lambda_percep)}, Perceptual Loss": loss_G.item() * lambda_percep if iteration>mid_num else 0.0,
    #                         f"{str(lambda_reg)}, offset_reg_loss": offset_reg_loss.item() * lambda_reg,
    #                         "Total Loss": accumulated_img_loss.item(),
    #                         "Iteration": iteration,
    #                         "learning_rate": learning_rate,
    #                         })
                
                
    #             # 重置累加的图像损失
    #             accumulated_img_loss = 0.0

    #             # 清空w_loss_queue
    #             # w_loss_queue.clear()

    #         param_list.clear()
    #         # flame_list.clear()
                

    # 使用2d 或 mlp
    for iteration in tqdm(range(first_iter, opt.iterations + 1),desc=f"Training {args.idname}"):
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            # random.shuffle(viewpoint_stack)
            if len(viewpoint_stack)>2000:
                viewpoint_stack = viewpoint_stack[:2000]

        #     REVERSE *= -1
        # # viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1)) 
        # if REVERSE == 1:
        #     viewpoint_cam = viewpoint_stack.pop(0) # 时间正顺序逐帧读
        # else:
        #     viewpoint_cam = viewpoint_stack.pop(-1) # 时间负顺序逐帧读

        uid = iteration % (len(viewpoint_stack) if len(viewpoint_stack)<2999 else 2999)
        
        # print("uid:", uid)
        viewpoint_cam = viewpoint_stack[uid] # 时间正顺序逐帧读

        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose 

        # print("gaussians.get_xyz():",gaussians.get_xyz.shape)

        verts_final, rot_delta, scale_coef, verts_offset, debug_tensors, _ = DeformModel.decode(codedict)
        
        
        if iteration == 1:
            gaussians.create_from_verts(verts_final[0])
            gaussians.training_setup(opt)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # 这里加入推土机损失计算
        # 如果prev_gauss_params不是None，即不是第一帧，计算Wasserstein损失
        w_loss = 0.
        if not iteration == 1:
            # 计算Wasserstein损失
            # import pdb; pdb.set_trace()
            rot_matrix = quaternion_to_matrix(gaussians.get_rotation)
            w_loss = wasserstein_loss(gaussians.get_xyz, 
                                      gaussians.get_scaling**2, 
                                      rot_matrix, 
                                      prev_gauss_params[0],
                                      prev_gauss_params[1]**2,
                                      quaternion_to_matrix(prev_gauss_params[2])).mean()
        # if iteration % 50 == 0:
        #     print("w_loss: ", w_loss.item())
        w_loss_queue.append(w_loss)
        # import pdb; pdb.set_trace()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        mouth_mask = viewpoint_cam.mouth_mask
        
        loss_huber = huber_loss(image, gt_image, 0.1) + lambda_mouth * huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)
        
        loss_G = 0.
        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image*head_mask)
        gt_image_percep = normalize_for_percep(gt_image*head_mask)
        if iteration>mid_num:
            loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * lambda_percep

            # 约束顶点损失
            reg_loss = torch.norm(verts_offset, p=2)
            offset_reg_loss = lambda_reg * reg_loss

        # 将Wasserstein损失加到总损失中
        # loss = loss_huber*1 + loss_G*1 + w_loss * wasserstein_weight
        # loss.backward()

        # 累加图像损失
        current_img_loss = loss_huber + loss_G + offset_reg_loss
        accumulated_img_loss += current_img_loss

        prev_gauss_params = [gaussians.get_xyz, 
                             gaussians.get_scaling, 
                             gaussians.get_rotation]

        # with torch.no_grad():
        #     # Optimizer step
        #     if iteration < opt.iterations :
        #         gaussians.optimizer.step()
        #         DeformModel.optimizer.step()
        #         gaussians.optimizer.zero_grad(set_to_none = True)
        #         DeformModel.optimizer.zero_grad(set_to_none = True)

        # 当队列填满时，计算队列中所有Wasserstein loss的平均值，并加到总损失中
        # with torch.no_grad():
        if len(w_loss_queue) == 5:
            if iteration>mid_num:
                    
                avg_w_loss = sum(w_loss_queue) / 5

            total_loss = accumulated_img_loss + avg_w_loss * lambda_wasserstein  # 总损失包括图像损失和Wasserstein loss
            total_loss = accumulated_img_loss  # 总损失包括图像损失和Wasserstein loss
                
            # 反向传播总损失
            total_loss.backward()
            
            # 更新参数
            gaussians.optimizer.step()
            DeformModel.optimizer.step()
            
            # 清零优化器
            gaussians.optimizer.zero_grad(set_to_none=True)
            DeformModel.optimizer.zero_grad(set_to_none=True)

            learning_rate = DeformModel.update_learning_rate(iteration)


            # 在末尾，更新 prev_gauss_params 为当前帧的参数，用于下一次迭代的比较
            if iteration > 1 and iteration % 250 == 0:
                wandb.log({
                        f"嘴巴：{str(lambda_mouth)}, 全图:{1}, Huber Loss": loss_huber.item() * lambda_mouth, 
                        f"{str(lambda_wasserstein)}, Wasserstein Loss": avg_w_loss.item() * lambda_wasserstein, 
                        f"{str(lambda_percep)}, Perceptual Loss": loss_G.item() * lambda_percep if iteration>mid_num else 0.0,
                        f"{str(lambda_reg)}, offset_reg_loss": offset_reg_loss.item() * lambda_reg,
                        "Total Loss": accumulated_img_loss.item(),
                        "Iteration": iteration,
                        "learning_rate": learning_rate,
                        })
            
            
            # 重置累加的图像损失
            accumulated_img_loss = 0.0

            # 清空w_loss_queue
            w_loss_queue.clear()
            
            # print loss
        if iteration % 1000 == 0:
            if iteration<=mid_num:
                print("step: %d, huber: %.5f, w_loss: %.5f " %(iteration, loss_huber.item(), w_loss.item()))
            else:
                print("step: %d, huber: %.5f,  w_loss: %.5f , percep: %.5f" %(iteration, 
                                                                            loss_huber.item(),  
                                                                            w_loss.item(), 
                                                                            loss_G.item()))
        
        # visualize results
        with torch.no_grad():
            if iteration % 500 == 0 or iteration==1:
                save_image = np.zeros((args.image_res, args.image_res*2, 3))
                gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)
                image_np = (image.detach().cpu()*255.).permute(1,2,0).numpy().astype(np.uint8)
                save_image[:, :args.image_res, :] = gt_image_np
                save_image[:, args.image_res:, :] = image_np
                cv2.imwrite(os.path.join(train_dir, f"{iteration}.jpg"), save_image[:,:,[2,1,0]])
                wandb_image = wandb.Image(save_image, caption=f"Iteration {iteration}")
                wandb.log({"Image": wandb_image})

                # import pdb; pdb.set_trace()

                for name, tensor in debug_tensors.items():
                    if len(tensor.shape) == 4:  # Assuming it's an image tensor BxCxHxW
                        wandb.log({name: [wandb.Image(tensor[i].permute(1,2,0).cpu().numpy(), caption=name) for i in range(tensor.shape[0])]})
                    else:
                        wandb.log({name: wandb.Histogram(tensor.cpu().numpy())})
            
        # save checkpoint
        with torch.no_grad():
            if iteration % 5000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration), "to", model_dir)
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")
                gaussians.save_ply(model_dir + "/point_cloud_3dgs" + str(iteration) + ".ply") # save gaussian ply
