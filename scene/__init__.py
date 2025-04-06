import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov

DEBUG = False


def count_jpg_files(directory):
    return len([f for f in os.listdir(directory) if f.endswith('.jpg')])


class Scene_mica:
    def __init__(self, datadir, mica_datadir_source, mica_datadir_driven, train_type, white_background, device):
        ## train_type: 0 for train, 1 for test, 2 for eval, 3 for expression transfer
        ## mica_datadir_source: for the source idname, used to get the camera parameters
        ## mica_datadir_driven: for the driven idname, used to get the flame parameters

        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")

        jpg_count = count_jpg_files(images_folder)
        print(f": {jpg_count} jpg images in {datadir}")

        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir_s = os.path.join(mica_datadir_source, 'checkpoint')
        mica_ckpt_dir_d = os.path.join(mica_datadir_driven, 'checkpoint')
        ################
        # import pdb ; pdb.set_trace()
        
        self.N_frames_d = len(os.listdir(mica_ckpt_dir_d))
        print(f": {self.N_frames_d} FLAME parameter in {mica_ckpt_dir_d}")

        self.N_frames_s = len(os.listdir(mica_ckpt_dir_s))
        print(f": {self.N_frames_s} FLAME parameter in {mica_ckpt_dir_s}")
        
        self.cameras = []
        test_num = 0 # 500
        eval_num = 10 # 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames_s - test_num)

        # load the first frame to get the camera parameters from the mica tracking from driven idname
        ckpt_path_d = os.path.join(mica_ckpt_dir_d, '00000.frame')
        payload_d = torch.load(ckpt_path_d)

        # load the first frame to get the camera parameters from the mica tracking from source idname
        ckpt_path_s = os.path.join(mica_ckpt_dir_s, '00000.frame')
        payload_s = torch.load(ckpt_path_s)
        orig_w, orig_h = payload_s['img_size']

        print("debug: load the first frame to get the camera parameters from the mica tracking from source idname")
        flame_params = payload_s['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])

        K = payload_s['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)

        if train_type == 0:
            range_down = 0
            range_up = train_num # 400 # train_num # 20 # 
            # print(f"range_up: {range_up}","调试用，加载少一些, 训练-观察过拟合情况")
            # print(f"train_num: {train_num}","调试用，加载少一些")
        if train_type == 1: # 测试
            range_down = 0 # self.N_frames - test_num ## 
            range_up = train_num # self.N_frames_s - 200 # self.N_frames
        if train_type == 2: # 验证
            range_down = 0 # self.N_frames - eval_num
            range_up = train_num # self.N_frames_s
        if train_type == 3: # 用其他idname 的 flame 参数驱动，做表情迁移
            range_down = 0
            range_up = min(self.N_frames_s, self.N_frames_d, jpg_count)  # Flame参数和图像帧的最小值
            # range_up = 180
            print(f"range_up: {range_up}","调试用，加载少一些, 测试-观察过拟合情况")
            

        source_frame_list = []
            
        for frame_id in tqdm(range(range_down, range_up-1), desc='Loading cameras'):
            
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking

            # 从 driven idname 获取的 flame 参数
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path_d = os.path.join(mica_ckpt_dir_d, image_name_mica+'.frame')
            payload_d = torch.load(ckpt_path_d)
            flame_params_d = payload_d['flame']

            ckpt_path_s = os.path.join(mica_ckpt_dir_s, image_name_ori+'.frame')
            payload_s = torch.load(ckpt_path_s)
            
            ## 从 driven idname 获取的 flame 参数
            # print("先从 source idname 获取的 flame 参数，debug 用，表情系数改变，眼睛，pose等不变")
            exp_param = torch.as_tensor(flame_params_d['exp'])
            eyes_pose = torch.as_tensor(flame_params_d['eyes'])
            eyelids = torch.as_tensor(flame_params_d['eyelids'])
            jaw_pose = torch.as_tensor(flame_params_d['jaw'])

            if DEBUG:

                flame_params_s = payload_s['flame']
                # 把source id flame参数concat 后保存到 list，最后转为一个二维 tensor
                exp_param_s = torch.as_tensor(flame_params_s['exp'])
                eyes_pose_s = torch.as_tensor(flame_params_s['eyes'])
                eyelids_s = torch.as_tensor(flame_params_s['eyelids'])
                jaw_pose_s = torch.as_tensor(flame_params_s['jaw'])
                # import pdb; pdb.set_trace()

                flame_params_of_source = torch.cat([exp_param_s, eyes_pose_s, eyelids_s, jaw_pose_s], dim=1)
                source_frame_list.append(flame_params_of_source)

            # 从 source idname 获取的 相机 参数
            oepncv = payload_d['opencv']
            # print("debug: 从 driven idname 获取的 flame 参数, 正常应该从 source idname payload_s获取的 相机 参数")
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(images_folder, image_name_ori+'.jpg')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'_segment.jpg') # my pipeline #todo
            # TODO
            # alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg') # origin mica pipeline
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)

        if DEBUG:
            source_frame_tensor = torch.stack(source_frame_list).squeeze(1) # shape: (N, 120)
            # 保存到本地,带着 id 名字
            idname = os.path.basename(mica_datadir_source)
            torch.save(source_frame_tensor, os.path.join(mica_datadir_source, f'source_frame_tensor_{idname}.pt'))
            # import pdb; pdb.set_trace()
            print(f"source_frame_tensor_{idname}.pt saved to {datadir}")
    
    def getCameras(self):
        return self.cameras





    
