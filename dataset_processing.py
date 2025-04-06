import os
import subprocess
import sys
import numpy as np

from PIL import Image
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import torch
import torchvision.transforms as transforms
import cv2
# from pathlib import Path
import os.path as osp

import yaml

# 将face-parsing相关的模块添加到系统路径
sys.path.append('./face_parsing_PyTorch_master')
from face_parsing_PyTorch_master.model import BiSeNet
import argparse

def is_video_file(filename):
    """
    Check if a file is a common video file based on its extension.

    Parameters:
    - filename (str): The name of the file to check.

    Returns:
    - bool: True if the file is a video file, False otherwise.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def generate_id_name_list_os(original_video_dir):
    files = os.listdir(original_video_dir)
    id_name_list = [os.path.splitext(file)[0] for file in files if file.endswith('.mp4')]
    return id_name_list

def convert_video_to_mp4_25fps(input_file, output_file_dir, max_length_sec=60):
    """
    将视频转换为25fps的MP4格式，并限制最大长度为指定秒数
    
    Parameters:
    - input_file: 输入视频文件路径
    - output_file_dir: 输出目录
    - max_length_sec: 最大视频长度（秒），默认为60秒
    """
    # 检查文件是否已经是MP4格式并且帧率为25fps
    if not input_file.endswith('.mp4'):
        needs_conversion = True
    else:
        clip = VideoFileClip(input_file)
        if clip.fps != 25:
            needs_conversion = True
        else:
            needs_conversion = False
        
        # 检查视频长度
        duration = clip.duration
        clip.close()
    
    # 如果需要转换，加载视频（如果还没加载）并转换
    if needs_conversion:
        try:
            with VideoFileClip(input_file) as clip:
                # 构建输出文件路径
                output_file = os.path.join(output_file_dir, os.path.basename(input_file))
                
                # 检查视频长度并截取
                if clip.duration > max_length_sec:
                    print(f"视频长度为 {clip.duration:.2f}秒，将截取前 {max_length_sec} 秒")
                    clip = clip.subclip(0, max_length_sec)
                
                # 转换视频并保存
                clip.write_videofile(output_file, fps=25)
                print(f"视频转换完成并保存为 {output_file}")
        except Exception as e:
            print(f"处理视频 {input_file} 时发生错误: {e}")
    else:
        # 即使格式和帧率正确，也需要检查长度
        with VideoFileClip(input_file) as clip:
            output_file = os.path.join(output_file_dir, os.path.basename(input_file))
            
            if clip.duration > max_length_sec:
                print(f"视频长度为 {clip.duration:.2f}秒，将截取前 {max_length_sec} 秒")
                subclip = clip.subclip(0, max_length_sec)
                subclip.write_videofile(output_file, fps=25)
                print(f"视频截取完成并保存为 {output_file}")
            else:
                print(f"视频 {input_file} 已经是MP4格式、25fps且长度小于 {max_length_sec} 秒")
                # 复制到目标文件夹
                subprocess.run(['cp', input_file, output_file_dir])


def extract_first_frame(source_folder, target_folder):
    for video_name in tqdm(os.listdir(source_folder),desc='Extracting first frame'):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(source_folder, video_name)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame_name = os.path.splitext(video_name)[0] + '.jpg'
                frame_path = os.path.join(target_folder, frame_name)
                cv2.imwrite(frame_path, frame)
            cap.release()


## 执行 MICA 面捕算法，获取 FLAME 的 shape 参数，保存在 identity.npy 文件
def extract_identity(input_folder, output_folder):
    # 构建命令行参数
    # input_folder 前面那个 MICA 不要
    parts = input_folder.split('/')[1:]  # 分割路径并获取第二到最后一个元素
    input_folder = '/'.join(parts[1:])  # 将第二到最后一个元素连接成一个新的路径
    # 回退一个文件夹
    output_folder = '.' + output_folder

    cmd = [
        'python', 'demo.py',
        '-i', input_folder,
        '-o', output_folder,
        # '-a', arcface_folder,
        # '-m', model_path
    ]

    # 进入 MICA 目录
    original_dir = os.getcwd()
    print("go into MICA")
    os.chdir('./MICA')

    # 调用subprocess.run执行命令
    print(f"Extracting identity shape parameters from {input_folder} to {output_folder}")
    subprocess.run(cmd)
    print(f"Identity shape parameters extracted and saved to {output_folder}")

    # 退回原目录
    os.chdir(original_dir)



def generate_yml_for_idnames(MICA_output_identity_shape, yml_directory, global_unique_suffix, video_path):
    # 确保目标目录存在
    os.makedirs(yml_directory, exist_ok=True)
    print(f"Generating YML files for idnames in {MICA_output_identity_shape}")
    print(f"YML files will be saved to {yml_directory}")
    
    # 遍历 source_path 下的每个子目录
    for idname in os.listdir(MICA_output_identity_shape):
        idname_path = os.path.join(MICA_output_identity_shape, idname)
        # 往每个子目录下放视频，并重命名为 video.mp4
        subprocess.run(['cp', os.path.join(video_path, idname+'.mp4') , idname_path + '/video.mp4'])

        if os.path.isdir(idname_path):
            # 这里创建一个示例配置字典，实际情况下你可能需要根据实际情况来填充
            config_dict = {
                'actor': f'./input/{global_unique_suffix}/{idname}',
                'save_folder': './output/',
                'optimize_shape': True,
                'optimize_jaw': True,
                'begin_frames': 1,
                'keyframes': [0, 1]
            }
            
            # 生成 YML 文件路径
            yml_file_path = os.path.join(yml_directory, f"{idname}.yml")
            
            # 保存配置字典到 YML 文件
            with open(yml_file_path, 'w') as yml_file:
                yaml.dump(config_dict, yml_file)


def execute_tracker_for_all_idnames(configs_directory, id_name_list):
    print("id_name_list: ", id_name_list)

    parts = configs_directory.split('/')[1:]  # 分割路径并获取第二到最后一个元素
    configs_directory = '/'.join(parts[1:])  # 将第二到最后一个元素连接成一个新的路径

    # 进入 metrical-tracker 目录
    original_dir = os.getcwd()
    os.chdir('./metrical-tracker')

    # 遍历 configs_directory 下的每个 YML 文件
    for config_file in tqdm(os.listdir(configs_directory), desc='Executing tracker for idnames'):
        # 如果 config file 的 idname 在 id_name_list 中
        if config_file.endswith('.yml') and os.path.splitext(config_file)[0] in id_name_list:
            print(f"Executing tracker for {config_file}")
            # 构建命令行指令
            command = f"python tracker.py --cfg ./configs/actors/{config_file}"
            
            # 在命令行中执行指令
            try:
                # 这里使用 subprocess.run()，注意shell=True的安全警告
                subprocess.run(command, shell=True, check=True)
                print(f"成功执行: {config_file}")
            except subprocess.CalledProcessError as e:
                print(f"执行失败: {config_file}", e)

    # 退回原目录
    os.chdir(original_dir)



# Step 1: Generate PNG subfolders list
def generate_png_subfolders_list(root_folder, id_names):
    """
    Generate a list of PNG subfolder paths based on the root folder and id names.

    Parameters:
    - root_folder (str): The root folder path.
    - id_names (list): A list of id names.

    Returns:
    - List[str]: A list of PNG subfolder paths.
    """
    return [f"{root_folder}/{name}/input" for name in id_names]

# Step 2: Generate JPG subfolders list
def generate_jpg_subfolders_list(jpg_folder, id_names):
    """
    Generate a list of JPG subfolder paths based on id names.

    Parameters:
    - id_names (list): A list of id names.

    Returns:
    - List[str]: A list of JPG subfolder paths.
    """
    return [f"{jpg_folder}/{name}/imgs" for name in id_names]

# step3: Convert PNG to JPG with progress
def convert_png_to_jpg_with_progress(source_folder, target_folder):
    """
    Convert all PNG images in a folder to JPG format with a progress bar.

    Parameters:
    - source_folder: Folder containing PNG images.
    - target_folder: Folder where JPG images will be saved.
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(source_folder) if f.endswith(".png")]
    
    # 使用tqdm显示进度条
    for filename in tqdm(png_files, desc="Converting PNG to JPG"):
        basename, extension = os.path.splitext(filename)
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, basename + ".jpg")
        
        # 打开并转换图像
        with Image.open(source_path) as img:
            rgb_im = img.convert('RGB')  # 转换为RGB模式以便正确保存为JPG
            rgb_im.save(target_path, "JPEG")
        
        # print(f"Converted {filename} to JPG format.")

# Step 4: Function to execute the whole process
def execute_conversion_process(png_folder, id_names, jpg_folder):
    """
    Execute the PNG to JPG conversion process for each id name.

    Parameters:
    - root_folder (str): The root folder where PNG folders are located.
    - id_names (list): A list of id names.
    """
    png_folders = generate_png_subfolders_list(png_folder, id_names)
    jpg_folders = generate_jpg_subfolders_list(jpg_folder, id_names)
    
    for png_folder, jpg_folder in zip(png_folders, jpg_folders):
        convert_png_to_jpg_with_progress(png_folder, jpg_folder)
        print(f"Completed conversion for {png_folder} to {jpg_folder}")



# 

            
def run_segmentation_for_id_names(id_name_list, base_input_dir, base_output_dir_alpha, base_output_dir_parsing, cp_path):
    """
    Run face segmentation for each id name in the provided list.

    Parameters:
    - id_name_list (list): A list of id names to process.
    - base_input_dir (str): The base input directory path, containing placeholders for id names.
    - base_output_dir_alpha (str): The base output directory path for alpha masks, containing placeholders for id names.
    - base_output_dir_parsing (str): The base output directory path for parsing masks, containing placeholders for id names.
    - cp_path (str): The checkpoint path for the segmentation model.
    """
    current_dir = os.getcwd()
        # 进入指定文件夹
    os.chdir('./face_parsing_PyTorch_master')

    for idname in id_name_list:
        # Build directory paths for the current id name
        input_dir = base_input_dir.format(idname=idname)
        output_dir_alpha = base_output_dir_alpha.format(idname=idname)
        output_dir_parsing = base_output_dir_parsing.format(idname=idname)

        # Call the segmentation function

        face_segmentation(dspth=input_dir, respth_alpha=output_dir_alpha, respth_parsing=output_dir_parsing, save_pth=cp_path)
        print(f"Segmentation completed for {idname}")
    
    # 退回原目录
    os.chdir(current_dir)



            

# BiSeNet 面部分割, 保存到 alpha 和 parsing 目录
def face_segmentation(dspth='./data', respth_alpha='./alpha', respth_parsing='./parsing', save_pth='model_final_diss.pth'):
    if not os.path.exists(respth_alpha):
        os.makedirs(respth_alpha)
    if not os.path.exists(respth_parsing):
        os.makedirs(respth_parsing)

    n_classes = 19  # 总共19个类别
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in tqdm([f for f in os.listdir(dspth) if f.endswith('.jpg')], desc='Processing'):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # 生成人物和背景的mask（类别1-17）
            mask_segment = np.isin(parsing, np.arange(1, 18)).astype(np.uint8)

            # 保存人物和背景的mask到alpha目录
            cv2.imwrite(osp.join(respth_alpha, os.path.splitext(image_path)[0] + '_segment.jpg'), mask_segment * 255)

            # 根据类别生成脖子+头部mask，排除衣服(16)
            mask_neck_head = mask_segment.copy()
            mask_neck_head[parsing == 16] = 0  # 排除衣服
            # 生成嘴部mask
            mask_mouth = np.isin(parsing, [11, 12, 13]).astype(np.uint8)

            # 保存脖子+头部的mask和嘴部的mask到parsing目录
            cv2.imwrite(osp.join(respth_parsing, os.path.splitext(image_path)[0] + '_neckhead.png'), mask_neck_head * 255)
            cv2.imwrite(osp.join(respth_parsing, os.path.splitext(image_path)[0] + '_mouth.png'), mask_mouth * 255)




if __name__ == '__main__':
    # 定义一个变量来存储全局唯一的后缀，您可以根据需要修改这个值
    # Command line argument handling to control which steps to execute

    parser = argparse.ArgumentParser(description='Dataset processing pipeline control')
    parser.add_argument('--steps', type=str, default= 'all', # '5' ,# 'all',
                        help='Comma-separated list of steps to execute (1-7) or "all"') # '1,2,3,4,5,6,7'‘all’
    args = parser.parse_args()
    
    # Parse steps to execute
    if args.steps.lower() == 'all':
        steps_to_execute = list(range(1, 8))  # Steps 1-7
    else:
        try:
            steps_to_execute = [int(step) for step in args.steps.split(',')]
            # Ensure steps are valid
            steps_to_execute = [step for step in steps_to_execute if 1 <= step <= 7]
            if not steps_to_execute:
                print("No valid steps provided. Please specify steps between 1-7.")
                sys.exit(1)
        except ValueError:
            print("Invalid step format. Please use comma-separated integers between 1-7.")
            sys.exit(1)
            
    print(f"Executing steps: {steps_to_execute}")


    global_unique_suffix = "spl_3"  # 文件夹名字，代表本轮处理的数据集

    original_video_dir = f'./original_dataset/{global_unique_suffix}' # 存放每个idname的original avi文件
    source_video = f'./original_dataset_25fps/{global_unique_suffix}' # 存放每个idname的25fps的mp4文件
    MICA_input_frame = f'./MICA/demo/input/{global_unique_suffix}' # 存放每个idname的第一帧图片
    MICA_output_identity_shape = f'./metrical-tracker/input/{global_unique_suffix}' # 存放每个idname文件夹，以及每帧的shape参数
    tracker_config_dir =  './metrical-tracker/configs/actors' # 存放每个idname的yml文件 

    ## 将输出的 png 文件转换为 jpg 文件  ./metrical-tracker/output -> ./dataset
    png_folder = "./metrical-tracker/output"
    jpg_folder = './dataset'
    #### ---------segmentation --------- ####
    base_input_dir = './dataset/{idname}/imgs'
    base_output_dir_alpha = './dataset/{idname}/alpha'
    base_output_dir_parsing = './dataset/{idname}/parsing'
    cp_path = './face_parsing_PyTorch_master/res/cp/79999_iter.pth' # 人脸分割视频路径

    ### --------- 生成 id_name_list --------- ###
    # 获取 id_name_list
    id_name_list = generate_id_name_list_os(original_video_dir)
    print("id_name_list: ", id_name_list)
    print("\n")

    ## 将视频转换为 25fps 的 mp4 格式
    os.makedirs(source_video, exist_ok=True)

    if 1 in steps_to_execute:
        print("1. 将视频转换为 25fps 的 mp4 格式 --------------------------------------------------------------------")
        for video_name in tqdm(os.listdir(original_video_dir), desc='Converting videos to 25fps mp4'):
            if is_video_file(video_name):
                video_path = os.path.join(original_video_dir, video_name)
                convert_video_to_mp4_25fps(video_path, source_video)
                print(f"视频转换完成: {video_name}, to 25fps mp4 format.")
                print("\n")
        print("视频转换完成, 25fps mp4 格式的视频保存在: ", source_video)
        print("\n")


    ## 提取视频首帧到文件夹：
    if 1 in steps_to_execute:
        print("2. 提取视频首帧到文件夹 --------------------------------------------------------------------")
        os.makedirs(MICA_input_frame, exist_ok=True)
        extract_first_frame(source_video, MICA_input_frame)
        print("提取视频首帧到文件夹完成")
        print("MICA_input_frame: ", MICA_input_frame)
        print("\n")


    ######### --------- MICA --------- #########
    ## 从 MICA_input_frame 文件夹中获取 id_name
    ## 执行 MICA 算法，获取 FLAME 的 shape 参数，保存在 identity.npy 文件
    if 3 in steps_to_execute:
        print("3. 执行 MICA 算法，获取 FLAME 的 shape 参数，保存在 identity.npy 文件 --------------------------------------------------------------------")
        os.makedirs(MICA_output_identity_shape, exist_ok=True)
        extract_identity(MICA_input_frame, MICA_output_identity_shape)
        print("MICA 算法执行完成，shape 参数保存在 identity.npy 文件, 路径: ", MICA_output_identity_shape)
        print("\n")


    ### --------- metrical-tracker --------- ###
    ## 为每个 id_name 制作 yml 文件
    if 4 in steps_to_execute:
        print("4. 为每个 id_name 制作 yml 文件 --------------------------------------------------------------------")
        # current_dir = os.getcwd()
        # os.chdir('./metrical-tracker')
        generate_yml_for_idnames(MICA_output_identity_shape, tracker_config_dir, global_unique_suffix, source_video)
        # os.chdir(current_dir)
        print("yml 文件制作完成, 路径: ", tracker_config_dir)
        print("\n")

    # 根据 yml 文件执行 MICA 算法，获取 FLAME 的 shape 参数，保存在 ./metrical-tracker/output
    if 5 in steps_to_execute:
        print("5. 根据 yml 文件执行 MICA 算法，获取 FLAME 的 shape 参数 --------------------------------------------------------------------")
        execute_tracker_for_all_idnames(tracker_config_dir, id_name_list)
        print("MICA 算法执行完成，shape 参数保存在 ./metrical-tracker/output")
        print("\n")

    ## --------- png to jpg --------- ##
    if 6 in steps_to_execute:
        print("6. 将输出的 png 文件转换为 jpg 文件 --------------------------------------------------------------------")
        execute_conversion_process(png_folder, id_name_list, jpg_folder)
        print("png 文件转换为 jpg 文件完成，jpg 文件保存在: ", jpg_folder)

    ## --------- face segmentation --------- ##
    ## 获取当前执行目录
    if 7 in steps_to_execute:
        print("7. 进行面部分割 --------------------------------------------------------------------")
        # 所有文件夹都往回退一个.
        base_input_dir = '.' + base_input_dir
        base_output_dir_alpha = '.' + base_output_dir_alpha
        base_output_dir_parsing = '.' + base_output_dir_parsing
        cp_path = '.' + cp_path
        run_segmentation_for_id_names(id_name_list, base_input_dir, base_output_dir_alpha, base_output_dir_parsing, cp_path)
        # os.chdir(current_dir)
        print("面部分割完成，alpha mask 保存在: ", base_output_dir_alpha)
        print("面部分割完成，parsing mask 保存在: ", base_output_dir_parsing)
        print("\n")

    print("此时 FlashAvatar 的数据集已经准备好了，可以开始训练了")
