# FlashAvatar -- with 推土机距离
**[Paper](https://arxiv.org/abs/2312.02214)|[Project Page](https://ustc3dv.github.io/FlashAvatar/)**

![teaser](exhibition/teaser.png)
Given a monocular video sequence, our proposed FlashAvatar can reconstruct a high-fidelity digital avatar in minutes which can be animated and rendered over 300FPS at the resolution of 512×512 with an Nvidia RTX 3090.

## Setup

This code has been tested on Nvidia RTX 3090. 

Create the environment:

```
conda env create --file environment.yml
conda activate FlashAvatar
```

Install PyTorch3D:

```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

## 数据集准备
- 流程代码待整理
数据准备：将视频文件放到original_video_dir中，起一个后缀名字global_unique_suffix

1. 使用 MICA  从单张图像获取 identity.npy：https://github.com/Zielon/MICA.git 官方安装说明
   1. 路径：/root/autodl-tmp/MICA
   2. 保存位置：workplace/WassersteinGS_org-main/MICA/demo/处理批次名字
2. 使用 metrical-tracker 从视频中获取 flame 参数：
   1. 路径：./metrical-tracker/
   2. flame 参数存放路径（FlashAvatar 会根据 idname 到此处读取）: ./metrical-tracker/output
   3. 保存位置 workplace/WassersteinGS_org-main/metrical-tracker/处理批次名字
      - 按照官方说明安装：https://github.com/Zielon/metrical-tracker
3. 进行 png 转 jpg 格式统一, 准备训练图像: ./dataset/face-parsing.PyTorch-master/face-parsing.ipynb
4. 使用脚本提取 segmentation 图像：./dataset/face-parsing.PyTorch-master/face-parsing.ipynb
   1. 脚本来自 https://github.com/zllrunning/face-parsing.PyTorch
5. 训练, 得到 idname 的 gausssian 模型，存放在 ./dataset/idname/log/ckpt

所需文件：https://github.com/zllrunning/face-parsing.PyTorch



## 文件夹路径说明
./output : debug 用的，查看点云位置，flame 面片形变情况

JunLi_exp.ipynb： 将 output 文件夹的图像合并成视频


## 做数据集
```python
python dataset_processing.py
```


## 训练

```python
python train.py
```

## 测试

```python
python test.py 
```
参数说明：
--start_checkpoint ：模型路径

调整嘴部偏移：workplace/WassersteinGS_org-main/flame/flame_mica.py
增加嘴部偏移的测试：test_with_random_noise.py

## 训练过程笔记
【金山文档 | WPS云文档】 wasserstein gaussian 笔记 - 个人工作命令
https://kdocs.cn/l/cji1n9i6vGd2




## Citation
```
@inproceedings{xiang2024flashavatar,
      author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
      title     = {FlashAvatar: High-Fidelity Digital Avatar Rendering at 300FPS},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2024},
  }
```
