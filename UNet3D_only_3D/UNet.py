import torch
import torch.nn as nn
import importlib

import importlib

## 跑实验的时候用这个
from .buildingblocks import DoubleConv, create_decoders, create_encoders

# # ## 调试 Unet.py 文件的时候用这个
# from buildingblocks import DoubleConv, create_decoders, create_encoders

import sys
sys.path.append('..')

from src.JunLi_2D import FLAMEToFeature, FLAMEToFeature3D

import torch
import torch.nn as nn

class DepthReductionNet(nn.Module):
    def __init__(self):
        super(DepthReductionNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(10, 16, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(8, 10, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            # New final convolutional layer with adjusted stride to reach the final depth of 3
            nn.Conv3d(10, 10, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

'''
# Initialize the network
model = DepthReductionNet()

# Test the model with an example input tensor
input_tensor = torch.randn(1, 1, 7, 128, 128)
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)
'''


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps * 2^ k, k=1,2,3
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
            self.final_depth_adjust = DepthReductionNet()
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x, flame):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        decoders_features = []
        for decoder, encoder_features, flame_feature in zip(self.decoders, encoders_features, flame):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            # import pdb; pdb.set_trace()
            x = decoder(encoder_features, x, flame_feature)  # 多加一个 FLAME 参数
            decoders_features.append(x)

        x = self.final_conv(x)

        x = self.final_depth_adjust(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        #print('encoders_features in reverse orders, with total numbers:', len(self.encoders))
        return {'final':x, 'en_features':encoders_features, 'de_features':decoders_features}



class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     pool_kernel_size=pool_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=True)


# class ResidualUNet3D(AbstractUNet):
#     """
#     Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
#     Uses ResNetBlock as a basic building block, summation joining instead
#     of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
#     Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
#     """

#     def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
#                  num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
#         super(ResidualUNet3D, self).__init__(in_channels=in_channels,
#                                              out_channels=out_channels,
#                                              final_sigmoid=final_sigmoid,
#                                              basic_module=ResNetBlock,
#                                              f_maps=f_maps,
#                                              layer_order=layer_order,
#                                              num_groups=num_groups,
#                                              num_levels=num_levels,
#                                              is_segmentation=is_segmentation,
#                                              conv_padding=conv_padding,
#                                              is3d=True)


# class ResidualUNetSE3D(AbstractUNet):
#     """_summary_
#     Residual 3DUnet model implementation with squeeze and excitation based on 
#     https://arxiv.org/pdf/1706.00120.pdf.
#     Uses ResNetBlockSE as a basic building block, summation joining instead
#     of concatenation joining and transposed convolutions for upsampling (watch
#     out for block artifacts). Since the model effectively becomes a residual
#     net, in theory it allows for deeper UNet.
#     """

#     def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
#                  num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
#         super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
#                                                out_channels=out_channels,
#                                                final_sigmoid=final_sigmoid,
#                                                basic_module=ResNetBlockSE,
#                                                f_maps=f_maps,
#                                                layer_order=layer_order,
#                                                num_groups=num_groups,
#                                                num_levels=num_levels,
#                                                is_segmentation=is_segmentation,
#                                                conv_padding=conv_padding,
#                                                is3d=True)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, conv_padding=1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=False)


def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model'
    ])
    return model_class(**model_config)


if __name__ == '__main__':


    print("flame3d")
    # Example usage
    flame_3d = FLAMEToFeature3D()
    flame_params = torch.randn(1, 3, 120)  # Simulate a batch of flame parameters
    output_feature_maps = flame_3d(flame_params)

    for i, fmap in enumerate(output_feature_maps):
        print(f"Shape of flame {i}: {fmap.size()}")


    print("3d***************")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = UNet3D(in_channels=3, 
                    out_channels=10, 
                    final_sigmoid=False,
                    layer_order = 'cr',
                    is_segmentation=False,
                    conv_kernel_size=(2, 3, 3),
                    # pool_kernel_size=(1, 2, 2),
                    # conv_padding=(1, 1, 1),
                    )
    
    model.to(device)

    # print("Unet3D:", model)
    
    data_input = torch.randn(1,3,3,128,128).to(device)

    ## 原本是 1 1 128（重要） 128 128

    
    # 修改 Unet3d 使其支持这种输入  # (1, 3, 3, 128, 128) # batch, channel, time, height, width

    ## 输出 Unet3D 的结果是 1 10 3 128 128  # batch, channel, time, height, width


    output = model(data_input, output_feature_maps)
    print({i:x.shape for i,x in enumerate(output['de_features'])})
    print(output['final'].shape)


    
    # print("2d---------------")
    # model_2D = UNet2D(in_channels=3, out_channels=10, final_sigmoid=False)
    # model_2D.to(device)

    # data_input_2D = torch.randn(1,3,64,64).to(device)
    # output_2D = model_2D(data_input_2D)

    # print({i:x.shape for i,x in enumerate(output_2D['de_features'])})
    # print(output_2D['final'].shape)




    ## 文件夹外的测试版本

    # # 创建一个3D转置卷积层
    # trans_conv = nn.ConvTranspose3d(
    #     in_channels=1,     # 输入的通道数
    #     out_channels=1,    # 输出的通道数
    #     kernel_size=(2, 2, 2),  # 核大小，depth，height和width方向都为2
    #     stride=(1, 2, 2),       # 步长，depth方向为1，height和width方向为2
    #     padding=(0, 0, 0)       # 填充，所有方向为0
    # )

    # # 模拟输入数据
    # # 假设批量大小为1，通道数为1，depth为5，height和width为4
    # input_tensor = torch.randn(1, 1, 5, 4, 4)

    # # 应用转置卷积
    # output_tensor = trans_conv(input_tensor)

    # # 输出张量的形状
    # print("Output tensor shape:", output_tensor.shape)
