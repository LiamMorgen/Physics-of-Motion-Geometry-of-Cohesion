import torch
import torch.nn as nn

## added by dengjunli
class DeformNet2D(nn.Module):
    def __init__(self, input_channels=4):
        super(DeformNet2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # 可能需要更多的卷积层或其他类型的层
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # 添加其他所需层

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # 应用更多层
        x = self.conv3(x)  # 不在最后一个卷积层使用激活，因为我们需要直接输出offsets
        return x


class FLAMEToFeature(nn.Module):
    def __init__(self, input_dim=120):
        super(FLAMEToFeature, self).__init__()
        # 将120维向量扩展到足以重塑为具有一定尺寸的特征图的维度
        self.fc = nn.Linear(input_dim, 512)  # 例如，重塑为 [1, 8, 8, 8] 
        self.conv_transpose1 = nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1) # 输出尺寸: [1, 8, 16, 16]
        self.conv_transpose2 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1) # 输出尺寸: [1, 4, 32, 32]
        self.conv_transpose3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 输出尺寸: [1, 2, 64, 64]
        self.conv_transpose4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1) # 输出尺寸: [1, 1, 128, 128]
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = x.view(-1, 8, 8, 8)  # 重塑到更高维度的特征图
        x = self.activation(self.conv_transpose1(x)) # 逐步上采样 torch.size([1, 8, 16, 16])
        
        feature_maps = []  # 初始化一个列表来保存特征图
        
        x = self.activation(self.conv_transpose2(x)) # torch.size([1, 4, 32, 32])
        feature_maps.append(x)  # 保存32x32的特征图
        
        x = self.activation(self.conv_transpose3(x)) # torch.size([1, 2, 64, 64])
        feature_maps.append(x)  # 保存64x64的特征图
        
        x = self.conv_transpose4(x)  # 最后一层不用ReLU，保持线性 # torch.size([1, 1, 128, 128])
        feature_maps.append(x)  # 保存128x128的特征图
        
        return feature_maps




class TimeStepMappingNet(nn.Module):
    def __init__(self):
        super(TimeStepMappingNet, self).__init__()
        
        self.fc = nn.Linear(120, 256)  # 3*120 to 3*100 (assuming H=W=16)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.fc(x)  # Map from 360 to 300
        x = self.relu(x)  # Activation function
        x = x.view(-1, 1, 3, 16, 16)  # Reshape to (batch_size, 1, 3, H, W)
        return x

# Example of usage
# net = TimeStepMappingNet()
# input_tensor = torch.randn(1, 3, 120)  # Random input of shape (1, 3, 120)
# output_tensor = net(input_tensor)

# print("Output shape:", output_tensor.shape)



class FLAMEToFeature3D(nn.Module):
    def __init__(self, input_dim=120):
        super(FLAMEToFeature3D, self).__init__()
        # Adjust the output of the fully connected layer to match the needed reshape
        

        self.Flameto3D = TimeStepMappingNet()

        self.conv_transpose1 = nn.ConvTranspose3d(1, 32, kernel_size=4, stride=2, padding=1)  

        self.conv_transpose2 = nn.ConvTranspose3d(
            in_channels=32,     # 输入的通道数
            out_channels=16,    # 输出的通道数
            kernel_size=(1, 2, 2),  # 核大小，depth方向为1，height和width方向为2
            stride=(1, 2, 2),       # 步长，depth方向为1，height和width方向为2
            padding=(0, 0, 0)       # 填充，所有方向为0
        )  

        self.conv_transpose3 = nn.ConvTranspose3d(
            in_channels=16,     # 输入的通道数
            out_channels=8,    # 输出的通道数
            kernel_size=(2, 2, 2),  # 核大小，depth，height和width方向都为2
            stride=(1, 2, 2),       # 步长，depth方向为1，height和width方向为2
            padding=(0, 0, 0)       # 填充，所有方向为0
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        feature_maps = []  # Initialize a list to save feature maps

        # import pdb; pdb.set_trace()

        x = self.Flameto3D(x)  # Map from 120 to 256
        # print("线性层之后:",x.shape)
        
        x = self.activation(self.conv_transpose1(x))  # Gradually upsample
        feature_maps.append(x)  # Save 16x16x16 feature map
        
        x = self.activation(self.conv_transpose2(x))
        feature_maps.append(x)  # Save 32x32x32 feature map
        
        x = self.conv_transpose3(x)
        feature_maps.append(x)  # Save 64x64x64 feature map
    
        
        return feature_maps









class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DeformNet2D_big(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DeformNet2D_big, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        return out

def DeformNet2D_ResNet():
    return DeformNet2D_big(ResidualBlock, [2, 2, 2, 2])



##### 以下为 FlashAvatar 作者之前写在 deform_model.py 中的代码



    
class MLP2(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim1, output_dim2, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.hidden_layers = hidden_layers
        mid_layers = math.ceil(hidden_layers/2.)
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        self.fcs1 = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i!=(mid_layers-1) else nn.Linear(hidden_dim, hidden_dim+output_dim1) for i in range(mid_layers)]
        )
        self.fcs2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) if i!=0 else nn.Linear(hidden_dim+condition_dim, hidden_dim) for i in range(mid_layers, hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim2)

    def forward(self, input, condition):
        # input: B,V,d1
        # condition: B,d2
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs1):
            h = self.fcs1[i](h)
            h = F.relu(h)
        oup1 = h[:, -self.output_dim1:]
        h = h[:, :-self.output_dim1]
        ...
        for i, l in enumerate(self.fcs2):
            h = self.fcs1[i](h)
            h = F.relu(h)
        # input_ori = input.reshape(batch_size*N_v, -1)
        # h = input_ori
        # for i, l in enumerate(self.fcs):
        #     h = self.fcs[i](h)
        #     h = F.relu(h)
        # output = self.output_linear(h)
        # output = output.reshape(batch_size, N_v, -1)


class SIRENMLP(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=3,
                 hidden_dim=256,
                 hidden_layers=8,
                 condition_dim=100,
                 device=None):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.z_dim = condition_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [FiLMLayer(self.input_dim, self.hidden_dim)] +
            [FiLMLayer(self.hidden_dim, self.hidden_dim) for i in range(self.hidden_layers - 1)]
        )
        self.final_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.mapping_network = MappingNetwork(condition_dim, 256,
                                              len(self.network) * self.hidden_dim * 2)

        self.network.apply(frequency_init(25))
        # self.final_layer.apply(frequency_init(25))
        self.final_layer.weight.data.normal_(0.0, 0.)
        self.final_layer.bias.data.fill_(0.)
        self.network[0].apply(first_layer_film_sine_init)

    def forward_vector(self, input, z):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)

        return sigma

    def forward(self, vertices, additional_conditioning):
        # vertices: in canonical space of flame
        # vertex eval: torch.Size([N, V, 3])
        # map eval:    torch.Size([N, 3, H, W])
        # conditioning
        # torch.Size([N, C])

        # vertex inputs (N, V, 3) -> (N, 3, V, 1)
        vertices = vertices.permute(0, 2, 1)[:, :, :, None]
        b, c, h, w = vertices.shape

        # to vector
        x = vertices.permute(0, 2, 3, 1).reshape(b, -1, c)

        z = additional_conditioning  # .repeat(1, h*w, 1)

        # apply siren network
        o = self.forward_vector(x, z)

        # to image
        o = o.reshape(b, h, w, self.output_dim)
        output = o.permute(0, 3, 1, 2)

        return output[:, :, :, 0].permute(0, 2, 1)

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift, ignore_conditions=None):
        x = self.layer(x)
        if ignore_conditions is not None:
            cond_freq, cond_phase_shift = freq[:-1], phase_shift[:-1]
            cond_freq = cond_freq.unsqueeze(1).expand_as(x).clone()
            cond_phase_shift = cond_phase_shift.unsqueeze(1).expand_as(x).clone()

            ignore_freq, ignore_phase_shift = freq[-1:], phase_shift[-1:]
            ignore_freq = ignore_freq.unsqueeze(1).expand_as(x)
            ignore_phase_shift = ignore_phase_shift.unsqueeze(1).expand_as(x)

            cond_freq[:, ignore_conditions] = ignore_freq[:, ignore_conditions]
            cond_phase_shift[:, ignore_conditions] = ignore_phase_shift[:, ignore_conditions]
            freq, phase_shift = cond_freq, cond_phase_shift

        else:
            freq = freq.unsqueeze(1).expand_as(x)
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        # print('x', x.shape)
        # print('freq', freq.shape)
        # print('phase_shift', phase_shift.shape)
        # x torch.Size([6, 5023, 256])
        # freq torch.Size([6, 5023, 256])
        # phase_shift torch.Size([6, 5023, 256])
        return torch.sin(freq * x + phase_shift)
    
def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
            elif isinstance(m, nn.Conv2d):
                num_input = torch.prod(
                    torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts
    
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
        elif isinstance(m, nn.Conv2d):
            num_input = torch.prod(
                torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
            m.weight.uniform_(-1 / num_input, 1 / num_input)
