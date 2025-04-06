import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AdaIN, HalfDropout, BatchNorm
from .nets import r_double_conv


class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self, input_channel = 3, num_classes = 10, output_channel = 3):
        super(Conditional_UNet, self).__init__()

        '''
        # self.dconv_down1 = r_double_conv(3, 64)
        self.dconv_down1 = r_double_conv(input_channel, 64)
        self.dconv_down2 = r_double_conv(64, 128)
        self.dconv_down3 = r_double_conv(128, 256)
        self.dconv_down4 = r_double_conv(256, 512)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        #self.dropout_half = HalfDropout(p=0.3)
        
        # self.adain3 = AdaIN(512, num_classes=num_classes)
        # self.adain2 = AdaIN(256, num_classes=num_classes)
        # self.adain1 = AdaIN(128, num_classes=num_classes)

        # flame input
        self.adain3 = AdaIN(512, num_classes=num_classes)
        self.adain2 = AdaIN(256, num_classes=num_classes)
        self.adain1 = AdaIN(128, num_classes=num_classes)

        self.dconv_up3 = r_double_conv(256 + 512, 256)
        self.dconv_up2 = r_double_conv(128 + 256, 128)
        self.dconv_up1 = r_double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, output_channel, 1)
        # self.activation = nn.Tanh()
        #self.init_weight() 
        '''
        

        # ''' 网络 channel 数量翻倍
        self.dconv_down1 = r_double_conv(input_channel, 64 * 2)
        self.dconv_down2 = r_double_conv(64 * 2, 128 * 2)
        self.dconv_down3 = r_double_conv(128 * 2, 256 * 2)
        self.dconv_down4 = r_double_conv(256 * 2, 512 * 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        # self.dropout = nn.Dropout(p=0.3)

        self.adain3 = AdaIN(512 * 2, num_classes=num_classes)
        self.adain2 = AdaIN(256 * 2, num_classes=num_classes)
        self.adain1 = AdaIN(128 * 2, num_classes=num_classes)

        self.dconv_up3 = r_double_conv((256 + 512) * 2, 256 * 2)
        self.dconv_up2 = r_double_conv((128 + 256) * 2, 128 * 2)
        self.dconv_up1 = r_double_conv((64 + 128) * 2, 64 * 2)

        self.conv_last = nn.Conv2d(64 * 2, output_channel, 1)
        # '''

        
        
    def forward(self, x, c):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)

        #dropout
        #x = self.dropout_half(x)
        
        x = self.adain3(x, c)
        x = self.upsample(x)
        # x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.adain2(x, c)
        x = self.upsample(x)        
        # x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)

        x = self.adain1(x, c)
        x = self.upsample(x)        
        # x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        # return self.activation(out)
        return out

class ExpandDimNet(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=180, output_dim=240):
        super(ExpandDimNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


## 程序入口
if __name__ == '__main__':

    from utils import AdaIN, HalfDropout, BatchNorm
    from nets import r_double_conv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Conditional_UNet(num_classes=240,output_channel=10).to(device)
    # print(model)

    input = torch.randn(1, 51+21, 128, 128).to(device)
    c = torch.randn(1, 120).to(device)

    net = ExpandDimNet(input_dim=120, hidden_dim=180, output_dim=240).to(device)
    c_expanded = net(c)

    import pdb; pdb.set_trace()

    output = model(input, c_expanded)

    print("input shape: ", input.shape)
    print("output shape: ", output.shape)