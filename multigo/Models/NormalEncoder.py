import torch.nn as nn
import torch.nn.functional as F


import sys
sys.path.append('/hpc2hdd/home/gzhang292/project6/LGM_final_stage1_singleview_basic_animation_original/Models')


from utils.net_util import *

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, opt):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.opt = opt

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, self.opt))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, self.opt))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module(
                'b2_plus_' + str(level), ConvBlock(self.features, self.features, self.opt)
            )

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, self.opt))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)
    
    
class HGFilter(nn.Module):
    def __init__(self, num_modules=4, in_dim=12):
        super(HGFilter, self).__init__()
        self.num_modules = num_modules
        [k, s, d, p] = [7,2,1,3]
        
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=k, stride=s, dilation=d, padding=p)

        self.bn1 = nn.GroupNorm(32, 64)
        self.conv2 = ConvBlock(64, 128, [3,1,1,1])
        self.conv3 = ConvBlock(128, 128, [3,1,1,1])
        self.conv4 = ConvBlock(128, 256, [3,1,1,1])

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256, [3,1,1,1]))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, [3,1,1,1]))
            self.add_module(
                'conv_last' + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            )
            self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))

            self.add_module(
                'l' + str(hg_module),
                nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)
            )

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
                )
                self.add_module(
                    'al' + str(hg_module),
                    nn.Conv2d(4, 256, kernel_size=1, stride=1, padding=0)
                )

    def forward(self, x):#x:normal map 1,3, 512, 512
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2) #1,128,128,128

        x = self.conv3(x) #1,128,128,128
        x = self.conv4(x) #1,256,128,128

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(
                self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True
            )

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs
      
      
if __name__ =="__main__":
    '''
    input: normal map x 4
    shape: (1,12,512,512)  B, C, H, W
    concatenate in the Channel dim 
    
    output:
    list of normal map feature: 
    single feature map shape: (1,4,128,128)
    '''
    net=HGFilter().cuda()
    input=torch.rand(1,12,512,512).cuda()
    output=net(input)