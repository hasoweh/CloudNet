import torch
import torch.nn as nn

def normConv(in_channels, out_channels, padding, bias):
    return nn.Conv2d(in_channels, 
                     out_channels, 
                     kernel_size = 1, 
                     padding = 0, 
                     bias = bias)

class ConvLayer(nn.Sequential):
    def __init__(self, 
                 n_block, 
                 n_layer, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation,
                 padding, 
                 bias):
        super(ConvLayer, self).__init__()
        
        self.add_module('conv%d_%d' % (n_block, n_layer), 
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  padding = padding, 
                                  bias = bias)
                       )
        self.add_module('bnorm%d_%d' % (n_block, n_layer), 
                        nn.BatchNorm2d(out_channels)
                        )
        self.add_module('relu%d_%d' % (n_block, n_layer), 
                        nn.ReLU()
                        )
        
        
class AtrousConv(nn.Sequential):
    def __init__(self, 
                 n_block, 
                 n_layer, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding, 
                 dilation,
                 bias):
        super(AtrousConv, self).__init__()
        
        self.add_module('conv%d_%d' % (n_block, n_layer), 
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  padding = padding, 
                                  dilation = dilation,
                                  bias = bias)
                       )
        self.add_module('bnorm%d_%d' % (n_block, n_layer), 
                        nn.BatchNorm2d(out_channels)
                        )
        self.add_module('relu%d_%d' % (n_block, n_layer), 
                        nn.ReLU()
                        )
        
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel, 
                 dilations, 
                 padding,
                 block_num,
                 bias
                 ):
        
        super(ConvBlock, self).__init__()
        
        # normalize the number of incoming layers
        self.norm = normConv(in_channels, 4, padding, bias) 
        self.aspp1 = AtrousConv(block_num,
                               1,
                               4, 
                               out_channels[0], 
                               kernel, 
                               padding = padding + dilations[0] -1, 
                               dilation = dilations[0],
                               bias = bias)
        self.aspp2 = AtrousConv(block_num,
                               2,
                               out_channels[0], 
                               out_channels[1], 
                               kernel, 
                               padding = padding + dilations[1]-1, 
                               dilation = dilations[1],
                               bias = bias)
        self.aspp3 = AtrousConv(block_num,
                               3,
                               out_channels[1], 
                               out_channels[2], 
                               kernel, 
                               padding = padding + dilations[2]-1, 
                               dilation = dilations[2],
                               bias = bias)
        self.aspp4 = AtrousConv(block_num,
                               4,
                               out_channels[2], 
                               out_channels[3], 
                               kernel, 
                               padding = padding + dilations[3]-1, 
                               dilation = dilations[3],
                               bias = bias)
        
        self.bn_out = nn.BatchNorm2d(out_channels[3]* (len(dilations) + 1)) 
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        x1 = self.norm(x) #residual
        x2 = self.aspp1(x1)
        x3 = self.aspp2(x1)
        x4 = self.aspp3(x1)
        x5 = self.aspp4(x1)
        x = torch.cat((x1, x2, x3, x4, x5), 
                      dim=1) # CORRECT CONCAT DIM?
        x = self.bn_out(x)
        x = self.relu_out(x)
        return x
    
    
class Cloudnet(nn.Module):
    def __init__(self, 
                 res,
                 n_classes,
                 n_channels, # number of bands of original image input
                 out_channels, # list
                 dilations, # list
                 kernel, # int
                 padding, # int
                 n_blocks, # int
                 bias = False
                ):
        assert len(dilations) == len(out_channels), "'dilations' must be same length as 'out_channels'"
        super(Cloudnet, self).__init__()
        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(ConvBlock(n_channels,
                                        out_channels, 
                                        kernel, 
                                        dilations, 
                                        padding,
                                        i,
                                        bias
                                        ))
            else:
                blocks.append(ConvBlock(out_channels[0] * (len(dilations) + 1), # plus 1 because we have the normalization layer and then the dilation layers, so dilation + norm
                                        out_channels, 
                                        kernel, 
                                        dilations, 
                                        padding,
                                        i,
                                        bias
                                        ))
        self.n_class = n_classes
        self.blocks = nn.Sequential(*blocks)
        self.normal = normConv((len(out_channels) +1)*4, 4, padding = 0, bias = bias)
        out = res*res*res*(len(out_channels) + 1)
        self.fc1 = nn.Linear(out, 128, bias = bias)
        self.fc2 = nn.Linear(128, self.n_class, bias = bias)
        
    def forward(self, x):
        x = self.blocks(x)
        x = self.normal(x)
        x = x.view(x.size(0), -1) # flat

        x = self.fc1(x)
        x = self.fc2(x)
        #x = F.softmax(x)
        
        return x
