"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Date: 2025-07-20 23:50
@Description: 
Based on the code repository https://github.com/shyhyawJou/DEC-Pytorch/blob/main/utils/nn.py
Define the corresponding Deep embedding Clustering model architecture
"""


import torch
import einops
from torch import nn

####################################################################################
####################################################################################
# EncodeDecode
# Deep Embedding Clustering
# https://github.com/shyhyawJou/DEC-Pytorch/blob/main/utils/for_train.py
####################################################################################
####################################################################################

class Encoder(nn.Module):
    
    def __init__(self, feature_in, dropout_p=0.2):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(int(feature_in), 500),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_p),
                                   nn.Linear(500, 500),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_p),
                                   nn.Linear(500, 2000),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_p),
                                   
                                   nn.Linear(2000, 10))
        
    def forward(self, x):
        x = self.dense(x)
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, feature_out,):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(10, 2000),
                                   nn.ReLU(),
                                   nn.Linear(2000, 500),
                                   nn.ReLU(),
                                   nn.Linear(500, 500),
                                   nn.ReLU(),
                                   nn.Linear(500, int(feature_out)))
        
    def forward(self, x):
        x = self.dense(x)
        return x
    

class Cluster(nn.Module):
    # Add a clustering layer to convert the high-dimensional features of each sample into
    # soft assignments, then compare them with the target distribution computed from the soft assignments
    # However, the soft assignments are registered with nn.Parameter and participate in subsequent gradient optimization
    def __init__(self, center, alpha=1):
        super().__init__()
        center = torch.from_numpy(center).to(torch.float32)
        self.center = nn.Parameter(center, requires_grad=False)
        # self.center = center
        self.alpha = alpha

    def forward(self, x):
        square_dist = torch.pow(x[:, None, :] - self.center, 2).sum(dim=2)
        nom = torch.pow(1 + square_dist / self.alpha, - (self.alpha + 1) / 2)
        denom = nom.sum(dim=1, keepdim=True)
        return nom / denom


def get_p(q):
    # Compute the target distribution; see Equation 3
    with torch.no_grad():
        f = q.sum(dim=0, keepdim=True)
        nom = q ** 2 / f
        denom = nom.sum(dim=1, keepdim=True)
    return nom / denom

# def get_p(q):
#     # Compute the target distribution; see Equation 3
#     f = q.sum(dim=0, keepdim=True)
#     nom = q ** 2 / f
#     denom = nom.sum(dim=1, keepdim=True)
#     return nom / denom



class CustomLoss01(nn.Module):
    """
    Compute the target distribution from the soft assignments
    to obtain the corresponding KL divergence values
    
    """
    def __init__(self,):
        super().__init__()
        
        pass
    
    def forward(self, y_input, _):
        # y_target = get_p(y_input).detach()
        y_target = get_p(y_input)
        loss = nn.KLDivLoss(reduction='batchmean')(y_input.log(), y_target)
        # loss = nn.KLDivLoss(reduction='batchmean')(y_target.log(), y_input)
        return loss
    
class HowToDefineCustomLoss(nn.Module):
    """
    Custom loss function
    Constrain the model outputs using the linear relationship s4max = a * dwidz + b
    by fitting a linear relationship between dwidz and s4max
    s4max = -0.043 * dwidz + 0.193
    s4max = a * lct + b
    s4max = a * exp /c 
    """
    
    def __init__(self,):
        super().__init__()
        pass
    
    def forward(self, dwidz, y_input, y_target):
        a = -0.043
        b = 0.193
        y_input_by_dwidz = a * dwidz + b
        
        loss1 = nn.MSELoss()(y_input, y_target)
        loss2 = nn.MSELoss()(y_input_by_dwidz, y_target)
        # loss2 = nn.MSELoss()(y_input_by_dwidz, y_target)
        # loss2 = nn.MSELoss()(y_input_by_dwidz, y_target)
        
        alpha1 = 1.
        alpha2 = 0.
        
        # config_03
        # alpha1 = 1e15
        # alpha2 = 1e-15
        
        loss = alpha1 * loss1 + alpha2 * loss2
        return loss
    
#################
### ResNet ###
##############
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: list,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    pass


class ResNet(nn.Module):
    
    def __init__(self, block_name: str, layers: list):
        super().__init__()
        if block_name == "BasicBlock":
            self.net = _ResNet(BasicBlock, layers)
            pass
        elif block_name == "Bottleneck":
            self.net = _ResNet(Bottleneck, layers)
            pass
        else:
            raise OSError("Unknown block")
            pass
    
    def forward(self, x):
        return self.net(x)
    
    pass

class Trans1DTo2DLayerWith3Channel(nn.Module):
    
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        out = einops.rearrange(x, "b (c w h) -> b c w h", c=3, w=64, h=64)
        return out
    
    pass


class FirstLinear(nn.Module):
    # The first layer transforms the input into 3 * 64 * 64
    def __init__(self, feature_in):
        super().__init__()
        # self.dense = nn.Sequential(nn.Linear(int(feature_in), 64),
        #                             nn.ReLU(),
        #                             nn.Linear(64, 64*32),
        #                             nn.ReLU(),
        #                             nn.Linear(64*32, 64*64*3),
        #                             nn.ReLU(),)
        self.dense = nn.Sequential(nn.Linear(int(feature_in), 64*64*3),
                                   nn.ReLU())
        pass
    
    def forward(self, x):
        return self.dense(x)
    
    pass


class LastLinear(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.dense1 = nn.Sequential(nn.Linear(1024, 64),
        #                             nn.ReLU(),
        #                             nn.Linear(64, 1))
        
        self.dense1 = nn.Sequential(nn.Linear(1000, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 10))
        pass
    
    def forward(self, x):
        # print(x.shape)
        return self.dense1(x)
    
    pass



if __name__ == "__main__":
    import numpy as np
    from collections import OrderedDict
    batch_size = 32
    feature_size = 129 
    input_features = torch.randn(batch_size, feature_size)  
    
    # model1 = nn.Sequential(Encoder(feature_in), Decoder(feature_in))
    # model2 = nn.Sequential(Encoder(feature_in), Cluster(center=np.random.rand(25, 10)))
    submodel = OrderedDict()
    submodel["first_layer"] = FirstLinear(feature_in=feature_size)
    submodel["transform_layer"] = Trans1DTo2DLayerWith3Channel()
    submodel["selfann"] = ResNet("Bottleneck", [3, 4, 6, 3])
    submodel["last_layer"] = LastLinear()

    submodel["decode"] = Decoder(feature_out=feature_size)
    layer = nn.Sequential(submodel)
    
    output = layer(input_features)
    print("Model output:", output.shape)  # [batch_size, 1]
    # gg = output.detach().numpy()
    # tt = get_p(output)
    # tt1 = get_p1(output)
    breakpoint()

