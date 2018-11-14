from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torchcv.models import SSD300
from torchcv.models.ssd.net import L2Norm


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


def from_state_dict(path='/home/lyan/Documents/torchcv/weights/mobilenet2/model_best.pth.tar'):
    state_dict = torch.load(path, map_location='cpu')['state_dict']
    net = MobileNet2Feature()
    net_dict = net.state_dict()

    state_dict = {k: v for k, v in state_dict.items() if k in net_dict}
    net_dict.update(state_dict)
    net.load_state_dict(net_dict)
    return net


class MobileNet2Feature(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNet2Feature, self).__init__()

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)

        self.num_of_channels = [32, 16, 24, 32]#, 64, 96, 160, 320]
        # assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3]#, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2]#, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        channels_ssd = 512
        self.conv2 = nn.Conv2d(kernel_size=1,
                               in_channels=self.num_of_channels[-1],
                               out_channels=channels_ssd)

        self.norm4 = L2Norm(512, 20) # 20 - ???

        self.conv5_1 = nn.Conv2d(channels_ssd, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        h = self.bottlenecks(x)
        h = F.relu(self.conv2(h))

        hs = []

        hs.append(self.norm4(h))  # conv4_3

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)  # conv7

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2
        return hs

        return hs


class SSD300MobNet2(SSD300):

    def __init__(self, num_classes):
        SSD300.__init__(self, num_classes)
        self.extractor = from_state_dict()
        self._init_layers()
