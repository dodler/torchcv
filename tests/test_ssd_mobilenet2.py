import unittest
from tqdm import *
import torch

from torchcv.models import SSD300
from torchcv.models.mobilenetv2.net import SSD300MobNet2, MobileNet2Feature, from_state_dict


class TestSSDMobileNet2(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_inference(self):
        los, cls = SSD300(7)(torch.zeros(1, 3, 300, 300))
        print(los.shape, cls.shape)

        self.net = SSD300MobNet2(7).float()
        with torch.no_grad():
            loc, cls = self.net(torch.zeros(1, 3, 300, 300).float())
            print(loc.shape, cls.shape)
            # fixme, no assertion

    def test_load_weights(self):
        state_dict = torch.load('/home/lyan/Documents/torchcv/weights/mobilenet2/model_best.pth.tar',map_location='cpu')['state_dict']
        net = MobileNet2Feature()
        net_dict = net.state_dict()

        state_dict = {k:v for k,v in state_dict.items() if k in net_dict}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)

    def test_inference(self):
        net = from_state_dict()
        net.to(0)
        assert len(net(torch.zeros(1,3,300,300).to(0))) == 6
