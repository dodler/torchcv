import numpy as np
import pickle
import unittest

import torch

from torchcv.models.mobilenetv2.net import MobileNet2Feature
from torchcv.models.ssd.net import VGG16Extractor300

VGG_EXTRACTOR_EXPECTED_OUTPUT = pickle.load(open('vgg_extractor.expected.pkl', 'rb'))


class TestExtractor(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.vgg_extractor = VGG16Extractor300()
        self.mobilenet_extractor = MobileNet2Feature()

    def test_vgg_extractor(self):
        expected_dim = [38, 19, 10, 5, 3, 1]
        expected_filters = [512, 1024, 512, 256, 256, 256]
        with torch.no_grad():
            feat = self.vgg_extractor(torch.zeros((1, 3, 300, 300)))
            for i, f in enumerate(feat):
                assert f.shape[2] == expected_dim[i]
                assert f.shape[1] == expected_filters[i]

    def test_mobilenet2_extractor_shape(self):
        expected_filters = [320, 640, 320, 160, 160, 160]
        expected_dim = (10, 10, 8, 6, 4, 1)

        with torch.no_grad():
            result = self.mobilenet_extractor(torch.zeros(1, 3, 300, 300))
            for i in range(len(expected_filters)):
                assert result[i].shape[1] == expected_filters[i]
                assert result[i].shape[2] == expected_dim[i]
