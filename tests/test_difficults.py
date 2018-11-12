import unittest


class TestDifficults(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_load_difficults(self):
        with open('/home/lyan/Documents/torchcv/torchcv/datasets/voc/voc07_test_difficult.txt') as f:
            gt_difficults = []
            for line in f.readlines():
                line = line.strip().split()
                d = [int(x) for x in line[1:]]
                gt_difficults.append(d)

        print(gt_difficults)

