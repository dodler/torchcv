import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from torchcv.models import SSD512
from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models.ssd import SSD300, SSDBoxCoder

from PIL import Image

NUM_CLASSES = 6 + 1  # ex 6+1, +1 is for background

BATCH_SIZE = 1

print('Loading model..')
net = SSD512(num_classes=NUM_CLASSES)
net.load_state_dict(torch.load('/home/lyan/Documents/torchcv/weights/ckpt.pth')['net'])
net.cuda()
net.eval()

print('Preparing dataset..')
img_size = 512


def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    return img, boxes, labels


dataset = ListDataset(root='/home/lyan/Documents/sample_uvb/all_imgs', \
                      list_file='/home/lyan/Documents/torchcv/torchcv/datasets/uvb/uvb_train.txt',
                      transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
box_coder = SSDBoxCoder(net)

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

with open('/home/lyan/Documents/torchcv/torchcv/datasets/uvb/uvb_train.txt') as f:
    gt_difficults = []
    for line in f.readlines():
        line = line.strip().split()
        d = [int(x) for x in line[1:]]
        gt_difficults.append(d)


def eval(net):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.01)

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    print(voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=True))


eval(net)
