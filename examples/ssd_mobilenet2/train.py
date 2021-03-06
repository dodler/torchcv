from __future__ import print_function

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms

from torchcv.datasets import ListDataset
from torchcv.loss import SSDLoss
from torchcv.models.mobilenetv2.net import SSD300MobNet2
from torchcv.models.ssd import SSDBoxCoder
from torchcv.transforms import resize, random_flip, random_paste, random_crop, random_distort

LIST_FILE = '/home/lyan/Documents/torchcv/torchcv/datasets/uvb/uvb_train.txt'
IMGS_ROOT = '/home/lyan/Documents/sample_uvb/all_imgs'
NUM_CLASSES = 6 + 1  # ex 6+1, +1 is for background
DEVICE='cpu'
BATCH_SIZE = 1
NUM_WORKERS = 2

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='/home/lyan/Documents/torchcv/weights/fpnssd512_20_trained.pth', type=str,
                    help='initialized model path')
# parser.add_argument('--model', default='./examples/ssd/model/ssd512_vgg16.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='checkpoint/mobilenet2.pth', type=str, help='checkpoint path')
args = parser.parse_args()

# Model
print('==> Building model..')
net = SSD300MobNet2(num_classes=NUM_CLASSES)
net.to(DEVICE)
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 300


def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size, img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


trainset = ListDataset(root=IMGS_ROOT,
                       list_file=[LIST_FILE],
                       transform=transform_train)


def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


testset = ListDataset(root=IMGS_ROOT,
                      list_file=LIST_FILE,
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

cudnn.benchmark = True

criterion = SSDLoss(num_classes=NUM_CLASSES)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = inputs.to(DEVICE)
        loc_targets = loc_targets.to(DEVICE)
        cls_targets = cls_targets.to(DEVICE)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.item(), train_loss / (batch_idx + 1), batch_idx + 1, len(trainloader)))


# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = inputs.to(DEVICE)
        loc_targets = loc_targets.to(DEVICE)
        cls_targets = cls_targets.to(DEVICE)

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.item()
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.item(), test_loss / (batch_idx + 1), batch_idx + 1, len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
