import numpy as np

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from torchcv.models import SSD300
from torchcv.models.ssd import SSDBoxCoder

print('Loading model..')
net = SSD300(num_classes=7)
net.load_state_dict(torch.load('/home/lyan/Documents/torchcv/weights/ssd300/ckpt.pth')['net'])
net.to(0)
net.eval()
box_coder = SSDBoxCoder(net, 0)

print('Loading image..')
img = Image.open('/home/lyan/Documents/sample_uvb/make_phone_photo/make_phone_photo_1002.jpg')
ow = oh = 300
img = img.resize((ow, oh))
print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

names = ["classic_phone", "make_phone_photo",
         "phone_reading",
         "phone_talking",
         "middle_finger",
         "prof_make_photo"]

colors = []
for n in names:
    colors.append(np.random.uniform(0, 255, size=3).astype(np.int).tolist())


def predict(image):
    with torch.no_grad():
        x = transform(image).to(0)
        loc_preds, cls_preds = net(x.unsqueeze(0))
        boxes, labels, scores = box_coder.decode(
            loc_preds.squeeze(), F.softmax(cls_preds.squeeze(), dim=1))
        x.detach().cpu()

        return boxes, labels, scores


cap = cv2.VideoCapture(0)


def print_labels(labels, scores):
    result = ''
    for i, l in enumerate(labels):
        result += ("{" + names[l] + ":" + str(scores[i].item()) + "},")

    print(result)


def plot_boxes(frame_cpy, boxes, labels, scores):
    for i, bb in enumerate(boxes):
        sh = frame_cpy.shape

        x1 = int(((bb[0] / 300) * sh[0]).item())
        y1 = int(((bb[1] / 300) * sh[1]).item())

        x2 = int(((bb[2] / 300) * sh[0]).item())
        y2 = int(((bb[3] / 300) * sh[1]).item())

        cv2.rectangle(frame_cpy, (x1, y1), (x2, y2), color=colors[i], thickness=2)

        cv2.putText(frame_cpy, names[labels[i].item()], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame_cpy, str(scores[i].item()), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)


while True:
    _, frame = cap.read()

    frame_cpy = frame.copy()

    frame = cv2.resize(frame, (oh, ow))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    boxes, labels, scores = predict(frame)

    boxes = [bb.detach().cpu() for bb in boxes]
    labels = [l.detach().cpu() for l in labels]
    scores = [s.detach().cpu() for s in scores]

    plot_boxes(frame_cpy, boxes, labels, scores)
    cv2.imshow('tset', frame_cpy)

    if cv2.waitKey(1) == 27:
        break  # esc to quit
