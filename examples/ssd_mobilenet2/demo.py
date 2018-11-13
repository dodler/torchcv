1import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from torchcv.models.ssd import SSD512, SSDBoxCoder

print('Loading model..')
net = SSD512(num_classes=7).to(0)
net.load_state_dict(torch.load('/home/lyan/Documents/torchcv/weights/ckpt.pth')['net'])
net.eval()
box_coder = SSDBoxCoder(net)


print('Loading image..')
img = Image.open('/home/lyan/Documents/sample_uvb/make_phone_photo/make_phone_photo_1002.jpg')
ow = oh = 512
img = img.resize((ow, oh))
print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def predict(image):
    with torch.no_grad():
        x = transform(image).to(0)
        loc_preds, cls_preds = net(x.unsqueeze(0))
        boxes, labels, scores = box_coder.decode(
            loc_preds.squeeze().to(0), F.softmax(cls_preds.squeeze(), dim=1).to(0))
        x.detach().cpu()

        return boxes, labels, scores



cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('tset', frame)

    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    boxes, labels, scores = predict(frame)

    if cv2.waitKey(1) == 27:
        break  # esc to quit
