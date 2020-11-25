import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from models import *
from dataset import get_test_img


output_list = list()
device = "cuda:1"
model = torch.load("fasterRcnn.pth").to(device)

model.eval()
for idx in range(13068):
    dic = {}
    image = get_test_img(idx+1).unsqueeze(0).to(device)
    output = model(image)

    boxes = output[0]["boxes"].to('cpu').detach().numpy()
    labels = output[0]["labels"].to('cpu').detach().numpy()
    scores = output[0]["scores"].to('cpu').detach().numpy()

    # ignore the bbox whose score is less than 0.5
    boxes = boxes[scores > 0.5]
    labels = labels[scores > 0.5]
    scores = scores[scores > 0.5]

    boxes = np.around(boxes).astype(int)

    for i in range(len(boxes)):
        boxes[i] = [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]]

    dic["bbox"] = boxes.tolist()
    dic["label"] = labels.tolist()
    dic["score"] = scores.tolist()

    output_list.append(dic)

with open('0856095.json', 'w') as jsonfile:
    json.dump(output_list, jsonfile)
