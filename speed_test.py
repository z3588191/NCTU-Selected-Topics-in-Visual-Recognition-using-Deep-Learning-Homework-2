import os
import time
import numpy as np

import torch
import torchvision
from models import *
from dataset import get_speed_test_img


output_list = list()
device = "cuda:1"
model = torch.load("fasterRcnn.pth").to(device)

model.eval()
start = time.time()
for idx in range(10):
    image = get_speed_test_img(idx+1).unsqueeze(0).to(device)
    output = model(image)
end = time.time()

print('{:.3f} seconds per image.'.format((end - start) / 10))