import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models import *
from util import *
from dataset import DigitDataset


min_size = 400
max_size = 600
batch_size = 12
epochs = 15
workers = 4
print_freq = 50
lr = 0.001
milestones = [10, 13]
gamma = 0.1
momentum = 0.9
weight_decay = 5e-4
device = "cuda:0"
torch.cuda.set_device(0)


train_dataset = DigitDataset(data_folder="train", is_train=True, split=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers)
valid_dataset = DigitDataset(data_folder="train", is_train=False, split=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                               collate_fn=valid_dataset.collate_fn, num_workers=workers)


anchor_sizes = ((32,), (64,), (96,), (128,), (160,))
aspect_ratios = ((0.33, 0.5, 0.67),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                             rpn_anchor_generator=anchor_generator,
                                                             min_size=min_size,
                                                             max_size=max_size)

num_classes = 11
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


backbone = CSPWithFPN()
model.backbone = backbone

model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


# training
for epoch in range(1, epochs+1):
    model.train()
    start = time.time()
    loss_value = 0
    for i, (images, targets) in enumerate(train_dataloader):
        # Move to default device
        images = list(image.to(device) for image in images)
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]

        # Forward prop.
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # Print status
        if (i+1) % print_freq == 0:
            end = time.time()
            print('Epoch: [{:>6d}][{:>6d}/{:>6d}]\tTime: {:.3f}\t'
                  'Loss: {:.4f}\t'.format(epoch, (i+1)*batch_size,len(train_dataset),end-start, loss_value/print_freq))
            start = time.time()
            loss_value = 0

    scheduler.step()

    model.eval()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    with torch.no_grad():
        val_start = time.time()
        for i, (images, targets) in enumerate(valid_dataloader):
            images = list(image.to(device) for image in images)

            output = model(images)

            # Store this batch's results for mAP calculation
            boxes = [t['boxes'].to(device) for t in targets]
            labels = [t['labels'].to(device) for t in targets]


            det_boxes.extend([o['boxes'] for o in output])
            det_labels.extend([o['labels'] for o in output])
            det_scores.extend([o['scores'] for o in output])
            true_boxes.extend(boxes)
            true_labels.extend(labels)

        # Calculate mAP
        _mAP = mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, device)
        val_end = time.time()
        print("valid mAP: {:.4f}, time: {:.3f}".format(_mAP, val_end - val_start))


torch.save(model, "fasterRcnn.pth")
