import torch
from torch.utils.data import Dataset
import glob
import os
import pickle
from PIL import Image
from util import transform


class DigitDataset(Dataset):
    def __init__(self, data_folder, is_train, split):
        self.data_folder = data_folder
        self.is_train = is_train

        self.annot_dict = {}
        with open("digit_annotation.pkl", 'rb') as open_file:
            self.annot_dict = pickle.load(open_file)

        image_list = glob.glob(os.path.join(data_folder,"*.png"))
        if split == True:
            if is_train == True:
                self.images = sorted(image_list)[:30000]
            else:
                self.images = sorted(image_list)[30000:]
        else:
            self.images = image_list

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i])
        image = image.convert('RGB')
        img_name = os.path.basename(self.images[i])

        # Read objects in this image (bounding boxes, labels)
        # (n_objects), (n_objects, 4)
        (labels, boxes) = self.annot_dict[img_name]
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, is_train=self.is_train)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["img_name"] = img_name

        return image, target


    def __len__(self):
        return len(self.images)


    def collate_fn(self, batch):
        images = list()
        targets = list()
        
        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        return images, targets


def get_test_img(idx):
    img_filename = "test/{}.png".format(idx)

    image = Image.open(img_filename)
    image = image.convert('RGB')

    image, _, _ = transform(image)

    return image

def get_speed_test_img(idx):
    img_filename = "for_speed_test/{}.png".format(idx)

    image = Image.open(img_filename)
    image = image.convert('RGB')

    image, _, _ = transform(image)

    return image

