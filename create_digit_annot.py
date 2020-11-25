import numpy as np
import cv2
import os
import pandas as pd
import h5py
import pickle

def get_img_name(idx, f):
    names = f['digitStruct']['name']
    # transform object to string
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return img_name


def get_img_boxes(idx, f):
    # read idx-th bbox
    bboxs = f['digitStruct']['bbox']
    box = f[bboxs[idx][0]]

    # add every key into label and bbox numpy array
    # shape: (n,4), n is number of digits in idx-th image
    n = box['label'].shape[0]
    bbox = np.zeros((n,4),dtype=float)
    # shape: (n)
    label = np.zeros(n,dtype=int)

    if n == 1:
        label[0] = int(box['label'][0][0])

        for attr, key in enumerate(['left', 'top', 'width', 'height']):
            bbox[0][attr] = int(box[key][0][0])

    else:
        for i in range(n):
            label[i] = int(f[box['label'][i][0]][()].item())

            for attr, key in enumerate(['left', 'top', 'width', 'height']):
                bbox[i][attr] = int(f[box[key][i][0]][()].item())

    # let bbox be [left, top, right, bottom]
    # right = left + width
    bbox[:,2] = bbox[:,0] + bbox[:,2]
    # bottom = top + height
    bbox[:,3] = bbox[:,1] + bbox[:,3]

    return (label, bbox)


def create_img_label_bbox_dict(mat_file):
    # open mat file
    f = h5py.File(mat_file,'r')

    data_dict = {}
    for j in range(f['/digitStruct/bbox'].shape[0]):
        # read image name, label and bbox, and add into dict 
        # data_dict['img_name'] = (label, bbox)
        img_name = get_img_name(j, f)
        attr = get_img_boxes(j, f)
        data_dict[img_name] = attr

    return data_dict


data_dict = create_img_label_bbox_dict("../train/digitStruct.mat")


f = open("digit_annotation.pkl","wb")
pickle.dump(data_dict,f)
f.close()
