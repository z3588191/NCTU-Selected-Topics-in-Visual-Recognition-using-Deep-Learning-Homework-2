import numpy as np
import pandas as pd
import glob
import os
from PIL import Image


min_size = 400
max_size = 600


def change_to_wh(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return [width, height]


# function from Tensorflow Object Detection API to resize image
def compute_resize_scale(width, height, min_size=min_size, max_size=max_size):
    old_width = width
    old_height = height

    # resize according to min size
    old_short_size = min(old_width, old_height)
    min_scale = min_size / float(old_short_size)
    new_width_min = int(round(old_width * min_scale))
    new_height_min = int(round(old_height * min_scale))
    new_size_min = [new_width_min, new_height_min]

    # resize according to max size
    old_long_size = max(old_height, old_width)
    max_scale = max_size / float(old_long_size)
    new_width_max = int(round(old_width * max_scale))
    new_height_max = int(round(old_height * max_scale))
    new_size_max = [new_width_max, new_height_max]

    if max(new_size_min) > max_size:
        resize_scale = max_scale
    else:
        resize_scale = min_scale

    return resize_scale


dic = pd.read_pickle("digit_annotation.pkl")
train_list = glob.glob("train/*.png")
all_boxes = []

for img_name in train_list:
    key = os.path.basename(img_name)
    _, boxes = dic[key]
    image = Image.open(img_name)
    (width, height) = image.size
    resize_scale = compute_resize_scale(width, height)
    
    for box in boxes:
        box = change_to_wh(box)
        box = [round(box[0]*resize_scale),round(box[1]*resize_scale)]
        all_boxes.append(box)

all_boxes = np.asarray(all_boxes)


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # random choose k boxes as initial cluster centers
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # E step
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)
        
        # check converge condition
        if (last_clusters == nearest_clusters).all():
            break

        # M step, choose median of widths and heights in same group as new cluster centers
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


# use K-means find 100 proper widths and heights of bbox
# with IOU metric
centers = kmeans(all_boxes, 100)


# compute size and aspect ratio of bbox found above
sizes = np.min(centers,axis=1).reshape(-1,1)
aspect_ratios = (centers[:,0] / centers[:,1]).reshape(-1,1)


# use K-means find proper 5 sizes
from sklearn.cluster import KMeans
K = KMeans(5, random_state=1)
proper_sizes = K.fit(sizes)
print("proper anchor size:")
print(np.round(proper_sizes.cluster_centers_))


# use K-means find proper 3 aspect ratios
K = KMeans(3, random_state=1)
proper_ratios = K.fit(aspect_ratios)
print("proper anchor ratio:")
print(proper_ratios.cluster_centers_)
