# NCTU Selected Topics in Visual Recognition using Deep Learning, Homework 2
Code for Object Detection on Street View House Numbers.


## Hardware
The following specs were used to create the submited solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- NVIDIA GeForce 2080Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#Installation)
2. [Dataset Download](#Dataset-Download)
3. [Prepapre Dataset](#Prepare-Dataset)
4. [Train models](#Train-models)
5. [Pretrained models](#Pretrained-models)
6. [Reference](#Reference)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hw2 python=3.7
source activate hw2
pip install -r requirements.txt
```

## Dataset Download
Dataset download link is in Data section of [HW2](https://drive.google.com/drive/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl)

## Prepare Dataset
After downloading, the data directory is structured as:
```
${ROOT}
  +- test
  |  +- 1.png
  |  +- 2.png
  |  +- ...
  +- train
  |  +- 1.png
  |  +- 2.png
  |  +- ...
  |  +- digitStruct.mat

```

Then, run `create_digit_annot.py` to decode `digitStruct.mat`, and get `digit_annotation.pkl`

```
$ python create_digit_annot.py
```


### Train models
To train models, run following commands.
```
$ python train.py 
```

Also, you could use jupyter notebook to open`find_Anchor.py`, and find the proper anchor box sizes in Street View House Numbers dataset.


## Pretrained models
You can download pretrained model that used for my submission from [link](https://drive.google.com/drive/folders/1HtXV0BdxCzo-E7XkbL7K8-Ti_7wHfNSm?usp=sharing).
And put it in the directory :
```
${ROOT}
  +- fasterRcnn.pth
  +- eval.py
  +- ...
```

Could use `jupyter notebook` to open `visualization.ipynb` and visualize the output of pretrained model and write the output into json file.


## Reference
1. [torchvision](https://github.com/pytorch/vision)
2. [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
3. [Cross-Stage Partial Networks](https://arxiv.org/abs/1911.11929)