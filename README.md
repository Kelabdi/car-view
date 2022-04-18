<h1 align="center">
    <img src="img/logo.png" />
</h1>

This is my Final Project for the **Big Data & Machine Learning Bootcamp** I am taking at Core CODE School.

**Car view Project**: Predicting model for car detection from images and videos. 

___

## Introduction

This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates bounding boxes for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

Starting from the dataset "Car Object Detection" from [kaggle](https://www.kaggle.com/sshikamaru/car-object-detection), we used the Mask R-CNN model to make a predicting model focused on car detection in images and videos.

Then some functions were added in order to make predictions directly with new images and videos.

## Installation

It's recommended to use a new virtual environment to install this dependencies. This environment must be created with *Python 3.6* version.

Use the package manager [pip](https://pypi.org/project/pip/) to install the requirements.txt 

```bash
pip install -r requirements.txt
```

## Usage

First you must consider clone [this repository]() in order to install the correct version of mrcnn==2.1. 

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

You can also download pre-trained COCO weights (mask_rcnn_coco.h5) from the [Mask_RCNN releases page](https://github.com/matterport/Mask_RCNN/releases) in order to start with better results in your model training.

With your own model already trained, you can store it in "/model" path and start using the dedicated functions.

---

This **functions** have default reading file paths:

    videos -> /videos
    images -> /images

> **NOTE:** < filename > must be always wrote with extension -> *.jpg, *.mp4 ...

Usage of functions: from terminal and in root path of the repo:


+ Play original video

```bash
python play_video.py <filename>
```

+ Play predicted video

```bash
python play_predicted_video.py <filename>
```

+ Predict image

```bash
python predict_image.py <filename>
```

+ Predict video

```bash
python predict_video.py <filename>
```

## Examples

<p align="center">
    <img src="/img/pred1.png" />
</p>

<p align="center">
    <img src="/img/pred3.png" />
</p>

<p align="center">
    <img src="/img/pred2.png" />
</p>

## Contribuiting

You can send a pull-request if you want to share some improvements/ideas.

## References

Thanks for this tools and free access info to:

[Matterport](https://github.com/matterport/Mask_RCNN) (Mask R-CNN repository)

[Jason Brownlee](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/) (very helpful post)

## Thanks to

###### [CORE Code School](https://github.com/core-school) for all the knowledge, the insights and inspiration they have given during the course.

