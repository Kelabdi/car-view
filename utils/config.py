from os import listdir
import pandas as pd
import numpy as np

#from mrcnn.utils import Dataset, extract_bboxes
#from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN



class CarPredictionConfig(Config):
    NAME = "car_cfg"
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.98
