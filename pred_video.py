import cv2
import time
from utils.config import CarPredictionConfig, MaskRCNN
from utils.functions import adapt_image, plot_video_pred


# Loading prediction model
predconfig = CarPredictionConfig()
model = MaskRCNN(mode="inference", model_dir="model/pred/", config=predconfig)
model.load_weights("model/mask_rcnn_car_0009.h5", by_name=True)


# Spliting the video into frames
cap = cv2.VideoCapture("videos/ferrari.mp4")
ptime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (960,540))
    sample = adapt_image(img, predconfig)
    
    # Put FPS Counter
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(sample, f"FPS: {int(fps)}", (850,50), cv2.FONT_HERSHEY_PLAIN, 
            1.5, (0,255,0),2)
    # ploting prediction
    plot_video_pred(sample, model)
    
