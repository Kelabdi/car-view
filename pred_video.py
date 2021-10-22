import cv2
import time
from os import listdir
from utils.config import CarPredictionConfig, MaskRCNN
from utils.functions import adapt_image, prediction
import warnings
warnings.filterwarnings("ignore")



filename = "ferrari.mp4"

def play_video(filename):
    cap = cv2.VideoCapture("videos/" + filename)
    while True:
        success, img = cap.read()
        if success==True:
            img = cv2.resize(img, (960,540))
            cv2.imshow("Video", img)
            cv2.waitKey(20)
        else:
            break
    
    

def save_video(filename):
    # Spliting the video into frames
    cap = cv2.VideoCapture("videos/" + filename)
    i = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (960,540))
        cv2.imwrite("frames/" + filename + f"/{i}.jpg", img)
        i+=1
        
def show_pred_video(filename):
    path = "predictions/"
    # AQUI HAY QUE MONTAR UNA FUNCION BONITA
    ptime = 0
    for filename in listdir(path):
        # Put FPS Counter
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img, f"FPS: {int(fps)}", (850,50), cv2.FONT_HERSHEY_PLAIN, 
                1.5, (0,255,0),2)
        # ploting prediction
        # plot_video_pred(img, sample, model)
        img = cv2.imread(path + "/" + filename + f"/{i}.jpg")
        cv2.imshow("Prediction", img)
        cv2.waitKey(20)

def pred_video(filename):
    path = "frames/" + filename
    # Loading prediction model
    predconfig = CarPredictionConfig()
    model = MaskRCNN(mode="inference", model_dir="model/pred/", config=predconfig)
    model.load_weights("model/mask_rcnn_car_0009.h5", by_name=True)
    for name in listdir(path):
        img = cv2.imread(path + f"/{i}.jpg")
        sample = adapt_image(img, predconfig)
        prediction(img, sample, model)
    return "ok"


def pred_image(filename):
    Path = "images/"
    # Loading prediction model
    predconfig = CarPredictionConfig()
    model = MaskRCNN(mode="inference", model_dir="model/pred/", config=predconfig)
    model.load_weights("model/mask_rcnn_car_0009.h5", by_name=True)    
    # predicting boxes
    img = cv2.imread(Path + filename)
    sample = adapt_image(img, predconfig)
    prediction(img, sample, model)

def show_pred_image(filename):
    Path = "images"


#Presentacion
#sacar metricas
#probar con videos