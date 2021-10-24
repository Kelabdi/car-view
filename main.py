import cv2
import time
import json
from os import chdir, getcwd, listdir, path, mkdir
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
    Path = "frames/" + filename + "/"
    chdir("frames")
    mkdir(f"{filename}")
    chdir(f"{filename}")
    print(getcwd())
    if cap.isOpened():
        while True:
            success, img = cap.read()
            if success==True:
                img = cv2.resize(img, (960,540))
                file = path.join(Path , f"{i}.jpg")
                print(file)
                cv2.imwrite(f"{i}.jpg", img)
                print("created")
                i+=1
            else:
                break
        

def show_pred_video(filename):
    Path = "frames/" + filename
    
    # Open Json with predictions
    with open(Path + "/pred_boxes.json", 'r') as f:
        boxes = json.load(f)
    print("json loaded")
    # AQUI HAY QUE MONTAR UNA FUNCION BONITA
    ptime = 0
    fr_num = len([name for name in listdir(Path)]) - 1
    print(fr_num)

    #for i in range(fr_num):
    i = 0
    while i<fr_num:
        img = cv2.imread(Path + f"/{i}.jpg")
        # Put FPS Counter
        #ctime = time.time()
        #fps = 1/(ctime - ptime)
        #ptime = ctime
        #cv2.putText(img, f"FPS: {int(fps)}", (850,50), cv2.FONT_HERSHEY_PLAIN, 
        #        1.5, (0,255,0),2)
        # ploting prediction
        # plot_video_pred(img, sample, model)
        if i%10 == 0:
            try:
                for box in boxes[i]:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255), 2)
                    print("rect" + "%"*20)
            except:
                pass
        cv2.imshow("Prediction", img)
        cv2.waitKey(20)
        i+=1
        

def pred_video(filename):
    path = "frames/" + filename
    # Loading prediction model
    predconfig = CarPredictionConfig()
    model = MaskRCNN(mode="inference", model_dir="model/pred/", config=predconfig)
    model.load_weights("model/mask_rcnn_car_0009.h5", by_name=True)
    print("model loaded "+"%"*20)
    pred_list = []
    # Percentage for data predicted
    per100 = len([name for name in listdir(path)])
    p = 1

    for name in listdir(path):
        if p%10 == 0:  
            img = cv2.imread(path + f"/{name}")
            sample = adapt_image(img, predconfig)
            boxes = prediction(sample, model)
            pred_list.append(boxes)
        print("predicting... " + str(int(100*p/per100)) + " %")
        p+=1
    with open(path + "/pred_boxes.json", 'w') as f:
        json.dump(pred_list, f, indent=2)    
    return pred_list


def pred_image(filename):
    Path = "images/"
    # Loading prediction model
    predconfig = CarPredictionConfig()
    model = MaskRCNN(mode="inference", model_dir="model/pred/", config=predconfig)
    model.load_weights("model/mask_rcnn_car_0009.h5", by_name=True)    
    # predicting boxes
    img = cv2.imread(Path + filename)
    sample = adapt_image(img, predconfig)
    boxes = prediction(sample, model)
    try:
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255), 2)
            print("rect" + "%"*20)
    except:
        pass
    cv2.imshow("Prediction", img)
    




#Presentacion
#sacar metricas
#probar con videos