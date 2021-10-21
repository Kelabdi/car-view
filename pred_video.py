import cv2
import time
from utils.config import CarPredictionConfig, MaskRCNN
from utils.functions import adapt_image, plot_video_pred
import warnings
warnings.filterwarnings("ignore")

ptime = 0

filename = "ferrari.mp4"

def save_video(filename):
    # Spliting the video into frames
    cap = cv2.VideoCapture("videos/" + filename)
    i = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (960,540))
		cv2.imwrite("prediction/" + filename + f"/{i}.jpg", img)
		i+=1
        

		# AQUI HAY QUE MONTAR UNA FUNCION BONITA
        # Put FPS Counter
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img, f"FPS: {int(fps)}", (850,50), cv2.FONT_HERSHEY_PLAIN, 
                1.5, (0,255,0),2)
        # ploting prediction
        plot_video_pred(img, sample, model)
        cv2.imshow("Prediction", img)
        cv2.waitKey(1)

def pred_(path):
	# Loading prediction model
	predconfig = CarPredictionConfig()
	model = MaskRCNN(mode="inference", model_dir="model/pred/", config=predconfig)
	model.load_weights("model/mask_rcnn_car_0009.h5", by_name=True)
	
	sample = adapt_image(img, predconfig)

	for filename in listdir(path):
        image_id = filename


