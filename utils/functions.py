import numpy as np
import cv2
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image



def adapt_image(image, config):
    scaled_image = mold_image(image, config)
    sample = np.expand_dims(scaled_image, 0)
    return sample


def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
# load image and mask
	for i in range(n_images):
		id = np.random.randint(0,len(dataset.image_ids))
# load the image and mask
		image = dataset.load_image(id)
		mask, _ = dataset.load_mask(id)
# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
# convert image into one sample
		sample = np.expand_dims(scaled_image, 0)
# make prediction
		yhat = model.detect(sample, verbose=0)[0]
# define subplot
		plt.figure(figsize=(60,60))
		plt.subplot(n_images, 2, i*2+1)
# plot raw pixel data
		plt.imshow(image)
		plt.title('Actual')
# plot masks
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# get the context for drawing boxes
		plt.subplot(n_images, 2, i*2+2)
# plot raw pixel data
		plt.imshow(image)
		plt.title('Predicted')
		ax = plt.gca()
# plot each box
		for box in yhat['rois']:
# get coordinates
			y1, x1, y2, x2 = box
# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
# draw the box
			ax.add_patch(rect)
# show the figure
	plt.show()


def plot_video_pred(img, sample, model):
	yhat = model.detect(sample, verbose=0)[0]
	# plot each box
	try:
		for box in yhat['rois']:
			y1, x1, y2, x2 = box
			cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255), 2)
	except:
		pass
