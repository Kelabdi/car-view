import numpy as np
from mrcnn.model import mold_image



def adapt_image(image, config):
    scaled_image = mold_image(image, config)
    sample = np.expand_dims(scaled_image, 0)
    return sample


def prediction(sample, model):
	pbox = model.detect(sample, verbose=0)[0]
	# save each box
	boxes = []
	try:
		for box in pbox['rois']:
			y1, x1, y2, x2 = box
			boxes.append([int(x1),int(y1),int(x2),int(y2)])
	except:
		boxes.append([None,None,None,None])
	return boxes 