import sys
from main import display_image, pred_image

print("Predicting...")
filename = sys.argv[1]

display_image(filename)
pred_image(filename)