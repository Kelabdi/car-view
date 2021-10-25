import sys
from main import pred_image

print("Predicting...")
filename = sys.argv[1]
pred_image(filename)