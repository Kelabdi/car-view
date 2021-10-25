import sys
from main import show_pred_video


print("Loading video...")
filename = sys.argv[1]

show_pred_video(filename)