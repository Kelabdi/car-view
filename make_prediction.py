import sys
from main import save_video, pred_video, show_pred_video


print("Loading video...")
filename = sys.argv[1]

#save_video(filename)
#pred_video(filename)
show_pred_video(filename)