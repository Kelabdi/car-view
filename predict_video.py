import sys
from main import save_video, pred_video, show_pred_video


print("Loading video...")
filename = sys.argv[1]

print("Turning video into Frames...")
#save_video(filename)
print("Makng prediction fo each Frame...")
pred_video(filename)
print("Displaying prediction video...")
show_pred_video(filename)