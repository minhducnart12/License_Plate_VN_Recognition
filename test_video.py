import cv2
from pathlib import Path
import argparse
import time
from imutils.video import VideoStream

from lp_recognition import E2E

import os

# def get_arguments():
#     arg = argparse.ArgumentParser()
#     arg.add_argument('-i', '--image_path', help='link to image', default='./samples/1.jpg')

#     return arg.parse_args()


# args = get_arguments()
# img_path = Path(args.image_path)
VIDEO_PATH = './img_test/test3.mp4'
out_path = './results'

# Declaring variables for video processing.
cap = cv2.VideoCapture(VIDEO_PATH)
codec = cv2.VideoWriter_fourcc(*'DIVX')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
file_name = os.path.join(out_path, 'output_' + VIDEO_PATH.split('/')[-1])
out = cv2.VideoWriter(file_name, codec, fps, (width,height))
model = E2E()
   
#   # Frame count variable.
ct = 0

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        print(ct)
 
        # Noting time for calculating FPS.
        prev_time = time.time()

        image = model.predict(img)

                  # Calculating time taken and FPS for the whole process.
        tot_time = time.time() - prev_time
        fps = 1/tot_time

        cv2.putText(img, 'frame: %d fps: %s' % (ct, fps),
                  (0, int(100 * 1)), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), thickness=2)
        cv2.imshow("vid_out", image)
        out.write(img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
          break
        ct = ct + 1
    else:
        break
      
cap.release()
out.release()
        
        ## closing all windows
cv2.destroyAllWindows()
