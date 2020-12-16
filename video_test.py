import cv2

from datautils import prepare_dataset

arr = []
vidcap = cv2.VideoCapture("data/CW/Maul.mp4")
success, image = vidcap.read()
count = 0
i = 0

while success:
    if len(arr) > 10:
        break
    i += 1
    image = image[(720 - 544) // 2: (720 + 544) // 2, :, :]
    if i > 15 * 30:
        arr.append(image)
    # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    count += 1

import numpy as np

nn_shape = (544, 1280, 3)
arr = np.asarray(arr)

X, _ = prepare_dataset(arr, ((),) * len(arr), nn_shape[:2])
Y_pred = []

video_file = "data/video2.avi"
video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'XVID'), 60, (arr.shape[1], arr.shape[2]))

# print(Y_pred)
for i in range(arr.shape[0]):
    video.write(arr[i])

video.release()
print(arr.shape)