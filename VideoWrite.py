import os
import cv2
import numpy as np

fps=20

# size = (530, 266)
size = (265, 133)
videowriter = cv2.VideoWriter("trainingResult.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
path = './sample/'
for i in os.listdir(path):
    print(i)
    img = cv2.imread(path + i)
    img = cv2.resize(img,size)

    videowriter.write(img)

videowriter.release()

# fps=4

# size = (398, 266)
# videowriter = cv2.VideoWriter("GenerationResult2.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
# path = './generateSample/sample_weight_mixing/'
# for i in os.listdir(path):
#     print(i)
#     img = cv2.imread(path + i)
#     img = cv2.resize(img,size)

#     videowriter.write(img)

# videowriter.release()