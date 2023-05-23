import cv2
import sys
import torch
import utils
import numpy as np
# from model import mtcnn
# from model import faceNet
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
from model import mtcnn
from model import faceNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
face_Detector = mtcnn.MTCNN()
face_Detector.to(device)
faceNet = faceNet.facenet()
root = os.getcwd()
encoding_All = []
classes = []
for i in os.listdir(root):
    if os.path.isdir(i):
        classes.append(i)
        img_path = os.path.join(root, i)
        count = 0
        for j in os.listdir(img_path):
            img = cv2.imread(os.path.join(img_path, j))
            out = face_Detector.detect_face(img)
            img_face = utils.Face_Alignment(img, out)
            coding_128dim = faceNet.detect_img(img_face[0])
            if count == 0:
                encoding_Temp = coding_128dim
            else:
                encoding_Temp = np.vstack((encoding_Temp, coding_128dim))
            count += 1
        encoding_All.append(np.mean(encoding_Temp, axis=0))

output = [classes, encoding_All]
with open('encoding_All.pkl', 'wb') as f:
    pickle.dump(output, f)