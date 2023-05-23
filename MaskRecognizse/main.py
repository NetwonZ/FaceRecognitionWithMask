import cv2
import torch
import utils
import numpy as np
from model import mtcnn
from model import faceNet
import pickle
#判断GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
face_Detector = mtcnn.MTCNN()
face_Detector.to(device)
faceNet = faceNet.facenet()
with open("./model/PreTrain/encoding_All.pkl", "rb") as f:
    encoding_All = pickle.load(f)

# img = cv2.imread("./testimg/zxh/zxh_4.jpg")
# out = face_Detector.detect_face(img)
# utils.Draw_Rec(out,img)


cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    out = face_Detector.detect_face(img)
    if out[0].shape[1] == 0:
        cv2.imshow('img', img)
        cv2.waitKey(20)
        continue
    img_face = utils.Face_Alignment(img, out)
    encoding_128dim = faceNet.detect_img(img_face[0])
    dis = utils.cal_distance(encoding_128dim,encoding_All[1])
    name = encoding_All[0][np.argmin(dis)]
    cv2.putText(img,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    utils.Draw_Rec(out,img)
cv2.destroyAllWindows()
