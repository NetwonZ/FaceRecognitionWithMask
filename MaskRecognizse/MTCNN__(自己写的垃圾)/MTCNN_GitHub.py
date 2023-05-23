import os

import cv2
import numpy as np
import tool
import torch
from torch import nn


class PNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)
        self.training = False
        if pretrained:#导入训练好的权重文件
            state_dict_path = 'modeldata/pnet.pt'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            state_dict_path = 'modeldata/rnet.pt'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN__(自己写的垃圾) ONet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = './modeldata/onet.pt'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a

def detect(img,threshold = [0.5,0.6,0.7]):
    img_copy = (img - 127.5) / 127.5
    origin_h ,origin_w = img_copy.shape[0],img_copy.shape[1]
    scales = []
    scales = tool.CalculateScales(img_copy)
    p_rec = []
    p_prob = []
    for scale in scales:
        sw = int(origin_w*scale)
        sh = int(origin_h*scale)
        scale_img = cv2.resize(img_copy,(sw,sh))
        scale_img = tool.cv2torch(scale_img)

        rec , prob = pnet(scale_img)

        p_rec.append(rec)
        p_prob.append(prob)
    rectangles = []
    for i in range(len(scales)):
        sca_prob = p_prob[i][0][1,:,:]#torch.Size([95, 90])
        sca_rec = p_rec[i][0]  # torch.Size([4, 95, 90])
        sca_h, sca_w = sca_prob.shape[0], sca_prob.shape[1]  # 缩放后的图片大小
        outside = max(sca_h,sca_w)
        rectangle = tool.face_12net(sca_prob,sca_rec,outside,1/scales[i],origin_w,origin_h,threshold[0])
        # rectangle = rectangle[0,:,:]
        if rectangle.shape[1] != 0:
            rectangles.append(rectangle)

    rectangles = torch.cat(rectangles,dim=1)[0,:]


    # RNet Part
    predict_24_batch = []
    for rectangle in rectangles:
        img_copy_2 = img_copy[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2]),:]
        try:
            scale_img = cv2.resize(img_copy_2,(24,24))
            predict_24_batch.append(scale_img)
        except:
            print("error:")
            print(rectangle)
            print('_'*100)
            continue

    for i in range(len(predict_24_batch)):
        predict_24_batch[i] = tool.cv2torch(predict_24_batch[i])

    predict_24_batch = torch.cat(predict_24_batch, dim=0)

    rec , prob = rnet(predict_24_batch)

    sca_prob = prob[:,1]
    sca_rec = rec

    rectangles = tool.filter_face_24net(sca_prob,sca_rec,rectangles,origin_w,origin_h,threshold[1])

    if rectangles.shape[1] == 0:
        return rectangles


    # ONet Part

    rectangles = rectangles[0,:,:]
    predict_batch = []
    for rectangle in rectangles:
        x1,y1,x2,y2,score = rectangle
        img_copy_3 = img_copy[int(y1):int(y2),int(x1):int(x2)]
        scale_img = cv2.resize(img_copy_3,(48,48))
        predict_batch.append(scale_img)

    for i in range(len(predict_batch)):
        predict_batch[i] = tool.cv2torch(predict_batch[i])
    predict_batch = torch.cat(predict_batch, dim=0)

    roi_prob,pts_prob,cls_prob = onet(predict_batch)

    rectangles = tool.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

    return rectangles















pnet = PNet()
rnet = RNet()
onet = ONet()

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     rectangles = detect(frame)
#     rectangles = rectangles[0,:,:]
#     for rectangle in rectangles:
#         x1,y1,x2,y2 = int(rectangle[0]),int(rectangle[1]),int(rectangle[2]),int(rectangle[3])
#         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
#         pstx1,psty1,pstx2,psty2,pstx3,psty3,pstx4,psty4,pstx5,psty5 = int(rectangle[5]),int(rectangle[6]),int(rectangle[7]),int(rectangle[8]),int(rectangle[9]),int(rectangle[10]),int(rectangle[11]),int(rectangle[12]),int(rectangle[13]),int(rectangle[14])
#         cv2.circle(frame,(pstx1,psty1),2,(0,0,255),2)
#         cv2.circle(frame,(pstx2,psty2),2,(0,0,255),2)
#         cv2.circle(frame,(pstx3,psty3),2,(0,0,255),2)
#         cv2.circle(frame,(pstx4,psty4),2,(0,0,255),2)
#         cv2.circle(frame,(pstx5,psty5),2,(0,0,255),2)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(10) == ord('q'):
#         break

img = cv2.imread('sg.jpg')
rectangles = detect(img)
rectangles = rectangles[0,:,:]
for rectangle in rectangles:
    x1,y1,x2,y2 = int(rectangle[0]),int(rectangle[1]),int(rectangle[2]),int(rectangle[3])
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    pstx1,psty1,pstx2,psty2,pstx3,psty3,pstx4,psty4,pstx5,psty5 = int(rectangle[5]),int(rectangle[6]),int(rectangle[7]),int(rectangle[8]),int(rectangle[9]),int(rectangle[10]),int(rectangle[11]),int(rectangle[12]),int(rectangle[13]),int(rectangle[14])
    cv2.circle(img,(pstx1,psty1),2,(0,0,255),2)
    cv2.circle(img,(pstx2,psty2),2,(0,0,255),2)
    cv2.circle(img,(pstx3,psty3),2,(0,0,255),2)
    cv2.circle(img,(pstx4,psty4),2,(0,0,255),2)
    cv2.circle(img,(pstx5,psty5),2,(0,0,255),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()