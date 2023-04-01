import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.ops

def drawRec(img,rec,color = (0,255,0)):
    for i in range(len(rec)):
        cv2.rectangle(img,(int(rec[i][0]),int(rec[i][1])),(int(rec[i][2]),int(rec[i][3])),color,1)

def cv2torch(input):
    input = torch.from_numpy(input).permute(2,0,1).float()
    input = input.reshape(1,*input.shape)
    return input


def Dense(input, output_channels,softmax = True):
    x = nn.Linear(input.shape[0], 512)(input)
    x = nn.PReLU()(x)
    x = nn.Linear(512, 256)(x)
    x = nn.PReLU()(x)
    x = nn.Linear(256, output_channels)(x)
    if softmax:
        x = nn.Softmax(dim=0)(x)
    return x

def CalculateScales(input,wide = 500):
    h,w = input.shape[0],input.shape[1]
    p_scale = 1
    if min(w,h) >wide:
        p_scale = wide/min(w,h)
        w = int(w*p_scale)
        h = int(h*p_scale)
    if max(w,h) < wide:
        p_scale = wide/max(w,h)
        w = int(w*p_scale)
        h = int(h*p_scale)
    scales = []
    factor = 0.709
    factor_count = 0
    minLen = min(w,h)
    while minLen >= 12:
        scales.append(p_scale*pow(factor,factor_count))
        minLen = minLen*factor
        factor_count += 1
    return scales#返回的是一个list

def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12
    #([1, 4, 95, 90])->([4, 1, 95, 90])
    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)# bb存的是所有满足阈值的网格的坐标
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def face_12net(prob,rec,outside,scale,width,height,threshold):
    stride = 0

    #stride 约等于 2
    if outside != 1:
        stride = float(2*outside-1)/(outside-1)
    mask = prob >=threshold
    bbox = mask.nonzero()
    bbox[:,[0,1]] = bbox[:,[1,0]] #(x,y)
    (x,y) = (bbox[:,0][:],bbox[:,1][:])

    #bbox 是大于阈值的网格坐标
    bb1 = ((stride*bbox).to(torch.float32)).floor()
    bb2 = (((stride*bbox+11)).to(torch.float32)).floor()
    bbox = torch.cat([bb1,bb2],dim=1)
    dx1 = rec[0][y,x].unsqueeze(1)
    dx2 = rec[1][y,x].unsqueeze(1)
    dx3 = rec[2][y,x].unsqueeze(1)
    dx4 = rec[3][y,x].unsqueeze(1)

    score = torch.transpose(prob[y,x].unsqueeze(0),0,1)
    offset = torch.cat([dx1,dx2,dx3,dx4],dim=1)
    bbox = (bbox + offset*12.0)*scale

    rectangle = torch.cat([bbox,score],dim=1)
    rectangle = rect2square(rectangle)

    rectangle[:,[0,2]] = torch.clamp(rectangle[:,[0,2]],0,width)
    rectangle[:,[1,3]] = torch.clamp(rectangle[:,[1,3]],0,height)

    pick = []
    for i in range(rectangle.shape[0]):
        x1 = int(max(0,rectangle[i][0]))
        y1 = int(max(0, rectangle[i][1]))
        x2 = int(max(0, rectangle[i][2]))
        y2 = int(max(0, rectangle[i][3]))
        score = rectangle[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,score])

    output = NMS(torch.tensor(pick),0.3)
    #output = torchvision.ops.nms(pick[:,:4],pick[:,4],0.3)

    return output

def filter_face_24net(prob,roi,rectangles,width,height,threshold,alpha=0.3):
    mask = prob >= threshold
    mask = mask.nonzero()
    score = prob[mask]
    roi = roi[mask,:][:,0,:]
    rectangles = rectangles[mask][:,0,:4]
    w = (rectangles[:,2] - rectangles[:,0]).unsqueeze(1)
    h = (rectangles[:,3] - rectangles[:,1]).unsqueeze(1)
    rectangles[:,[0,2]] = rectangles[:,[0,2]] + roi[:,[0,2]]*w*alpha
    rectangles[:,[1,3]] = rectangles[:,[1,3]] + roi[:,[1,3]]*h*alpha

    rectangles = torch.cat([rectangles,score],dim=1)
    rectangles = rect2square(rectangles)
    rectangles[:,[0,2]] = torch.clamp(rectangles[:,[0,2]],0,width)
    rectangles[:,[1,3]] = torch.clamp(rectangles[:,[1,3]],0,height)
    pick = []
    for i in range(rectangles.shape[0]):
        x1 = int(max(0,rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(max(0, rectangles[i][2]))
        y2 = int(max(0, rectangles[i][3]))
        score = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,score])
    rectangles = torch.tensor(pick)
    return NMS(rectangles,0.7)

def filter_face_48net(prob,roi,pts,rectangles,width,height,threshold):
    mask  = prob[:,1] >= threshold
    mask = mask.nonzero()[:,0]
    score = prob[mask,1].unsqueeze(1)
    roi = roi[mask,:]
    pts = pts[mask,:]
    rectangles = rectangles[mask,:4]
    w = (rectangles[:,2] - rectangles[:,0]).unsqueeze(1)
    h = (rectangles[:,3] - rectangles[:,1]).unsqueeze(1)

    face_marks = torch.zeros_like(pts)
    face_marks[:, [0,2,4,6,8]] = w * pts[:, [0,1,2,3,4]] + rectangles[:, 0:1]
    face_marks[:, [1,3,5,7,9]] = h * pts[:, [5,6,7,8,9]] + rectangles[:, 1:2]
    rectangles[:, [0,2]]  = rectangles[:, [0,2]] + roi[:, [0,2]] * w
    rectangles[:, [1,3]]  = rectangles[:, [1,3]] + roi[:, [1,3]] * w
    rectangles = torch.cat([rectangles,score,face_marks],dim= 1)#[0,1,2,3, 4 ,5,6,7,8,9,10,11,12,13,14]

    rectangles[:,[0,2]] = torch.clamp(rectangles[:,[0,2]],0,width)
    rectangles[:,[1,3]] = torch.clamp(rectangles[:,[1,3]],0,height)
    out = NMS(rectangles,0.3)
    return out







def rect2square(rec):
    w = (rec[:,2] - rec[:,0])
    h = (rec[:,3] - rec[:,1])
    l, temp = torch.max(torch.cat([w.unsqueeze(1),h.unsqueeze(1)],dim=1),dim=1)
    rec[:,0] = rec[:,0] + w*0.5 - l*0.5
    rec[:,1] = rec[:,1] + h * 0.5 - l * 0.5
    l = l.unsqueeze(1).expand(-1,2)
    rec[:,[2,3]] = rec[:,[2,3]] + l
    return rec



def NMS(input,threshold):
    """
    intput:Tensor (nums,5)
    threshold:Int
    -> Tensor
    将输入的一些方框,通过计算排序得到置信度最高的框,然后每个框和该框计算Iou
    将Iou大于阈值的框删去,然后剩下的框继续循环
    然后输出input[pick]
    """
    if input.shape[0] == 0 or input.shape[0] == 1:
        return input.unsqueeze(0)
    x1 = input[:,0]
    y1 = input[:, 1]
    x2 = input[:, 2]
    y2 = input[:, 3]
    score = input[:, 4]
    area = torch.mul(x2-x1+1,y2-y1+1)
    I = torch.argsort(score,dim=0)
    pick = []
    while len(I)>0:
        xx1 = max(torch.cat([x1[I[-1]].unsqueeze(0),x1[I[0:-1]]],dim=0))
        yy1 = max(torch.cat([y1[I[-1]].unsqueeze(0),y1[I[0:-1]]],dim=0))
        xx2 = min(torch.cat([x2[I[-1]].unsqueeze(0), x2[I[0:-1]]], dim=0))
        yy2 = min(torch.cat([y2[I[-1]].unsqueeze(0), y2[I[0:-1]]], dim=0))
        w = max(0.0,xx2-xx1+1)
        h = max(0.0,yy2-yy1+1)
        inter = w*h
        o = inter / (area[I[-1]] + area[I[0:-1]] -inter)
        pick.append(I[-1])
        mask = o < threshold
        I = I[(o<threshold).nonzero()].reshape(-1)
    return input[torch.cat([torch.tensor(pick).unsqueeze(0)],dim = 0)]




