import torchvision.transforms
import tool
from torch import nn
import torch
import cv2
img = cv2.imread('TEST.jpg')
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.float()
img_tensor = img_tensor.permute(2,0,1)
print("origin image shape:",img_tensor.shape)
def WAfterConv(in_W, kernel_size, stride, padding):
    return (in_W - kernel_size + 2 * padding) / stride + 1

def PNet(input):#后面的所以input都默认为ndarray
    input = torch.from_numpy(input).float().permute(2,0,1)
    PnetModel = nn.Sequential(
        nn.Conv2d(3, 10, kernel_size=3, stride=1,padding=1), # 10, h, w
        nn.PReLU(), # 10, h, w
        nn.MaxPool2d(kernel_size=2, stride=2), # 10, h/2, w/2
        nn.Conv2d(10, 16, kernel_size=3, stride=1), # 16, h/2-2, w/2-2,
        nn.PReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1), # 32, h/2-4, w/2-4 69 66
        nn.PReLU()
    )
    output = PnetModel(input)
    classifier = nn.Conv2d(32, 2, kernel_size=1, stride=1)(output) #表示方框的置信度
    bbox_regress = nn.Conv2d(32, 4, kernel_size=1, stride=1)(output) # 表示方框的位置
    return [classifier, bbox_regress]

def RNet(input):
    input = cv2.resize(input, (24, 24))
    input = torch.from_numpy(input)
    input = input.float()
    input = input.permute(2,0,1)
    RNetModel = nn.Sequential(
        nn.Conv2d(3, 28, kernel_size=3, stride=1), # 28, 22, 22
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True), # 28, 11, 11

        nn.Conv2d(28, 48, kernel_size=3, stride=1), # 48, 8, 8
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True), # 48, 3, 3

        nn.Conv2d(48, 64, kernel_size=2, stride=1), # 64, 3, 3
        nn.PReLU(),

    )
    output = RNetModel(input)
    output = output.view(-1)
    output = tool.Dense(output, 128)
    classifier = tool.Dense(output, 2)
    bbox_regress = tool.Dense(output, 4,False)
    return [classifier, bbox_regress]
def ONet(input):
    input = cv2.resize(input, (48, 48))
    input = torch.from_numpy(input)
    input = input.float()
    input = input.permute(2,0,1)
    ONetModel = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1), # 32, 46, 46
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True), # 32, 23, 23

        nn.Conv2d(32, 64, kernel_size=3, stride=1), # 64, 21, 21
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True), # 64, 10, 10

        nn.Conv2d(64, 64, kernel_size=3, stride=1), # 64, 8, 8
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True), # 64, 4, 4

        nn.Conv2d(64, 128, kernel_size=2, stride=1), # 128, 3, 3
        nn.PReLU(),
    )
    output = ONetModel(input)
    output = output.view(-1)
    output = tool.Dense(output, 256)

    classifier = tool.Dense(output, 2)
    bbox_regress = tool.Dense(output, 4)


    return output

def detectFace(input,threshold):#input 是一个tensor类型
    input = input / 255
    origin_h, origin_w = input.shape[1],input.shape[2]
    scales = tool.CalculateScales(input)#一个list ,有十几个缩放的比例
    out = []
    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)


        scale_img = cv2.resize(input.permute(1,2,0).numpy(), (ws, hs),interpolation=cv2.INTER_LINEAR)
        # scale_img = scale_img.reshape(1,*scale_img.shape) 添加一个batchsize维度
        output = PNet(scale_img)
        out.append(output)
    # 循环结束后的out就是不同缩放比例通过PNet得到的[classifier,bbox]
    image_num = len(scales)
    rectanges = []
    for i in range(image_num):
        pass






detectFace(img_tensor,0.5)
