import cv2
import numpy as np
def cal_distance(current_encoding,all_encoding):
    all_encoding = np.array(all_encoding)
    distance = np.linalg.norm(all_encoding - current_encoding, axis=1)
    return distance

def Draw_Rec(Out_rec,Ori_img):
    img_copy = Ori_img.copy()
    for i in range(len(Out_rec[0][0])):
        cv2.rectangle(img_copy, (int(Out_rec[0][0][i][0]), int(Out_rec[0][0][i][1])), (int(Out_rec[0][0][i][2]), int(Out_rec[0][0][i][3])),
                      (0, 255, 0), 2)
        for j in range(5):
            cv2.circle(img_copy, (int(Out_rec[1][0][i][j][0]), int(Out_rec[1][0][i][j][1])), 2, (0, 0, 255), 2)
    cv2.imshow('img', img_copy)
    cv2.waitKey(20)


def Face_Alignment(img,out):#输入原图,和out,输出矫正后的s*s的人脸,返回的img是个列表,包含多张人脸
    rec = out[0][0]
    landmark = out[1][0]
    number = rec.shape[0]
    height = img.shape[0]
    weight = img.shape[1]
    size = 160
    img_output = []
    for i in range(number):

        ldmk_temp =(landmark[i] - np.array([rec[i][0], rec[i][1]])) / np.array([rec[i][2] - rec[i][0], rec[i][3] - rec[i][1]]) *160

        img_crop = img[int(rec[i][1]):int(rec[i][3]), int(rec[i][0]):int(rec[i][2])]
        img_crop = cv2.resize(img_crop, (size, size))

        x = ldmk_temp[1][0] - ldmk_temp[0][0]
        y = ldmk_temp[1][1] - ldmk_temp[0][1]
        if x == 0:
            angle = 0
        else:
            angle = np.arctan(y / x) * 180 / np.pi
        center = (size // 2, size // 2)
        Rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        #cv2仿射变换
        img_output.append(cv2.warpAffine(img_crop, Rotation_matrix, (size, size)))
        # 还有个ladnmark也要旋转,还木有做. --- 其实是用不上


    return img_output
