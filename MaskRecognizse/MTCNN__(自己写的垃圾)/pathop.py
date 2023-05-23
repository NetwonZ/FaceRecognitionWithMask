import os
import torch
root = 'C:\\Users\\张人元\\Desktop\\MaskRecognizse\\data\\AFDB_face_dataset'

name_class = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root,name))]
class_indices = dict((k,v)for v,k in enumerate(name_class))
print(class_indices)

# 输出txt文件
# root = os.path.dirname(os.getcwd())
# nomask_path = os.path.join(root,'data','AFDB_face_dataset')
# img_name = os.listdir(nomask_path)
# count = 0
# for i in img_name:
#     label = i
#     img_path = os.path.join(nomask_path,i)
#
#     for j in os.listdir(img_path):
#         with open(os.path.join(root,'data','nomask_tr.txt'),'a+',encoding='utf-8') as f:
#             img_x_path = os.path.join(img_path,j)
#             f.writelines(img_x_path+' '+label+'\n')
#             f.seek(0)
#             count += 1
#             print(count)

