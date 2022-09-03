# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""

import torch.utils.data as D
from torchvision import transforms as T
import numpy as np
import torch
import cv2
import albumentations as A
import random
from utils.edge_utils import mask_to_onehot, onehot_to_binary_edges
from utils.kmeans_copy_paste import kmeans_copy_paste


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集IoU
def cal_val_iou(model, loader):
    val_iou = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        # print(output.shape,target.shape)
        iou = cal_iou(output, target)
        val_iou.append(iou)
    return val_iou

#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集f1
def cal_val_f1(model, loader):
    val_f1 = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        f1 = cal_f1(output, target)
        val_f1.append(f1)
    return val_f1

#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集f1
def cal_val_fwiou(model, loader):
    val_fwiou = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        f1 = cal_fwiou(output, target)
        val_fwiou.append(f1)
    return val_fwiou

# 计算IoU
def cal_iou(pred, mask, c=1):
    iou_result = []
    # for idx in range(c):
    idx = c
    p = (mask == idx).int().reshape(-1)
    t = (pred == idx).int().reshape(-1)
    uion = p.sum() + t.sum()
    overlap = (p*t).sum()
    #  0.0001防止除零
    iou = 2*overlap/(uion + 0.000001)
    iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

# 计算fwiou
def cal_fwiou(pred, mask, c=1):
    
    pred = pred.cpu().numpy().flatten()
    mask = mask.cpu().numpy().flatten()
    pred = pred.astype(np.uint8)
    mask = mask.astype(np.uint8)
    confusionMatr = ConfusionMatrix(2, pred, mask)
    FWIoU = Frequency_Weighted_Intersection_over_Union(confusionMatr)
    return FWIoU

# 计算IoU
def cal_f1(pred, mask, c=1):
    f1_result = []
    # TP    predict 和 label 同时为1
    TP = ((pred == 1) & (mask == 1)).cpu().sum()
    # TN    predict 和 label 同时为0
    # TN = ((pred == 0) & (mask == 0)).cpu().sum()
    # FN    predict 0 label 1
    FN = ((pred == 0) & (mask == 1)).cpu().sum()
    # FP    predict 1 label 0
    FP = ((pred == 1) & (mask == 0)).cpu().sum()
    
    p = TP / (TP + FP + 0.000001)
    r = TP / (TP + FN + 0.000001)
    f1 = 2 * r * p / (r + p + 0.000001)

    f1_result.append(f1)
    return np.stack(f1_result)

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode, is_kmeans_copy_paste):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.is_kmeans_copy_paste = is_kmeans_copy_paste
        self.transform = A.Compose([
            # 亮度变换
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomBrightness(limit=0.1, p=0.5),
            ], p=1),
            # 垂直翻转
            A.HorizontalFlip(p=0.5),
            # 水平翻转
            A.VerticalFlip(p=0.5),
            # 旋转90度
            # A.RandomRotate90(p=0.5),
            # 平移.尺度加旋转变换
            A.ShiftScaleRotate(rotate_limit=1, p=0.5),
            # # 增加模糊
            # A.OneOf([
            #     A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),
            # ], p=0.5),
            # # 随机擦除
            # A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        ])
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
        
    # 获取数据操作
    def __getitem__(self, index):
        
        image = cv2.imread(self.image_paths[index])
        # image = cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)
        if self.mode == "train":
            label = cv2.imread(self.label_paths[index],0)
            # label = cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST)
            
            if self.is_kmeans_copy_paste:
                # kmeans_copy_paste增强
                if random.random()>0.5 :
                    image, label = kmeans_copy_paste(image, label)
                
                
            label[label == 255] = 1
        
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            
            # 获取正样本边缘->服务于边缘加权bce
            oneHot_label = mask_to_onehot(label,2) #edge=255,background=0
            edge = onehot_to_binary_edges(oneHot_label,2,2)
            
            # 消去图像边缘
            edge[:2, :] = 0
            edge[-2:, :] = 0
            edge[:, :2] = 0
            edge[:, -2:] = 0
            
            label = label.reshape((1,) + label.shape)
            edge = edge.reshape((1,) + edge.shape)
            return self.as_tensor(image), label.astype(np.int64), edge
        
        elif self.mode == "val":
            
            label = cv2.imread(self.label_paths[index],0)
            # label = cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST)
            label[label == 255] = 1
            label = label.reshape((1,) + label.shape)
            return self.as_tensor(image), label.astype(np.int64)
        elif self.mode == "test":
            return self.as_tensor(image), self.image_paths[index]
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_paths, label_paths, mode, is_kmeans_copy_paste, batch_size, 
                   shuffle, num_workers):
    dataset = OurDataset(image_paths, label_paths, mode, is_kmeans_copy_paste)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def split_train_val(image_paths, label_paths, val_index=0):
    # 分隔训练集和验证集
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
    for i in range(len(image_paths)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == val_index:
            val_image_paths.append(image_paths[i])
            val_label_paths.append(label_paths[i])
        else:
            train_image_paths.append(image_paths[i])
            train_label_paths.append(label_paths[i])
    print("Number of train images: ", len(train_image_paths))
    print("Number of val images: ", len(val_image_paths))
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths
