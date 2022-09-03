# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:03:31 2022

@author: DELL
"""

import numpy as np
import cv2
import os
import glob

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  


def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


model_names = []
model_names.append("mobileunet+")
# model_names.append("deeplabv3+")
# model_names.append("pspnet")
# model_names.append("resunet")
# model_names.append("efficientunet")
# model_names.append("unet")
# model_names.append("efficientunet+_noedge")
# model_names.append("efficientunet+_whu")
# model_names.append("hrnet")

for model_name in model_names:

    #################################################################
    # #  标签图像文件夹
    # LabelPath = r"..\data\test_labels\*.tif"
    #  预测图像文件夹
    # PredictPath = r"..\data\test_pred_efficientunet+\*.tif"
    PredictPath = f"../../data/test_pred_{model_name}/*.png"
    #  类别数目(包括背景)
    classNum = 2
    #################################################################
    
    # LabelPaths = glob.glob(LabelPath)
    PredictPaths = glob.glob(PredictPath)
    LabelPaths = []
    for PredictPath in PredictPaths:
        LabelPaths.append(PredictPath.replace(f"test_pred_{model_name}","label"))
    
    label_all = []
    predict_all = []
    for i in range(len(LabelPaths)):
        label = cv2.imread(LabelPaths[i],0)
        label_all.append(label)
        pred = cv2.imread(PredictPaths[i],0)
        predict_all.append(pred)
    
    
    #  拉直成一维
    label_all = np.array(label_all).flatten()
    predict_all = np.array(predict_all).flatten()
    
    label_all[label_all==255] = 1
    predict_all[predict_all==255] = 1
    
    #  计算混淆矩阵及各精度参数
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    
    print("========================")
    print(f"{model_name}")
    # print("混淆矩阵:")
    # print(confusionMatrix)
    print("精确度:")
    print(precision[1])
    print("召回率:")
    print(recall[1])
    print("F1-Score:")
    print(f1ccore[1])
    # print("整体精度:")
    # print(OA)
    print("IoU:")
    print(IoU[1])
    print("mIoU:")
    print(mIOU)
    print("FWIoU:")
    print(FWIOU)