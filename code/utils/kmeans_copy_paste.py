# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:32:39 2022

@author: DELL
"""
import cv2
import numpy as np
import random
 
def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
        
    sub_img_src = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask_src)
    sub_img_src_wh = cv2.resize(sub_img_src, (w, h),interpolation=cv2.INTER_NEAREST)
    mask_src_wh = cv2.resize(mask_src, (w, h), interpolation=cv2.INTER_NEAREST)
    sub_img_main = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8), mask=mask_src_wh)
    img_main = img_main - sub_img_main + sub_img_src_wh
    return img_main


def kmeans_copy_paste(image, label):
    
    copy_image = image.copy()
    copy_label = label.copy()
    
    # 转换成3列
    data = copy_image.reshape((-1,3))
    data = np.float32(data)
     
    # 定义终止条件 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
     
    # 设置初始中心的选择
    flags = cv2.KMEANS_PP_CENTERS
     
    # K-Means聚类成4类
    _, kmeans_label, _ = cv2.kmeans(data, 4, None, criteria, 2, flags)
    kmeans_label = kmeans_label.reshape((image.shape[0],image.shape[1]))
    kmeans_label = kmeans_label.astype(np.uint8)
    # 使类别从1开始
    kmeans_label = kmeans_label + 1
    
    # 卷积核
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(11,11))
    # 开运算
    kmeans_label_open = cv2.morphologyEx(kmeans_label, cv2.MORPH_OPEN, kernel=kernel)
    # 闭运算
    kmeans_label_open_close = cv2.morphologyEx(kmeans_label_open, cv2.MORPH_CLOSE, kernel=kernel)
    
    kmeans_label_open_close = kmeans_label_open_close.astype(np.float32)
    kmeans_label_open_close_class = kmeans_label_open_close.copy()
    kmeans_label_open_close_class[label==0] = np.nan
    kmeans_class = int(np.nanmean(kmeans_label_open_close_class))
    
    
    copy_label[copy_label==255] = kmeans_class
    
    # if random.random()<0.5 :
    #     # 水平翻转
    #     copy_image = cv2.flip(copy_image, 1)
    #     copy_label = cv2.flip(copy_label, 1)
    # if random.random()<0.5 :
    #     # 垂直翻转
    #     copy_image = cv2.flip(copy_image, 0)
    #     copy_label = cv2.flip(copy_label, 0)
    
    copy_image = cv2.flip(copy_image, 1)
    copy_label = cv2.flip(copy_label, 1)
    
    copy_label[copy_label!=kmeans_label_open_close]=0
    
    copy_label[copy_label==kmeans_class] = 255
    image_add = img_add(copy_image, image, copy_label)
    label_add = img_add(copy_label, label, copy_label)
    
    return image_add, label_add
    


if __name__ == '__main__':
    
    # 读取原始图像
    image = cv2.imread(r"E:\WangZhenQing\2022BeiJing\code\74.png")
    label = cv2.imread(r"E:\WangZhenQing\2022BeiJing\code\74_label.png",0)
    image_add, label_add = kmeans_copy_paste(image,label)
    cv2.imwrite(r"E:\WangZhenQing\2022BeiJing\code\74_kcp_new.png",image_add)
    cv2.imwrite(r"E:\WangZhenQing\2022BeiJing\code\74_label_kcp_new.png",label_add)
