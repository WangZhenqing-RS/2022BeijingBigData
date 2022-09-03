# -*- coding: utf-8 -*-
"""
@author: wangzhenqing
ref:
1. 
2.https://github.com/DLLXW/data-science-competition/tree/main/%E5%A4%A9%E6%B1%A0
3.https://github.com/JasmineRain/NAIC_AI-RS/tree/ec70861e2a7f3ba18b3cc8bad592e746145088c9
"""
import numpy as np
import torch
import warnings
import time
from data_process import get_dataloader, split_train_val, cal_val_iou
import segmentation_models_pytorch as smp
import glob
import os
from loss.edgebce_dice_loss import edgebce_dice_loss
import random


# 忽略警告信息
warnings.filterwarnings('ignore')

# 将模型加载到指定设备DEVICE上
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# 设置随机种子
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = True   
    torch.manual_seed(seed)
    

def train(model, epoches, batch_size, train_image_paths, train_label_paths, 
          val_image_paths, val_label_paths, channels, model_path, early_stop, 
          copy_paste):
    
    train_loader = get_dataloader(train_image_paths, train_label_paths,
                                  "train", copy_paste, batch_size, shuffle=True, num_workers=0)
    valid_loader = get_dataloader(val_image_paths, val_label_paths,
                                  "val", copy_paste, batch_size, shuffle=False, num_workers=0)
    
    
    model.to(DEVICE)
    # 采用AdamM优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2, # T_0就是初始restart的epoch数目
        T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-5, # 最低学习率
        )
    
    # 损失函数采用EdgeBCELoss+DiceLoss
    loss_fn = edgebce_dice_loss
    
    header = r'Epoch/EpochNum | TrainLoss | ValidIoU | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.5f} | {:9.5f} | {:9.2f}'
    print(header)
 
    # 记录当前验证集最优IoU,以判定是否保存当前模型
    best_iou = 0
    best_iou_epoch = 0
    train_loss_epochs, val_iou_epochs, lr_epochs = [], [], []
    # 开始训练
    for epoch in range(1, epoches+1):
        # 存储训练集每个batch的loss
        losses = []
        start_time = time.time()
        model.train()
        model.to(DEVICE);
        for batch_index, (image, target, edge) in enumerate(train_loader):
            image, target, edge = image.to(DEVICE), target.to(DEVICE), edge.to(DEVICE)
            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
            # 模型推理得到输出
            output = model(image)
            output=output.to(torch.float32)
            target=target.to(torch.float32)
            # 求解该batch的loss
            loss = loss_fn(output, target, edge)
            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            losses.append(loss.item())
        
        scheduler.step()
        # 计算验证集IoU
        val_iou = cal_val_iou(model, valid_loader)
        # 保存当前epoch的train_loss val_IoU lr
        train_loss_epochs.append(np.array(losses).mean())
        val_iou_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        # 输出进程
        print(raw_line.format(epoch, epoches, np.array(losses).mean(), 
                              np.mean(val_iou), 
                              (time.time()-start_time)/60**1), end="")   
        
        if best_iou < np.stack(val_iou).mean(0).mean() or 1:
            best_iou = np.stack(val_iou).mean(0).mean()
            best_iou_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("  valid fwiou is improved. the model is saved.")
        else:
            print("")
            if (epoch - best_iou_epoch) >= early_stop:
                break
    
    return train_loss_epochs, val_iou_epochs, lr_epochs
    

if __name__ == '__main__': 
    
    seed_it(619)
    epoches = 66
    batch_size = 2

    image_paths = sorted(glob.glob("../data/image/*.png"))
    label_paths = sorted(glob.glob("../data/label/*.png"))
    
    # train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths, label_paths, val_index=0)
    train_image_paths = image_paths
    train_label_paths = label_paths
    val_image_paths = sorted(glob.glob("../data/image/*.png"))
    val_label_paths = sorted(glob.glob("../data/label/*.png"))

    channels = 3
    
    early_stop = 100
    is_kmeans_copy_paste = True
    
    models, model_names = [], []
    
    
    model = smp.Unet(
            encoder_name="mobilenet_v2", 
            encoder_weights="imagenet",
            in_channels=channels,
            decoder_attention_type="scse",
            classes=1,
            encoder_depth=3,
            decoder_channels=(64, 32, 16),
            activation="sigmoid",
    )

    model_name = "mobileunet_scse_c16_d3_s619_alltrain_notest"
    
    print(f"模型: {model_name}")
    model_path = f"../model/{model_name}_kcp{is_kmeans_copy_paste}.pth"
    train_loss_epochs, val_f1_epochs, lr_epochs = train(
        model,
        epoches,
        batch_size,
        train_image_paths,
        train_label_paths,
        val_image_paths,
        val_label_paths,
        channels,
        model_path,
        early_stop,
        is_kmeans_copy_paste,
        )
        
