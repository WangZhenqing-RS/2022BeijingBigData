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
import sys
from torch.optim.swa_utils import AveragedModel

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
          model_best_path, swa_model_path, early_stop, copy_paste, cosine_t0, lr):
    
    train_loader = get_dataloader(train_image_paths, train_label_paths,
                                  "train", copy_paste, batch_size, shuffle=True, num_workers=0)
    
    #swa_model = AveragedModel(model).to(DEVICE)
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_best_path))
    swa_model = AveragedModel(model).to(DEVICE)

    # 采用AdamM优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=1e-3)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cosine_t0, # T_0就是初始restart的epoch数目
        T_mult=1, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-6, # 最低学习率
        )
    
    # 损失函数采用EdgeBCELoss+DiceLoss
    loss_fn = edgebce_dice_loss
    
    header = r'Epoch/EpochNum | TrainLoss |  Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.5f} | {:9.2f}'
    print(header)
    
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
        
        if epoch%8 == 0:
            swa_model.update_parameters(model)
            
        scheduler.step()
        
        # 输出进程
        print(raw_line.format(epoch, epoches, np.array(losses).mean(), 
                              (time.time()-start_time)/60**1))   
        
    # 最后更新BN层参数
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device= DEVICE)
    torch.save(swa_model.state_dict(), swa_model_path)

    

if __name__ == '__main__': 
    
    # seed = 619
    seed = int(sys.argv[1])
    # batch_size = 2
    batch_size = int(sys.argv[2])
    # cosine_t0 = 8
    cosine_t0 = int(sys.argv[3])
    # cosine_num = 4
    cosine_num = int(sys.argv[4])
    # lr = 5e-4
    lr = float(sys.argv[5])
    # model_best_path = 
    model_best_path = sys.argv[6]
    
    
    epoches = cosine_t0*cosine_num
    
    seed_it(seed)

    image_paths = sorted(glob.glob("../data/image/*.png"))
    label_paths = sorted(glob.glob("../data/label/*.png"))
    
    channels = 3
    
    early_stop = 100
    is_kmeans_copy_paste = True
    
    
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

    
    
    swa_model_path = f"../model/swa_s{seed}_t{cosine_t0}_n{cosine_num}_l{lr}.pth"
    train(
        model,
        epoches,
        batch_size,
        image_paths,
        label_paths,
        model_best_path,
        swa_model_path,
        early_stop,
        is_kmeans_copy_paste,
        cosine_t0,
        lr,
        )
        
