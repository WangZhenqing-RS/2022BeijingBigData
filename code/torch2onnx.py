# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:33:18 2022

@author: DELL
"""
import segmentation_models_pytorch as smp
import torch
from torch.optim.swa_utils import AveragedModel
import sys

model = smp.Unet(
            encoder_name="mobilenet_v2", 
            encoder_weights="imagenet",
            in_channels=3,
            decoder_attention_type="scse",
            classes=1,
            encoder_depth=3,
            decoder_channels=(64, 32, 16),
            activation="sigmoid",
    )

# is_swa = 1
is_swa = int(sys.argv[1])
torch_model_path = sys.argv[2]
onnx_model_path = sys.argv[3]

if is_swa:
    model = AveragedModel(model)

model.load_state_dict(torch.load(torch_model_path))

model.eval()

# 随机生成的输入参数(shape需要和模型输入对应)
x=torch.randn((1,3,480,960))

input_names = ["input"]

torch.onnx.export(model=model,
                  args = (x),
                  f = onnx_model_path, # 转换输出的模型的地址
                  input_names = input_names, # 指定输入节点名称
                  opset_version = 11, # 默认的9不支持Upsample/Resize
                  )

