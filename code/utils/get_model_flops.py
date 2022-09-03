# -*- coding: utf-8 -*-
"""
@author: wangzhenqing
ref:
1. 
2.https://github.com/DLLXW/data-science-competition/tree/main/%E5%A4%A9%E6%B1%A0
3.https://github.com/JasmineRain/NAIC_AI-RS/tree/ec70861e2a7f3ba18b3cc8bad592e746145088c9
"""
import segmentation_models_pytorch as smp
from ptflops import get_model_complexity_info
import sys

def get_model_flops(model, image_shape):
    ost = sys.stdout
    flops, params = get_model_complexity_info(model, (3, 480, 960),
                            as_strings=True,
                            print_per_layer_stat=False,
                            ost=ost)
    
    print(f"{model.name} flops: {flops}  params: {params}.")
    
if __name__ == '__main__': 
    
    image_shape = (3, 480, 960)
    
    model = smp.Unet(
            encoder_name="mobilenet_v2", 
            encoder_weights="imagenet",
            in_channels=3,
            decoder_attention_type="scse",
            classes=1,
            encoder_depth=3,
            decoder_channels=(256, 128, 64),
            activation="sigmoid",
    )
    
    get_model_flops(model,image_shape)
    