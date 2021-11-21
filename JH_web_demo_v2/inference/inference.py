import time
import sys

PROJECT_PATH = "/home/jh/Final_Project/JH_web_demo_v2/inference"
sys.path.append(PROJECT_PATH)
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image

<<<<<<< Updated upstream

opt = TestOptions().parse()
opt.resize_or_crop = 'none'

=======
opt = TestOptions().parse()
opt.resize_or_crop = 'none'
print(opt.resize_or_crop)
>>>>>>> Stashed changes

start_epoch, epoch_iter = 1, 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
<<<<<<< Updated upstream
=======
print(dataset_size)
>>>>>>> Stashed changes

warp_model = AFWM(opt, 3)
warp_model.eval()
warp_model.cuda()
<<<<<<< Updated upstream
=======
# load_checkpoint(warp_model, opt.warp_checkpoint)
>>>>>>> Stashed changes

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
<<<<<<< Updated upstream
=======
# load_checkpoint(gen_model, opt.gen_checkpoint)
>>>>>>> Stashed changes

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

def inference_image(category, input_path, result_path):
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
    # category 따라모델 불러오기
    if category == 'top':
        warp_checkpoint = './pretrained_models/top_warp_model_1.pth'
        gen_checkpoint = './pretrained_models/top_gen_model_1.pth'
    else:
        warp_checkpoint = './pretrained_models/bottom_warp_model_1.pth'
        gen_checkpoint = './pretrained_models/bottom_gen_model_1.pth'

    load_checkpoint(warp_model, warp_checkpoint)
    load_checkpoint(gen_model, gen_checkpoint)

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    I = Image.open(input_path['image']).convert('RGB')

    params = get_params(opt, I.size)
    transform = get_transform(opt, params)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    
    I_tensor = transform(I)
    
    C = Image.open(input_path['clothes']).convert('RGB')
    C_tensor = transform(C)
    
    E = Image.open(input_path['edge']).convert('L')
    E_tensor = transform_E(E)

    data = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}    

    iter_start_time = time.time()
    real_image = data['image']
    real_image = real_image.unsqueeze(0) 
<<<<<<< Updated upstream
    clothes = data['clothes']
    clothes = clothes.unsqueeze(0) 
    ##edge is extracted from the clothes image with the built-in function in python
    edge = data['edge']
    edge = edge.unsqueeze(0) 
=======
    print(real_image.shape)
    print(type(real_image))
    clothes = data['clothes']
    clothes = clothes.unsqueeze(0) 
    print(clothes.shape)
    ##edge is extracted from the clothes image with the built-in function in python
    edge = data['edge']
    edge = edge.unsqueeze(0) 
    print(edge.shape)
>>>>>>> Stashed changes
    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
    clothes = clothes * edge        

    flow_out = warp_model(real_image.cuda(), clothes.cuda())
    warped_cloth, last_flow, = flow_out
    warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                        mode='bilinear', padding_mode='zeros')

    gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

<<<<<<< Updated upstream
    c = p_tryon
    combine = torch.cat([c[0]], 2).squeeze()
    cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    rgb=(cv_img*255).astype(np.uint8)
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path,bgr) 
    return bgr
=======
    a = real_image.float().cuda()
    b= clothes.cuda()
    c = p_tryon
    combine = torch.cat([c[0]], 2).squeeze()
    cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    rgb = (cv_img*255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, bgr)
    cv2.imwrite('result2.jpg', rgb)

    return result_path
>>>>>>> Stashed changes



