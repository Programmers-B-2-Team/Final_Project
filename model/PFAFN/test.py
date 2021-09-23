import time
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


class RunModel():
    def __init__(self) -> None:
        self.opt = TestOptions().parse()

        self.warp_model = AFWM(self.opt, 3)
        print(self.warp_model)
        self.warp_model.eval()
        self.warp_model.cuda()
        load_checkpoint(self.warp_model, self.opt.warp_checkpoint)

        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        print(self.gen_model)
        self.gen_model.eval()
        self.gen_model.cuda()
        load_checkpoint(self.gen_model, self.opt.gen_checkpoint)


    def run_model(self, baseroot, output_path):
        self.opt.dataroot = baseroot

        start_epoch, epoch_iter = 1, 0

        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print(dataset_size)

        total_steps = (start_epoch-1) * dataset_size + epoch_iter
        step = 0
        step_per_batch = dataset_size / self.opt.batchSize

        for epoch in range(1,2):

            for i, data in enumerate(dataset, start=epoch_iter):
                iter_start_time = time.time()
                total_steps += self.opt.batchSize
                epoch_iter += self.opt.batchSize

                real_image = data['image']
                clothes = data['clothes']
                ##edge is extracted from the clothes image with the built-in function in python
                edge = data['edge']
                edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
                clothes = clothes * edge        

                flow_out = self.warp_model(real_image.cuda(), clothes.cuda())
                warped_cloth, last_flow, = flow_out
                warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                                mode='bilinear', padding_mode='zeros')

                gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
                gen_outputs = self.gen_model(gen_inputs)
                p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                p_rendered = torch.tanh(p_rendered)
                m_composite = torch.sigmoid(m_composite)
                m_composite = m_composite * warped_edge
                p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

                path = output_path + self.opt.name
                os.makedirs(path, exist_ok=True)
                sub_path = path + '/PFAFN'
                os.makedirs(sub_path,exist_ok=True)

                if step % 1 == 0:
                    c = p_tryon
                    combine = torch.cat([c[0]], 2).squeeze()
                    cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
                    rgb=(cv_img*255).astype(np.uint8)
                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)

                step += 1
                if epoch_iter >= dataset_size:
                    break

