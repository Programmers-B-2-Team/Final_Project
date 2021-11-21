import time
from options.train_options import TrainOptions
from models.networks import VGGLoss,save_checkpoint
from models.afwm import TVLoss,AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime

opt = TrainOptions().parse()
path = 'runs/'+opt.name
os.makedirs(path,exist_ok=True)

def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

os.makedirs('sample',exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.local_rank}')

############################################################################
# start_epoch를 이전 checkpoint의 epoch에서 이어서 시작하기 위한 코드 부분
# checkpoint 폴더에서 가장 마지막 checkpoint 파일명 알아내기
if opt.continue_train:
    checkpoints_list = []
    checkpoints_optimizer = []
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name)
    cp_dirs = os.listdir(checkpoint_path)
    for cp in cp_dirs:
        if 'PBAFN_warp' in cp:
            checkpoints_list.append(cp)
        elif 'optimizer' in cp:
            checkpoints_optimizer.append(cp)
    checkpoints_list.sort()
    checkpoints_optimizer.sort()
    latest_checkpoint = checkpoints_list[-1]

  # 지정 epoch 혹은 가장 마지막 체크포인트에서 마지막으로 학습한 epoch 알아내기
    if opt.which_epoch != 'latest':
        for i in checkpoints_list:
            if opt.which_epoch in i:
                restored_epoch = int(opt.which_epoch)
                break
        else:
            raise AssertionError("%s epoch is not in checkpoints" % opt.which_epoch)
    else:
        restored_epoch = int(latest_checkpoint.split('_')[-1][:3])

    print("==================== restored epoch :", restored_epoch, "====================")

    start_epoch = restored_epoch
    epoch_iter = 0
else:
	start_epoch = 1
	epoch_iter = 0
############################################################################

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                                               num_workers=4, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

warp_model = AFWM(opt, 44)
print(warp_model)
warp_model.train()
warp_model.cuda()
warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

####################################################################
if opt.isTrain and len(opt.gpu_ids):
        # continue_train 옵션을 준 경우 저장된 checkpoint에서 불러온 정보를 학습 모델에 적용
    if opt.continue_train:
        print("==================== continue train: YES ====================")
        checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, latest_checkpoint)
        load_checkpoint_parallel(warp_model, checkpoint_path)
    model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.local_rank])
####################################################################

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp = [p for p in model.parameters()]

if opt.continue_train:
    optimizer_warp = torch.load('checkpoints/PBAFN_stage1/PBAFN_stage1_optimizer_%03d.pth' % (restored_epoch))
    print("************* optimizer_warp has loaded from 'PBAFN_stage1_optimizer_%03d.pth' *************" % (restored_epoch))
else:
    optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

if opt.local_rank == 0:
    writer = SummaryWriter(path)



for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        #t_mask = torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float))
        #data['label'] = data['label']*(1-t_mask)+t_mask*4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==5).astype(np.int))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
        densepose_fore = data['densepose']/24.0
        face_mask = torch.FloatTensor((data['label'].cpu().numpy()==1).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==2).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==13).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy()==9).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==14).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==15).astype(np.int)) + \
                             torch.FloatTensor((data['label'].cpu().numpy()==16).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==17).astype(np.int)) + \
                             torch.FloatTensor((data['label'].cpu().numpy()==18).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==19).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==20).astype(np.int))
        preserve_mask = torch.cat([face_mask,other_clothes_mask],1)
        concat = torch.cat([preserve_mask.cuda(),densepose,pose.cuda()],1)

        flow_out = model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b,c,h,w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x)/(b*c*h*w)
            loss_flow_y = (delta_y_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y)/(b*c*h*w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

        loss_all = 0.01 * loss_smooth + loss_all

        if opt.local_rank == 0:
          writer.add_scalar('loss_all', loss_all, step)

        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        ############## Display results and errors ##########

        path = 'sample/'+opt.name
        os.makedirs(path,exist_ok=True)
        if step % 1000 == 0:
          if opt.local_rank == 0:
            a = real_image.float().cuda()
            b = person_clothes.cuda()
            c = clothes.cuda()
            d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
            e = warped_cloth
            f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_image('combine', (combine.data + 1) / 2.0, step)
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite('sample/'+opt.name+'/'+str(step)+'.jpg',bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
          if opt.local_rank == 0:
            print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter,step, loss_all,eta))

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    if opt.local_rank == 0:
      print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if opt.local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_checkpoint(model.module, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch+1)))
            print('saved learning rate :', optimizer_warp.param_groups[0]['lr'])
            torch.save(optimizer_warp, 'checkpoints/PBAFN_stage1/PBAFN_stage1_optimizer_%03d.pth' % (epoch+1))

    if epoch > opt.niter:
        if opt.continue_train:
            model.module.continue_update_learning_rate(optimizer_warp)
        else:
            model.module.update_learning_rate(optimizer_warp)
