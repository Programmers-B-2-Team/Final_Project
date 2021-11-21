python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 train_PBAFN_stage1.py --name PBAFN_stage1   \
--resize_or_crop None --verbose --tf_log --batchSize 16 --num_gpus 1 --label_nc 21 --launcher pytorch --continue_train --save_epoch_freq 10










