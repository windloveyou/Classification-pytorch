import os
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from utils.utils import (download_weights, get_classes, weights_init, show_config, 
                         get_lr_scheduler, set_optimizer_lr)
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from nets import get_model_from_name
from utils.utils_fit import fit_one_epoch






if __name__ == "__main__":
    # 使用Cuda（含GPU）
    Cuda = True 
    
    # 不使用单机多卡训练
    distributed = False 
    sync_bn = False
    
    # 是否使用混合精度训练
    fp16 = False 
    
    # 数据集类别名--txt路径
    classes_path = "model_data/cls_classes.txt"
    
    # 输入图片大小
    input_shape = [224, 224]
    
    # Backbone(可选)
    backbone = "resnet50"
    
    # 是否进行预训练?
    pretrained = False
    model_path = ""
    
    
    # 冻结训练设置
    Init_Epoch = 0
    Freeze_Epoch = 5
    Freeze_batch_size = 32
    
    # 解冻
    UnFreeze_Epoch = 20 # 即模型总共训练的轮数
    Unfreeze_batch_size = 32
    
    # 是否进行冻结训练
    Freeze_Train = True
    
    # 训练参数设置：lr, optimizer_type, lr_decay_type等
    Init_lr = 1e-3 # Adam
    Min_lr = Init_lr * 0.01
    
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 5e-4
    
    lr_decay_type = "cos"
    
    save_period = 10 # 多少轮保存一次权值
    save_dir = 'logs'
    
    num_workers = 4 # 是否使用多线程
    
    # 图片路径及其对应标签
    train_annotation_path = "cls_train.txt"
    test_annotation_path = "cls_test.txt"
    
    
    # 显卡设置
    ngpus_per_node = torch.cuda.device_count()
    # print(ngpus_per_node)
    if distributed:
        pass
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)
        local_rank = 0
        rank = 0
        
     # 若预训练，则加载预训练权重
    if pretrained:
        if distributed:
            pass
        else:
            download_weights(backbone)        
            
    # 获取classes
    class_name, num_classes = get_classes(classes_path)
        
    if backbone not in ['vit']:
        model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)      
        
    if not pretrained:
        weights_init(model) # 需要理解
    if model_path != "":
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # 对模型的key 加载预训练权重的key
        model_dict = model.state_dict() # 取模型的字典
        pretrained_dict = torch.load(model_path, map_location=device) # map_location?
        load_key, no_load_key, temp_dict = [], [], {}
        # 遍历预训练权重的字典（k,v）
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(v)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        # 显示没有匹配上的key
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
            
    
    # loss相关
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None


    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler # 梯度缩放器
        scaler = GradScaler() # 用以在正向传播和反向传播中缩放梯度以保持梯度稳定（fp16 16位浮点数 比正常的32位浮点数 少 会在训练时出现数值溢出、下溢的问题）
    else:
        scaler = None
    
    # ------------------------------------------- # 
    #   区分训练模式和测试模式（Dropout, BN层的行为会发生改变）
    #   1. 训练模式
    #      Dropout: 按照指定的概率随机将部分神经元的输出置0，防止过拟合
    #      
    #      BN:  根据每个小批量计算均值和方差
    #
    #   2. 测试模式
    #      Dropout: 保持所有神经元的激活
    #
    #      BN:  使用训练期间累计的均值和方差
    # ------------------------------------------- # 
    
    # 开启训练模式
    model_train = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    
    
    if Cuda:
        if distributed:
            pass
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True # 启用 cuDNN 的自动优化 --> 自动寻找最优的卷积算法，提升训练速度
            model_train = model_train.cuda()
            

    # 加载数据
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    # 统计数据量
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    # 使得实验可重复
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None) # 再使后续的操作变为完全随机的状态
    
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    # "num_train // Unfreeze_batch_size" --> 每轮步数
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    
    if True:
        UnFreeze_flag = False
        
        # 冻结训练
        if Freeze_Train:
            pass
        
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        # 根据当前的bs，调整学习率
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            nbs             = 256
            lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
            'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        # 获取学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        # 计算每个epoch的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续训练，请扩充数据集！")
        
        train_dataset   = DataGenerator(train_lines, input_shape, True)
        val_dataset     = DataGenerator(val_lines, input_shape, False)
        
        if distributed:
            pass
        else:
            train_sampler = None
            val_sampler = None
            shuffle = None

        # ------------------------------------------------ #
        # DataLoader
        # 一种可迭代的数据加载器
        # 
        #
        # ------------------------------------------------ #
        
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
        
        # -------------------- # 
        # 开始模型训练
        # -------------------- # 
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ------------------------ #
            # 若模型度过冻结阶段，则解冻
            # ------------------------ #
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
                    nbs             = 256
                    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                # 模型解冻
                model.Unfreeze_backbone()
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size
                
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                
                if distributed:
                    pass

                gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

                UnFreeze_flag = True
                
            if distributed:
                pass
            
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
                        
    