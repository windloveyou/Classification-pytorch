import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr

# 注意这是对于一个epoch的，一个epoch需要跑完所有的样本，而一个epoch中含batch_size，跑完一个batch_size就代表了一个iteration
# 即一个epoch要跑n个iteration，一个iteration内含batch_size个样本
# 在这里iteration就是epoch_step
def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    # gen是一个迭代器的对象
    # 它的组成为元组(batch_size, imgs, targets)，每一个gen对象中包含的是一个批次内的图片及其对应标签
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step() # 针对一个批次的持续优化
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            # acc = 预测正确的数目 / 总数目
            # .argmax(dim) 获取输入维度dim下最大值的下标
            # 对输出的最后一维（类别维度）进行softmax，转换到一个概率分布上[0, 1]
            # 对类别维度取最大值索引->模型预测出的类别
            # 将模型预测出的类别（对应下标索引号）与真实值进行对比，相同为True，不同为False，使用type(torch.FloatTensor)将True转为1.0，False转为0.0的浮点型张量
            
            
            total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),         # 平均损失 = 损失值 / steps
                                'accuracy'  : total_accuracy / (iteration + 1),     # 平均准确率 = 预测正确的数量 / 总个数
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs     = model_train(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            
            val_loss    += loss_value.item()
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))