# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable, Optional
import os
import numpy as np


import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import torch.nn.functional as F


import utils


# save_result_dir = './save_result_test'

def train_class_batch(model, samples, patch_indices, target, criterion_cls, criterion_mse):
    # 计算注视点对应的块索引
    outputs = model(samples, patch_indices)

    loss = criterion_cls(outputs, target)
    # loss_cls = criterion_cls(outputs, target)
    # loss_mse = criterion_mse(outputs[1], outputs[2])
    # loss = 0.5*loss_cls + 0.5*loss_mse
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def calculate_metrics(output, target):
    # 将输出概率转换为类别预测
    _, pred = torch.max(output, 1)
    
    # 将tensor转换为numpy数组来使用sklearn的函数
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # 应用softmax函数将logits转换成概率
    probabilities = F.softmax(output, dim=1)
    # print(probabilities)
    # 使用正类的概率计算AUC
    # ASD的索引为1，为正类
    pre = probabilities[:, 1].detach().cpu().numpy()

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(target_np, pred_np).ravel()

    # 计算sensitivity (recall)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

    # 计算specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # 检查target中是否存在多于一个类
    if len(np.unique(target_np)) == 1:
        AUC = 0
    else:
        AUC = roc_auc_score(target_np, pre)


    return AUC, sensitivity, specificity

def train_one_epoch(model: torch.nn.Module, criterion_cls: torch.nn.Module, criterion_mse: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

   
    for data_iter_step, (samples, patch_indices, targets, sub_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        

        indices = []
        for indice in patch_indices:
            indice = indice.to(device, non_blocking=True)
            indices.append(indice)

        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, indices, targets = mixup_fn(samples, indices, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion_cls, criterion_mse)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, indices, targets, criterion_cls, criterion_mse)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            # 获取得分最高的类别索引
            _, preds = torch.max(output, dim=1)

            # 计算准确率
            class_acc = (preds == targets).float().mean()
            # class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)


        
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



#换新的数据集时验证集受试者ASD和TD中序号不能有重复，记得更改
@torch.no_grad()
def evaluate(data_loader, model, device, epoch, output_dir):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    # switch to evaluation mode
    model.eval()

    sub_outputs = {}
    sub_label = {}
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        patch_indices = batch[1]
        target = batch[2]
        sub_idx = batch[-1]
        images = images.to(device, non_blocking=True)
        indices = []
        for indice in patch_indices:
            indice = indice.to(device, non_blocking=True)
            indices.append(indice)
        target = target.to(device, non_blocking=True)
        # print(sub_idx)
        sub_idx = sub_idx.to(device, non_blocking=True)



        # 统计受试者标签，将当前批次的数据添加到sub_label字典中
        # 只添加那些还没有出现过的受试者ID
        for subject, label in zip(sub_idx, target):
            subject = subject.item()
            label = label.item()
            if subject not in sub_label:
                sub_label[subject] = label
        
        with torch.cuda.amp.autocast():
            output = model(images, indices)
            loss = criterion(output, target)
            
            
        acc1 = accuracy(output, target, topk=(1,))
        AUC1, sensitivity1, specificity1 = calculate_metrics(output, target)
      

        AUC1 = torch.tensor(AUC1)
        sensitivity1 = torch.tensor(sensitivity1)
        specificity1 = torch.tensor(specificity1)     
        AUC1 = AUC1.to(device)
        sensitivity1 = sensitivity1.to(device)
        specificity1 = specificity1.to(device)


            
        for idx, single_output in zip(sub_idx, output):
            sub = idx.item()
            output_list = single_output.tolist()
            # 检查sub是否已经在字典中
            if sub not in sub_outputs:
                # 如果不在，初始化这个sub的累加列表
                sub_outputs[sub] = output_list
            else:
                # 如果已经存在，对应元素相加
                sub_outputs[sub] = [x + y for x, y in zip(sub_outputs[sub], output_list)]


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # 如果 acc1 是一个包含单个张量的列表
        acc1_tensor = acc1[0]  # 获取列表中的第一个元素（张量）
        acc1_value = acc1_tensor.item()  # 从张量中提取数值

        AUC1_value = AUC1.item()  # 从张量中提取数值

        sensitivity1_value = sensitivity1.item()  # 从张量中提取数值

        specificity1_value = specificity1.item()  # 从张量中提取数值


        # 更新 metric_logger
        metric_logger.meters['acc1'].update(acc1_value, n=batch_size)
        metric_logger.meters['AUC1'].update(AUC1_value, n=batch_size)
        metric_logger.meters['sensitivity1'].update(sensitivity1_value, n=batch_size)
        metric_logger.meters['specificity1'].update(specificity1_value, n=batch_size)
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # 同步所有进程的统计数据
    metric_logger.synchronize_between_processes()


    # 打印并返回统计数据
    print('* Acc@1: {top1.global_avg:.3f}, loss: {losses.global_avg:.3f}\n\
    AUC@1: {auc1.global_avg:.3f}, sensitivity@1: {sen1.global_avg:.3f}\n\
    specificity@1: {spe1.global_avg:.3f}'
        .format(top1=metric_logger.acc1, losses=metric_logger.loss,\
                auc1=metric_logger.AUC1, sen1=metric_logger.sensitivity1,\
                spe1=metric_logger.specificity1))
    

    # 初始化输出和标签列表
    sub_outputs_list = []
    sub_labels_list = []
    
    # 遍历所有受试者的输出
    for subject, output in sub_outputs.items():
        if subject in sub_label:
            # 将模型输出和实际标签添加到列表中
            sub_outputs_list.append(output)
            sub_labels_list.append(sub_label[subject])

    #统计受试者的输出的概率，方便画ROC图像      
    sub_probabilities = {subj: F.softmax(torch.tensor(logits), dim=0).tolist() for subj, logits in sub_outputs.items()}
    # 写入文件
    with open(f'{output_dir}/probabilities.txt', 'a') as f:
        f.write(f'epoch{epoch}\n')
        for subj in sub_outputs:
            probabilities = sub_probabilities[subj]
            label = sub_label[subj]
            f.write(f'{subj}: {probabilities}, label: {label}\n')


    # 转换列表为Tensors
    outputs_tensor = torch.tensor(sub_outputs_list)
    labels_tensor = torch.tensor(sub_labels_list)
    
    # 计算总体指标
    AUC2, sensitivity2, specificity2 = calculate_metrics(outputs_tensor, labels_tensor)
    

    #计算受试者为单位准确率
    sub_pre = {sub: torch.tensor(probs).argmax().item() for sub, probs in sub_outputs.items()}
    correct = 0
    total = len(sub_pre)
    with open(f'{output_dir}/evaluation_results.txt', 'a') as file:
        file.write(f'epoch{epoch}\n')
        for key in sub_pre:
            if key in sub_label:
                if sub_pre[key] == sub_label[key]:
                    correct += 1
                    file.write(f"sub {key}: Correct\n")
                else:
                    file.write(f"sub {key}: Incorrect\n")
            else:
                total -= 1  # 如果某个键在ground_truth中不存在，则减少总数

        acc2 = correct / total if total > 0 else 0
        print(f"Accuracy for all subs: {acc2*100:.2f}%")
        file.write(f"Accuracy for all subs: {acc2*100:.2f}%\n")
        file.write(f"AUC: {AUC2}\nsensitivity: {sensitivity2}\nspecificity: {specificity2}\n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

