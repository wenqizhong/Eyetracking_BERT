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


def train_class_batch(model, samples, patch_indices, target, criterion_cls, criterion_mse):
    outputs = model(samples, patch_indices)
    loss = criterion_cls(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def calculate_metrics(output, target):
    _, pred = torch.max(output, 1)
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    probabilities = F.softmax(output, dim=1)
    pre = probabilities[:, 1].detach().cpu().numpy()
    cm = confusion_matrix(target_np, pred_np).ravel()

    if len(cm) == 4:
        tn, fp, fn, tp = cm
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

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
        it = start_steps + step
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
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
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
            _, preds = torch.max(output, dim=1)
            class_acc = (preds == targets).float().mean()
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

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, output_dir):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
        
    if not os.path.exists(f'./{output_dir}/sub'):
        os.makedirs(f'./{output_dir}/sub')
    if not os.path.exists(f'./{output_dir}/scanpath'):
        os.makedirs(f'./{output_dir}/scanpath')

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
        sub_idx = sub_idx.to(device, non_blocking=True)

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

        AUC1 = torch.tensor(AUC1).to(device)
        sensitivity1 = torch.tensor(sensitivity1).to(device)
        specificity1 = torch.tensor(specificity1).to(device)

        for idx, single_output, single_target in zip(sub_idx, output, target):
            sub = idx.item()
            output_list = single_output.tolist()
            scanpath_lable = single_target.item()

            scan_probabilities = F.softmax(single_output, dim=0)
            scan_probabilities_list = scan_probabilities.tolist()

            with open(f'{output_dir}/scanpath/{sub}_output.txt', 'a') as f:
                f.write(f'{output_list}, label: {scanpath_lable}\n')
            with open(f'{output_dir}/scanpath/{sub}_probabilities.txt', 'a') as f:
                f.write(f'{scan_probabilities_list}, label: {scanpath_lable}\n')
            with open(f'{output_dir}/scanpath/all_scanpath_output.txt', 'a') as f:
                f.write(f'{output_list}, label: {scanpath_lable}\n')
            with open(f'{output_dir}/scanpath/all_scanpath_probabilities.txt', 'a') as f:
                f.write(f'{scan_probabilities_list}, label: {scanpath_lable}\n')

            if sub not in sub_outputs:
                sub_outputs[sub] = {
                    'sum': scan_probabilities_list.copy(),
                    'all_data': [scan_probabilities_list.copy()],
                    'count': 1
                }
            else:
                sub_outputs[sub]['sum'] = [x + y for x, y in zip(sub_outputs[sub]['sum'], scan_probabilities_list)]
                sub_outputs[sub]['count'] += 1
                sub_outputs[sub]['all_data'].append(scan_probabilities_list.copy())

            for sub in sub_outputs:
                sub_outputs[sub]['average'] = [x / sub_outputs[sub]['count'] for x in sub_outputs[sub]['sum']]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        acc1_value = acc1[0].item()
        AUC1_value = AUC1.item()
        sensitivity1_value = sensitivity1.item()
        specificity1_value = specificity1.item()

        metric_logger.meters['acc1'].update(acc1_value, n=batch_size)
        metric_logger.meters['AUC1'].update(AUC1_value, n=batch_size)
        metric_logger.meters['sensitivity1'].update(sensitivity1_value, n=batch_size)
        metric_logger.meters['specificity1'].update(specificity1_value, n=batch_size)

    metric_logger.synchronize_between_processes()

    print('* Acc@1: {top1.global_avg:.3f}, loss: {losses.global_avg:.3f}\n\
    AUC@1: {auc1.global_avg:.3f}, sensitivity@1: {sen1.global_avg:.3f}\n\
    specificity@1: {spe1.global_avg:.3f}'
        .format(top1=metric_logger.acc1, losses=metric_logger.loss,\
                auc1=metric_logger.AUC1, sen1=metric_logger.sensitivity1,\
                spe1=metric_logger.specificity1))
    
    for sub, _ in sub_outputs.items():
        with open(f'{output_dir}/scanpath/{sub}_output.txt', 'a') as f:
            f.write(f'epoch{epoch}\n')
        with open(f'{output_dir}/scanpath/{sub}_probabilities.txt', 'a') as f:
            f.write(f'epoch{epoch}\n')
    with open(f'{output_dir}/scanpath/all_scanpath_output.txt', 'a') as f:
        f.write(f'epoch{epoch}\n')
    with open(f'{output_dir}/scanpath/all_scanpath_probabilities.txt', 'a') as f:
        f.write(f'epoch{epoch}\n')

    ave_sub_outputs_list = []
    sub_labels_list = []
    
    for subject, output in sub_outputs.items():
        if subject in sub_label:
            ave_sub_outputs_list.append(output['average'])
            sub_labels_list.append(sub_label[subject])

    sub_logits = {subj: logits['average'] for subj, logits in sub_outputs.items()}
    with open(f'{output_dir}/sub/sub_output_avg.txt', 'a') as f:
        f.write(f'epoch{epoch}\n')
        for subj in sub_outputs:
            logits = sub_logits[subj]
            label = sub_label[subj]
            f.write(f'{subj}: {logits}, label: {label}\n')

    outputs_tensor = torch.tensor(ave_sub_outputs_list)
    labels_tensor = torch.tensor(sub_labels_list)
    
    AUC2, sensitivity2, specificity2 = calculate_metrics(outputs_tensor, labels_tensor)
    
    sub_pre = {sub: torch.tensor(probs['average']).argmax().item() for sub, probs in sub_outputs.items()}
    correct = 0
    total = len(sub_pre)
    with open(f'{output_dir}/sub/sub_evaluation_results.txt', 'a') as file:
        file.write(f'epoch{epoch}\n')
        for key in sub_pre:
            if key in sub_label:
                if sub_pre[key] == sub_label[key]:
                    correct += 1
                    file.write(f"sub {key}: Correct\n")
                else:
                    file.write(f"sub {key}: Incorrect\n")
            else:
                total -= 1

        acc2 = correct / total if total > 0 else 0
        print(f"Accuracy for all subs: {acc2*100:.2f}%")
        file.write(f"Accuracy for all subs: {acc2*100:.2f}%\n")
        file.write(f"AUC: {AUC2}\nsensitivity: {sensitivity2}\nspecificity: {specificity2}\n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
