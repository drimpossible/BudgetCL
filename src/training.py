import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

from utils import AverageMeter, ProgressMeter, Summary, accuracy


def train(opt,
          loader,
          model,
          optimizer,
          scheduler,
          logger,
          prevmodel=None,
          temperature=2.0):
    """
    Trains a PyTorch model on a given dataset.

    Args:
        opt (argparse.Namespace): The command-line arguments.
        loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        model (torch.nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        logger (Logger): The logger to use for logging training progress.
        prevmodel (torch.nn.Module, optional): The previous model to use for distillation.
        temperature (float, optional): The temperature to use for distillation.

    Returns:
        torch.nn.Module: The trained PyTorch model.
        torch.optim.Optimizer: The optimizer used for training.
        torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler used for training.
    """

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(logger,
                             len(loader), [losses, top1, top5],
                             prefix="Timestep: [{}]".format(opt.timestep))

    # Switch to train mode
    model.train()
    if prevmodel is not None:
        prevmodel.eval()
    distill_target = None

    for i, (images, target) in enumerate(loader):
        # measure data loading time

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            # compute output
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, target)

            # Distillation of different types -- lambda is hyperparam-searched over
            if opt.distill is not None and prevmodel is not None:
                with torch.no_grad():
                    prevoutput = prevmodel(images)
                    prevoutput = prevoutput.detach() / temperature

                distill_output = output[:, :prevoutput.size(1)] / temperature

                if opt.distill == 'BCE':
                    # Ref: iCaRL Incremental Classifier and Representation Learning (https://arxiv.org/abs/1611.07725)
                    prevprobs = F.softmax(prevoutput, dim=1)
                    loss += 1.0 * F.binary_cross_entropy_with_logits(
                        input=distill_output, target=prevprobs)
                elif opt.distill == 'CrossEntropy':
                    # Ref: Large Scale Incremental Learning (https://arxiv.org/abs/1905.13260)
                    prevprobs = F.softmax(prevoutput, dim=1)
                    log_inp = F.log_softmax(distill_output, dim=1)
                    loss += prevoutput.size(1) / (output.size(1)) * F.kl_div(
                        input=log_inp, target=prevprobs)
                elif opt.distill == 'Cosine':
                    if distill_target is None:
                        distill_target = torch.ones(prevoutput.shape[0]).cuda()
                    # Ref: Learning a Unified Classifier Incrementally via Rebalancing (http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf)
                    loss += max(
                        0.5,
                        np.sqrt((output.size(1) - prevoutput.size(1)) /
                                prevoutput.size(1))) * F.cosine_embedding_loss(
                                    input1=distill_output,
                                    input2=prevoutput,
                                    target=distill_target)
                elif opt.distill == 'MSE':
                    # Ref: Dark Experience for General Continual Learning: A Strong, Simple Baseline (https://arxiv.org/pdf/2004.07211.pdf)
                    loss += 0.5 * F.mse_loss(input=distill_output,
                                             target=prevoutput)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if opt.calibrator == 'WA':
                torch.clamp(model.fc.weight, min=0)

            if i > ((opt.total_steps)):
                return model, optimizer, scheduler
    return model, optimizer, scheduler


def validate(loader, model, logger):
    """
    Evaluates a PyTorch model on a given dataset.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        model (torch.nn.Module): The PyTorch model to evaluate.
        logger (Logger): The logger to use for logging evaluation progress.

    Returns:
        None
    """
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(logger,
                             len(loader), [top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    progress.display_summary()
    return


def test(loader, model, logger):
    """
    Evaluates a PyTorch model on a given dataset.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        model (torch.nn.Module): The PyTorch model to evaluate.
        logger (Logger): The logger to use for logging evaluation progress.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The predicted labels and true labels for each sample.
    """
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(logger, len(loader), [top1, top5], prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    predsarr, labelsarr = None, None
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)

            # compute output
            output = model(images)
            output = output.cpu()
            pred = torch.argmax(output, dim=1)
            loss = nn.CrossEntropyLoss()(output, target)

            if predsarr is None:
                labelsarr = target
                predsarr = pred
            else:
                labelsarr = torch.cat((labelsarr, target), dim=0)
                predsarr = torch.cat((predsarr, pred), dim=0)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    progress.display_summary()
    return predsarr, labelsarr


def save_representations(opt, loader, model, mode='features'):
    """
    Saves the representations of a PyTorch model on a given dataset.

    Args:
        opt (argparse.Namespace): The command-line arguments.
        loader (torch.utils.data.DataLoader): The data loader for the dataset.
        model (torch.nn.Module): The PyTorch model to extract features from.
        mode (str, optional): The mode to use for feature extraction. Can be 'features' or 'predictions'.

    Returns:
        None
    """
    assert (mode in ['features', 'predictions'])
    X_arr, y_arr = None, None

    node = 'flatten' if mode == 'features' else 'fc'
    new_model = create_feature_extractor(model, return_nodes=[node])
    new_model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            out = new_model(images)

            X = out[node]
            X = X.cpu()
            X_arr = torch.cat((X_arr, X), dim=0) if X_arr is not None else X
            y_arr = torch.cat(
                (y_arr, labels), dim=0) if y_arr is not None else labels

    np.save(
        opt.log_dir + '/' + opt.exp_name + '/labels_' + str(opt.timestep) +
        '_train.npy', y_arr.numpy())

    if mode == 'features':
        np.save(
            opt.log_dir + '/' + opt.exp_name + '/feats_' + str(opt.timestep) +
            '_train.npy', X_arr.numpy())

        if opt.timestep > 2:  # Optimize storage space
            if os.path.exists(opt.log_dir + '/' + opt.exp_name + '/feats_' +
                              str(opt.timestep - 2) + '_train.npy'):
                os.remove(opt.log_dir + '/' + opt.exp_name + '/feats_' +
                          str(opt.timestep - 2) + '_train.npy')
    else:
        np.save(
            opt.log_dir + '/' + opt.exp_name + '/predprobs_' +
            str(opt.timestep) + '_train.npy', X_arr.numpy())

        if opt.timestep > 2:  # Optimize scratch storage space
            if os.path.exists(opt.log_dir + '/' + opt.exp_name +
                              '/predprobs_' + str(opt.timestep - 2) +
                              '_train.npy'):
                os.remove(opt.log_dir + '/' + opt.exp_name + '/predprobs_' +
                          str(opt.timestep - 2) + '_train.npy')
